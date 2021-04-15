#include "stmttkrp.hpp"

#include "cpptimer.hpp"
#include "stsort.hpp"

#include <vector>
#include <algorithm>

// intel parallel stl includes:
#if __INTEL_COMPILER
#include "pstl/algorithm"
#include "pstl/execution"
#include "tbb/task_scheduler_init.h"
#endif

#ifdef SPLATT_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

extern "C" {
#include "util.h"
}

inline void my_matmul(
  matrix_t const * const A,
  bool transA,
  matrix_t const * const B,
  bool transB,
  matrix_t  * const C, val_t beta = 0.) {
    splatt_blas_int const M = transA ? A->J : A->I;
    splatt_blas_int const N = transB ? B->I : B->J;
    splatt_blas_int const K = transA ? A->I : A->J;
    splatt_blas_int const LDA = A->J;
    splatt_blas_int const LDB = B->J;
    splatt_blas_int const LDC = N;

    assert(K == (splatt_blas_int)(transB ? B->J : B->I));
    assert((splatt_blas_int)(C->I * C->J) <= M*N);

    /* TODO precision! (double vs not) */
    cblas_dgemm(
        CblasRowMajor,
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        M, N, K,
        1.,
        A->vals, LDA,
        B->vals, LDB,
        beta,
        C->vals, LDC);
}

matrix_t* mttkrp_seq(sptensor_t * const tt, matrix_t ** mats, const idx_t mode) {
  idx_t const I = tt->dims[mode];
  idx_t const nfactors = mats[0]->J;
  matrix_t* M = mat_alloc(I, nfactors);
  val_t * const outmat = M->vals;


  idx_t const nmodes = tt->nmodes;

  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[m]->vals;
  }

  val_t const * const restrict vals = tt->vals;

  for(idx_t x=0; x < I * nfactors; ++x) {
    outmat[x] = 0.;
  }

  val_t * restrict accum = (val_t*) splatt_malloc(nfactors * sizeof(*accum));

  /* stream through nnz */
  for(idx_t n=0; n < tt->nnz; ++n) {
    /* initialize with value */
    for(idx_t f=0; f < nfactors; ++f) {
      accum[f] = vals[n];
    }

    for(idx_t m=0; m < nmodes; ++m) {
      if(m == mode) {
        continue;
      }

      assert(tt->ind[m][n] < mats[m]->I);
      val_t const * const restrict inrow = mvals[m] + (tt->ind[m][n] * nfactors);
      for(idx_t f=0; f < nfactors; ++f) {
        accum[f] *= inrow[f];
      }
    }

    /* write to output */
    idx_t const out_ind = tt->ind[mode][n];
    val_t * const restrict outrow = outmat + (out_ind * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      outrow[f] += accum[f];
    }
  }
  splatt_free(accum);

  return M;
}

matrix_t* mttkrp_stream_sort_bin(sptensor_t * const tt, matrix_t ** mats, const idx_t mode) {
  idx_t const I = tt->dims[mode];
  idx_t const nfactors = mats[0]->J;

  matrix_t* M = mat_alloc(I, nfactors);
  val_t * restrict const outmat = M->vals;

  idx_t num_bins;

  idx_t* hists;
  { timed_section t("tt_bucket_sort");
    hists = tt_bucket_sort(tt, mode, true, num_bins);
  }

  /* clear output */
  #pragma omp parallel for schedule(static)
  for(idx_t x=0; x < I * nfactors; ++x) {
    outmat[x] = 0.;
  }

  idx_t const nmodes = tt->nmodes;

  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[m]->vals;
  }

  val_t const * const restrict vals = tt->vals;

  int nthreads = omp_get_max_threads();
  idx_t* restrict const hist = hists + nthreads*num_bins;

  #pragma omp parallel
  {
    val_t * restrict accum = (val_t*) splatt_malloc(nfactors * sizeof(val_t));
    val_t * restrict row_accum = (val_t*) splatt_malloc(nfactors * sizeof(val_t));

    /* stream through buckets of nnz */
    #pragma omp for schedule(static)
    for(idx_t hi = 0; hi < num_bins; ++hi) {
      idx_t start = hi == 0 ? 0 : hist[hi-1];
      idx_t end = hist[hi];
      //memset(row_accum, 0, nfactors*sizeof(val_t));

      for (idx_t i = start; i < end; ++i) {
        /* initialize with value */
        for(idx_t f=0; f < nfactors; ++f) {
          accum[f] = vals[i];
        }

        for(idx_t m=0; m < nmodes; ++m) {
          if(m == mode) {
            continue;
          }

          assert(tt->ind[m][i] < mats[m]->I);
          val_t const * const restrict inrow = mvals[m] + (tt->ind[m][i] * nfactors);
          for(idx_t f=0; f < nfactors; ++f) {
            accum[f] *= inrow[f];
          }
        }

        idx_t oidx = tt->ind[mode][i];
        val_t * const restrict outrow = outmat + (oidx * nfactors);
        for (idx_t f = 0; f < nfactors; ++f) {
          outrow[f] += accum[f];
        }
      }
    }

    splatt_free(accum);
    splatt_free(row_accum);
  } /* end omp parallel */
  splatt_free(hists);
  return M;
}

#define myassert(X) if (!(X)) { std::cerr << "ERROR: Assertion '" #X "' failed!" << std::endl; }
#define PV(X) do { std::cerr << "[DEBUG] " #X " = " << X << std::endl; } while (0);

matrix_t* mttkrp_stream_sort(const sptensor_t * tt, matrix_t ** mats, const idx_t mode) {
  timed_section t("mttkrp_stream_sort setup");
  idx_t const I = tt->dims[mode];
  idx_t const nfactors = mats[0]->J;

  matrix_t* M = mat_alloc(I, nfactors);
  val_t * const outmat = M->vals;

  if (I < tt->nnz) {
    // use counting sort -> some opportunity for re-use guaranteed
  } else {
    // sparisfiy (range ? or CSR type?)
    // then estimate reuse opportunity
  }


  // TODO: other sorting algos
  // TODO: counting sort (as performed atm) becomes super inefficient when I >> nnz (which is the case here)
  //       instead we shall sort the nzz via sort(ind[mode]) (this allows creating a compressed histogram)
  //       or index sort of ind[mode]
  //       TODO: test the different sorting methods
  //       TODO: is the mttkrp step at least faster then the stream version!??
  //             if so, how much? (ie, how much do we have to improve sorting to make it viable?)
  //      TODO: ie, in the sparse case: build a type of CSR representation on the fly (sort & unique count (or prefix) per item)
  /*
  timed_section t("tt_bucket_sort");
  idx_t num_bins;
  idx_t* hists = tt_bucket_sort(tt, mode, false, num_bins);
  idx_t* restrict const hist = hists + nthreads*num_bins;
  */
  t.new_section("sort tt");
  sptensor_t* sorted_tt;
  tensor_stdsort(tt, mode, &sorted_tt);
  tt = sorted_tt;

  t.new_section("find num unique");
  idx_t num_unique = 1;
  for (idx_t i = 1; i < tt->nnz; ++i) {
    if (tt->ind[mode][i] != tt->ind[mode][i-1])
      ++num_unique;
  }

  // create hist from sorted order
  t.new_section("make hist");
  idx_t* hists = (idx_t*) splatt_malloc((num_unique+1)*sizeof(idx_t));
  idx_t num_bins = num_unique;
  idx_t c = 0;
  hists[0] = 0;
  hists[num_bins] = tt->nnz;
  for (idx_t i = 1; i < tt->nnz; ++i) {
    if (tt->ind[mode][i] != tt->ind[mode][i-1]) {
      ++c;
      hists[c] = i;
    }
  }
  idx_t* restrict const hist = hists + 1;


  t.new_section("prep mttkrp");

  /* clear output */
  #pragma omp parallel for schedule(static)
  for(idx_t x=0; x < I * nfactors; ++x) {
    outmat[x] = 0.;
  }

  idx_t const nmodes = tt->nmodes;

  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[m]->vals;
  }

  val_t const * const restrict vals = tt->vals;


  t.new_section("mttkrp par loop");
  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    val_t * restrict accum = (val_t*) splatt_malloc(nfactors * sizeof(val_t));
    val_t * restrict row_accum = (val_t*) splatt_malloc(nfactors * sizeof(val_t));

    /* stream through buckets of nnz */
    #pragma omp for schedule(static,1)
    for(idx_t hi = 0; hi < num_bins; ++hi) {
      idx_t start = hi == 0 ? 0 : hist[hi-1];
      idx_t end = hist[hi];
      if (start == end)
        continue;

      memset(row_accum, 0, nfactors*sizeof(val_t));

      idx_t oidx = tt->ind[mode][start];
      for (idx_t i = start; i < end; ++i) {
        /* initialize with value */
        for(idx_t f=0; f < nfactors; ++f) {
          accum[f] = vals[i];
        }

        for(idx_t m=0; m < nmodes; ++m) {
          if(m == mode) {
            continue;
          }

          val_t const * const restrict inrow = mvals[m] + (tt->ind[m][i] * nfactors);
          for(idx_t f=0; f < nfactors; ++f) {
            accum[f] *= inrow[f];
          }
        }

        for (idx_t f=0; f < nfactors; ++f) {
          row_accum[f] += accum[f];
        }
      }

      val_t * const restrict outrow = outmat + (oidx * nfactors);
      for (idx_t f = 0; f < nfactors; ++f) {
        outrow[f] = row_accum[f];
      }
    }

    splatt_free(accum);
    splatt_free(row_accum);
  } /* end omp parallel */
  splatt_free(hists);
  tt_free(sorted_tt);

  return M;
}

void seq_idxsort_hist(const sptensor_t * const tt, const idx_t mode, std::vector<size_t>& idx, std::vector<size_t>& buckets) {
  // arg sort the given mode
  if (idx.size() != tt->nnz) {
    idx.resize(tt->nnz);
  }
  for (size_t i = 0; i < tt->nnz; ++i) {
    idx[i] = i;
  }
  std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
      return (tt->ind[mode][x] < tt->ind[mode][y]);});

  // count unique
  idx_t num_unique = 1;
  for (size_t i = 1; i < tt->nnz; ++i) {
    if (tt->ind[mode][idx[i]] != tt->ind[mode][idx[i-1]]) {
      ++num_unique;
    }
  }

  // allocate buckets
  buckets.resize(num_unique+1);
  buckets[0] = 0;
  idx_t c = 0;
  for (size_t i = 1; i < tt->nnz; ++i) {
    if (tt->ind[mode][idx[i]] != tt->ind[mode][idx[i-1]]) {
      ++c;
      buckets[c] = i;
    }
  }
  buckets[num_unique] = tt->nnz;
}

void idxsort_hist(const sptensor_t * const tt, const idx_t mode, std::vector<size_t>& idx, std::vector<size_t>& buckets) {
  timed_section t("idxsort: init idx");
  if (idx.size() != tt->nnz) {
    idx.resize(tt->nnz);
  }

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < tt->nnz; ++i) {
    idx[i] = i;
  }

  t.new_section("idxsort: parsort idx");
#if __INTEL_COMPILER
  std::sort(std::execution::par, idx.begin(), idx.end(), [&](size_t x, size_t y) {
      return (tt->ind[mode][x] < tt->ind[mode][y]);});
#else
  // TODO: parallel sorting for non-intel
  std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
      return (tt->ind[mode][x] < tt->ind[mode][y]);});
#endif

  t.new_section("find num unique");
  idx_t* hists;
  idx_t num_bins;
  size_t* counts;
  #pragma omp parallel
 {
   int nthreads = omp_get_num_threads();
   int tid = omp_get_thread_num();
   size_t start = (tt->nnz * tid) / nthreads;
   size_t end = (tt->nnz * (tid+1)) / nthreads;


   size_t my_first = std::numeric_limits<size_t>::max();
   size_t my_last = std::numeric_limits<size_t>::max();
   size_t my_unique = 0;
   if (start == 0 && start < end) { // first thread with elements
     ++start;
     my_first = 0;
     my_unique = 1;
   }

  for (size_t i = start; i < end; ++i) {
    if (tt->ind[mode][idx[i]] != tt->ind[mode][idx[i-1]]) {
      if (my_first > i) {
        my_first = i;
      }
      ++my_unique;
      my_last = i;
    }
  }

  // get total count and prefix sum of `my_unique`
  // TODO: this impleementation is pretty ad-hoc and has a lot of potential
  // to be improved (if necessary - currently this is not even close to a
  // bottleneck)
#pragma omp single
  {
    counts = (size_t*) splatt_malloc(nthreads*sizeof(size_t));
  }
#pragma omp barrier
  counts[tid] = my_unique; // FIXME: lots of false sharing here
  // prefix
#pragma omp barrier

  size_t my_prefix = 0;
  size_t num_unique = 0;
  for (int i = 0; i < tid; ++i) {
    my_prefix += counts[i];
  }
  for (int i = 0; i < nthreads; ++i) {
    num_unique += counts[i];
  }

  // create bucket offset array from sorted order
  #pragma omp single
  {
    num_bins = num_unique;
    buckets.resize(num_bins+1);
    buckets[0] = 0;
    buckets[num_bins] = tt->nnz;
  }
  #pragma omp barrier

  if (my_first <= my_last && my_last < tt->nnz) {
    idx_t c = my_prefix;
    buckets[c] = my_first;
    for (size_t i = my_first+1; i < my_last+1; ++i) {
      if (tt->ind[mode][idx[i]] != tt->ind[mode][idx[i-1]]) {
        ++c;
        buckets[c] = i;
      }
    }
  }
 }
}


matrix_t*  mttkrp_stream_idx_sort(sptensor_t * const tt, matrix_t ** mats, const idx_t mode) {
  timed_section t("mttkrp_stream_idx_sort setup");
  idx_t const I = tt->dims[mode];
  idx_t const nfactors = mats[0]->J;

  matrix_t* M = mat_alloc(I, nfactors);
  val_t * restrict const outmat = M->vals;

  std::vector<size_t> idx;
  std::vector<size_t> buckets;
  idxsort_hist(tt, mode, idx, buckets); // XXX test this one?

  size_t* restrict const hist = buckets.data() + 1;
  size_t num_bins = buckets.size() - 1;

  t.new_section("prep mttkrp");

  /* clear output */
  #pragma omp parallel for schedule(static)
  for(idx_t x=0; x < I * nfactors; ++x) {
    outmat[x] = 0.;
  }

  idx_t const nmodes = tt->nmodes;

  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[m]->vals;
  }
  val_t const * const restrict vals = tt->vals;


  t.new_section("mttkrp par loop");
  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    val_t * restrict accum = (val_t*) splatt_malloc(nfactors * sizeof(val_t));
    val_t * restrict row_accum = (val_t*) splatt_malloc(nfactors * sizeof(val_t));

    /* stream through buckets of nnz */
    // TODO stream through nnz vs hist buckets?
    #pragma omp for schedule(static,1)
    for(idx_t hi = 0; hi < num_bins; ++hi) {
      idx_t start = hi == 0 ? 0 : hist[hi-1];
      idx_t end = hist[hi];
      if (start == end)
        continue;
      memset(row_accum, 0, nfactors*sizeof(val_t));

      idx_t oidx = tt->ind[mode][idx[start]];
      for (idx_t i = start; i < end; ++i) {
        /* initialize with value */
        for(idx_t f=0; f < nfactors; ++f) {
          accum[f] = vals[idx[i]];
        }

        for(idx_t m=0; m < nmodes; ++m) {
          if(m == mode) {
            continue;
          }

          val_t const * const restrict inrow = mvals[m] + (tt->ind[m][idx[i]] * nfactors);
          for(idx_t f=0; f < nfactors; ++f) {
            accum[f] *= inrow[f];
          }
        }

        for (idx_t f=0; f < nfactors; ++f) {
          row_accum[f] += accum[f];
        }
      }

      val_t * const restrict outrow = outmat + (oidx * nfactors);
      for (idx_t f = 0; f < nfactors; ++f) {
        outrow[f] = row_accum[f];
      }
    }

    splatt_free(accum);
    splatt_free(row_accum);
  } /* end omp parallel */

  return M;
}

// Allocate the row-sparse matrix (uninitialized)
rsp_matrix_t* rspmat_alloc(idx_t nrows, idx_t ncols, idx_t nnzr) {
  rsp_matrix_t * mat = (rsp_matrix_t*) splatt_malloc(sizeof(rsp_matrix_t));
  mat->I = nrows;
  mat->J = ncols;
  mat->nnzr = nnzr;
  mat->mat = mat_alloc(nnzr, ncols);
  mat->rowind = (idx_t*)splatt_malloc(nnzr * sizeof(idx_t));
  return mat;
}

void rspmat_free(rsp_matrix_t* mat) {
  mat_free(mat->mat);
  splatt_free(mat->rowind);
  splatt_free(mat);
}

// add reverse idx
void nonzero_slices(
    sptensor_t * const tt, const idx_t mode,
    std::vector<size_t> &nz_rows,
    std::vector<size_t> &idx,
    std::vector<int> &ridx,
    std::vector<size_t> &buckets) 
{
  idxsort_hist(tt, mode, idx, buckets);
  size_t num_bins = buckets.size() - 1;

  for (idx_t i = 0; i < num_bins; i++) {
    nz_rows.push_back(tt->ind[mode][idx[buckets[i]]]);
  }
  // Create array for reverse indices
  // We traverse through all rows i 
  // if it is a non zero row then add i to ridx array
  // if not, push value -1, which means invalid
  // For example if I = 10: [0, 1, 2, 3, 4, 5, ... 9] and non zero rows are [2, 4, 5]
  // then ridx would have [-1, -1, 0, -1, 1, 2, -1, ...]
  idx_t _ptr = 0;
  for (idx_t i = 0; i < tt->dims[mode]; i++) {
    if (nz_rows[_ptr] == i) {
      ridx.push_back(_ptr);
      _ptr++;
    } else {
      ridx.push_back(-1);
    }
  }
}


rsp_matrix_t*  rsp_mttkrp_stream_idx_sort(sptensor_t * const tt, matrix_t ** mats, const idx_t mode) {
  timed_section t("mttkrp_stream_idx_sort setup");
  idx_t const I = tt->dims[mode];
  idx_t const nfactors = mats[0]->J;

  std::vector<size_t> idx;
  std::vector<size_t> buckets;
  idxsort_hist(tt, mode, idx, buckets);

  size_t* restrict const hist = buckets.data() + 1;
  size_t num_bins = buckets.size() - 1;
  t.new_section("alloc/init output matrix");

  /* allocate and initialize the row-sparse output matrix */
  rsp_matrix_t* M = rspmat_alloc(I, nfactors, num_bins);
  val_t * restrict const outmat = M->mat->vals;
  #pragma omp parallel for schedule(static)
  for(idx_t x=0; x < num_bins * nfactors; ++x) {
    outmat[x] = 0.;
  }
  #pragma omp parallel for schedule(static)
  // Write row index for rsp matrix
  for (idx_t i = 0; i < num_bins; ++i) {
    M->rowind[i] = tt->ind[mode][idx[buckets[i]]];
  }

  idx_t const nmodes = tt->nmodes;
  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[m]->vals;
  }
  val_t const * const restrict vals = tt->vals;

  t.new_section("mttkrp par loop");
  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    val_t * restrict accum = (val_t*)splatt_malloc(nfactors * sizeof(val_t));
    val_t * restrict row_accum = (val_t*)splatt_malloc(nfactors * sizeof(val_t));

    /* stream through buckets of nnz */
    #pragma omp for schedule(static,1)
    for(idx_t hi = 0; hi < num_bins; ++hi) {
      idx_t start = hi == 0 ? 0 : hist[hi-1];
      idx_t end = hist[hi];
      if (start == end)
        continue;
      memset(row_accum, 0, nfactors*sizeof(val_t));

      idx_t oidx = hi;
      for (idx_t i = start; i < end; ++i) {
        /* initialize with value */
        for(idx_t f=0; f < nfactors; ++f) {
          accum[f] = vals[idx[i]];
        }

        for(idx_t m=0; m < nmodes; ++m) {
          if(m == mode) {
            continue;
          }

          val_t const * const restrict inrow = mvals[m] + (tt->ind[m][idx[i]] * nfactors);
          for(idx_t f=0; f < nfactors; ++f) {
            accum[f] *= inrow[f];
          }
        }

        for (idx_t f=0; f < nfactors; ++f) {
          row_accum[f] += accum[f];
        }
      }

      val_t * const restrict outrow = outmat + (oidx * nfactors);
      for (idx_t f = 0; f < nfactors; ++f) {
        outrow[f] = row_accum[f];
      }
    }

    splatt_free(accum);
    splatt_free(row_accum);
  } /* end omp parallel */

  return M;
}

rsp_matrix_t* rsp_mttkrp_stream_rsp(
  sptensor_t * const tt, 
  rsp_matrix_t ** rsp_mats,
  const idx_t mode,
  const idx_t stream_mode,
  std::vector<size_t>& idx,
  std::vector<std::vector<int>>& ridx,
  std::vector<size_t>& buckets) 
{
  idx_t const I = tt->dims[mode];
  idx_t const nfactors = rsp_mats[0]->J;

  /* If never computed, normally shouldn't be called in 
   * Hypersparse ALS because at the beginning of every time slice
   * we compute for every mode */
  if (idx.size() != tt->nnz) {
    idxsort_hist(tt, mode, idx, buckets);
  }

  size_t* restrict const hist = buckets.data() + 1;
  size_t num_bins = buckets.size() - 1;

  /* allocate and initialize the row-sparse output matrix */
  rsp_matrix_t* M = rspmat_alloc(I, nfactors, num_bins);

  val_t * restrict const outmat = M->mat->vals;

  // timer_fstart(&clear_output);
  #pragma omp parallel for schedule(static)
  for(idx_t x=0; x < num_bins * nfactors; ++x) {
    outmat[x] = 0.;
  }

  #pragma omp parallel for schedule(static)
  // Write row index for rsp matrix
  for (idx_t i = 0; i < num_bins; ++i) {
    M->rowind[i] = tt->ind[mode][idx[buckets[i]]];
  }
  
  idx_t const nmodes = tt->nmodes;
  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = rsp_mats[m]->mat->vals;
  }
  val_t const * const restrict vals = tt->vals;

  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    val_t * restrict accum = (val_t*)splatt_malloc(nfactors * sizeof(val_t));
    val_t * restrict row_accum = (val_t*)splatt_malloc(nfactors * sizeof(val_t));

    /* stream through buckets of nnz */
    #pragma omp for schedule(static,1)
    for(idx_t hi = 0; hi < num_bins; ++hi) {
      idx_t start = hi == 0 ? 0 : hist[hi-1];
      idx_t end = hist[hi];
      if (start == end)
        continue;
      memset(row_accum, 0, nfactors*sizeof(val_t));

      idx_t oidx = hi;
      for (idx_t i = start; i < end; ++i) {
        /* initialize with value */
        for(idx_t f=0; f < nfactors; ++f) {
          accum[f] = vals[idx[i]];
        }

        for(idx_t m=0; m < nmodes; ++m) {
          if(m == mode) {
            continue;
          }
          idx_t m_idx;

          // Given the 'raw' index of the nonzero we need 
          // to find the corresponding rowind from A_nz[m]
          // FYI tt->ind[m] is the array that contains the indices of non zeros in m-th mode
          if (m == stream_mode) m_idx = 0; // Because time-mode has only one row
          else {
            m_idx = ridx[m][tt->ind[m][idx[i]]];
          }

          /* We no longer need this expensive part 
          // Go through all the rowind in A_nz[m]
          for (idx_t r=0; r < rsp_mats[m]->nnzr; r++) {
            if (rsp_mats[m]->rowind[r] == tt->ind[m][idx[i]]) {
              m_idx = r;
//              printf("matched: rowind[r]: %d, tt->ind[m]: %d\n", rsp_mats[m]->rowind[r], tt->ind[m][idx[i]]);
            }
          }
          */

          val_t const * const restrict inrow = mvals[m] + (m_idx * nfactors);
          for(idx_t f=0; f < nfactors; ++f) {
            accum[f] *= inrow[f];
          }
        }

        for (idx_t f=0; f < nfactors; ++f) {
          row_accum[f] += accum[f];
        }
      }

      val_t * const restrict outrow = outmat + (oidx * nfactors);
      for (idx_t f = 0; f < nfactors; ++f) {
        outrow[f] = row_accum[f];
      }
    }

    splatt_free(accum);
    splatt_free(row_accum);
  } /* end omp parallel */
  return M;
}

/*
  std::vector<size_t> idx;
  std::vector<size_t> buckets;
  idxsort_hist(tt, mode, idx, buckets);
  */
// void idxsort_hist(const sptensor_t * const tt, const idx_t mode, std::vector<size_t>& idx, std::vector<size_t>& buckets) {

rsp_matrix_t* rsp_mttkrp_stream_with_idx(
  sptensor_t * const tt, 
  matrix_t ** mats, 
  const idx_t mode,
  std::vector<size_t>& idx,
  std::vector<size_t>& buckets) 
{
  idx_t const I = tt->dims[mode];
  idx_t const nfactors = mats[0]->J;

  /* If never computed */
  if (idx.size() != tt->nnz) {
    idxsort_hist(tt, mode, idx, buckets);
  }

  size_t* restrict const hist = buckets.data() + 1;
  size_t num_bins = buckets.size() - 1;

  /* allocate and initialize the row-sparse output matrix */
  rsp_matrix_t* M = rspmat_alloc(I, nfactors, num_bins);

  val_t * restrict const outmat = M->mat->vals;

  // timer_fstart(&clear_output);
  #pragma omp parallel for schedule(static)
  for(idx_t x=0; x < num_bins * nfactors; ++x) {
    outmat[x] = 0.;
  }

  #pragma omp parallel for schedule(static)
  // Write row index for rsp matrix
  for (idx_t i = 0; i < num_bins; ++i) {
    M->rowind[i] = tt->ind[mode][idx[buckets[i]]];
  }
  

  idx_t const nmodes = tt->nmodes;
  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[m]->vals;
  }
  val_t const * const restrict vals = tt->vals;

  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    val_t * restrict accum = (val_t*)splatt_malloc(nfactors * sizeof(val_t));
    val_t * restrict row_accum = (val_t*)splatt_malloc(nfactors * sizeof(val_t));

    /* stream through buckets of nnz */
    #pragma omp for schedule(static,1)
    for(idx_t hi = 0; hi < num_bins; ++hi) {
      idx_t start = hi == 0 ? 0 : hist[hi-1];
      idx_t end = hist[hi];
      if (start == end)
        continue;
      memset(row_accum, 0, nfactors*sizeof(val_t));

      idx_t oidx = hi;
      for (idx_t i = start; i < end; ++i) {
        /* initialize with value */
        for(idx_t f=0; f < nfactors; ++f) {
          accum[f] = vals[idx[i]];
        }

        for(idx_t m=0; m < nmodes; ++m) {
          if(m == mode) {
            continue;
          }

          val_t const * const restrict inrow = mvals[m] + (tt->ind[m][idx[i]] * nfactors);
          for(idx_t f=0; f < nfactors; ++f) {
            accum[f] *= inrow[f];
          }
        }

        for (idx_t f=0; f < nfactors; ++f) {
          row_accum[f] += accum[f];
        }
      }

      val_t * const restrict outrow = outmat + (oidx * nfactors);
      for (idx_t f = 0; f < nfactors; ++f) {
        outrow[f] = row_accum[f];
      }
    }

    splatt_free(accum);
    splatt_free(row_accum);
  } /* end omp parallel */
  return M;
}

rsp_matrix_t* rsp_mttkrp_stream(
  sptensor_t * const tt, 
  matrix_t ** mats, 
  const idx_t mode) 
{

  sp_timer_t clear_output;
  sp_timer_t write_output;
  sp_timer_t total_runtime;

  // timer_fstart(&total_runtime);

  timed_section t("mttkrp_stream_idx_sort setup");
  idx_t const I = tt->dims[mode];
  idx_t const nfactors = mats[0]->J;

  // printf("rsp_mttkrp_stream,%d,%d\n", mode, I);

  std::vector<size_t> idx;
  std::vector<size_t> buckets;
  idxsort_hist(tt, mode, idx, buckets);

  size_t* restrict const hist = buckets.data() + 1;
  size_t num_bins = buckets.size() - 1;
  t.new_section("alloc/init output matrix");

  /* allocate and initialize the row-sparse output matrix */
  rsp_matrix_t* M = rspmat_alloc(I, nfactors, num_bins);

  val_t * restrict const outmat = M->mat->vals;

  // timer_fstart(&clear_output);
  #pragma omp parallel for schedule(static)
  for(idx_t x=0; x < num_bins * nfactors; ++x) {
    outmat[x] = 0.;
  }
  // timer_stop(&clear_output);

  #pragma omp parallel for schedule(static)
  // Write row index for rsp matrix
  for (idx_t i = 0; i < num_bins; ++i) {
    M->rowind[i] = tt->ind[mode][idx[buckets[i]]];
  }
  

  idx_t const nmodes = tt->nmodes;
  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[m]->vals;
  }
  val_t const * const restrict vals = tt->vals;

  t.new_section("mttkrp par loop");
  // timer_fstart(&write_output);
  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    val_t * restrict accum = (val_t*)splatt_malloc(nfactors * sizeof(val_t));
    val_t * restrict row_accum = (val_t*)splatt_malloc(nfactors * sizeof(val_t));

    /* stream through buckets of nnz */
    #pragma omp for schedule(static,1)
    for(idx_t hi = 0; hi < num_bins; ++hi) {
      idx_t start = hi == 0 ? 0 : hist[hi-1];
      idx_t end = hist[hi];
      if (start == end)
        continue;
      memset(row_accum, 0, nfactors*sizeof(val_t));

      idx_t oidx = hi;
      for (idx_t i = start; i < end; ++i) {
        /* initialize with value */
        for(idx_t f=0; f < nfactors; ++f) {
          accum[f] = vals[idx[i]];
        }

        for(idx_t m=0; m < nmodes; ++m) {
          if(m == mode) {
            continue;
          }

          val_t const * const restrict inrow = mvals[m] + (tt->ind[m][idx[i]] * nfactors);
          for(idx_t f=0; f < nfactors; ++f) {
            accum[f] *= inrow[f];
          }
        }

        for (idx_t f=0; f < nfactors; ++f) {
          row_accum[f] += accum[f];
        }
      }

      val_t * const restrict outrow = outmat + (oidx * nfactors);
      for (idx_t f = 0; f < nfactors; ++f) {
        outrow[f] = row_accum[f];
      }
    }

    splatt_free(accum);
    splatt_free(row_accum);
  } /* end omp parallel */
  // timer_stop(&write_output);
  // timer_stop(&total_runtime);  

  // Normal matrix, we need to convert M to MM
  /*
  matrix_t * const MM = mats[MAX_NMODES];
  val_t * const outmat_MM = MM->vals;
  idx_t row_size = M->J;

  // initialize to all zeros
  #pragma omp parallel for schedule(static)
  for (idx_t x=0; x < I * nfactors; ++x) {
    outmat_MM[x] = 0.;
  }

  //overwrite rsp_matrix:M to matrix:MM
  #pragma omp parallel for schedule(static)
  for (idx_t i = 0; i < M->nnzr; ++i) {
    // get row index for all non-zero rows
    idx_t row_start_idx = M->rowind[i] * row_size;
    memcpy(
      &outmat_MM[row_start_idx], 
      &M->mat->vals[i * row_size], 
      row_size*sizeof(val_t));
  }
  */
  return M; 

  // printf(
  //   "%d, %d, %f, %f, %f, %f, %f\n", 
  //   mode, I, 
  //   clear_output.seconds * 1e9, 
  //   write_output.seconds * 1e9,
  //   total_runtime.seconds * 1e9,
  //   clear_output.seconds / total_runtime.seconds,
  //   write_output.seconds / total_runtime.seconds
  //   );  
}

rsp_matrix_t* convert_to_rspmat(
    matrix_t* fm, idx_t nnzr, idx_t* rowind)
{
  rsp_matrix_t* rspmat = rspmat_alloc(fm->I, fm->J, nnzr);
  idx_t I = fm->I;
  idx_t J = fm->J;

#pragma omp parallel for schedule(static, 1) if(nnzr > 50)
  for (idx_t i = 0; i < nnzr; i++) {
    memcpy(
        &rspmat->mat->vals[i * J],
        &fm->vals[rowind[i] * J],
        sizeof(*fm->vals) * J);
  }

  // Returned RSP matrix has same row indices as rowind
  par_memcpy(rspmat->rowind, rowind, sizeof(idx_t) * nnzr);
  return rspmat;
}

void rsp_mataTb(rsp_matrix_t* A, rsp_matrix_t* B, matrix_t* dest)
{
  assert(A->I == B->I);
  assert(A->J == B->J);
  // aTb op is used when a and b have identical row indices
  // The result gram matrix therefore doesn't have to consider the 
  // row indices of A and B
  assert(A->nnzr == B->nnzr);

  // We don't need this if we're using dgemm
  size_t _size = dest->J;
  size_t nnzr = A->nnzr;
 
  // Try using dgemm, my_matmul defined in StreamCPD.cc

  my_matmul(A->mat, true, B->mat, false, dest);

  // SLOW, non-optimized version
  /*
  for (idx_t i = 0; i < _size; i ++) {
    for (idx_t j = 0; j < _size; j ++) {
      val_t tmp = 0.0;
      for (idx_t k = 0; k < nnzr; k++) {
        tmp += A->mat->vals[k * _size + i] * B->mat->vals[k * _size + j];
      }
      dest->vals[i * _size + j] = tmp;
    }
  }
  */
}

void rsp_mat_add(
    rsp_matrix_t* A, rsp_matrix_t* B)
{
  assert(A->I == B->I);
  assert(A->J == B->J);
  assert(A->nnzr == B->nnzr);
  idx_t I = A->I;
  idx_t J = A->J;
  idx_t nnzr = A->nnzr;

  // Need to compare the row indices as well
  val_t * A_vals = A->mat->vals;
  val_t const * const B_vals = B->mat->vals;

#pragma omp parallel for schedule(static, 1) if(nnzr > 50)
  for (idx_t i = 0; i < nnzr; i++) {
    for (idx_t j = 0; j < J; j++) {
      A_vals[i * J + j] += B_vals[i * J + j];
    }
  }
}

rsp_matrix_t* rsp_mat_mul(
    rsp_matrix_t* A, matrix_t* B)
{
  idx_t nrows = A->I;
  idx_t ncols = A->J;
  idx_t nnzr = A->nnzr;

  rsp_matrix_t* C = rspmat_alloc(nrows, ncols, nnzr);

  // Use conventional matmul for A and B and write to C
  my_matmul(A->mat, false, B, false, C->mat);

  // A and C have same row indices
  par_memcpy(C->rowind, A->rowind, sizeof(idx_t) * nnzr);

  return C;
}
