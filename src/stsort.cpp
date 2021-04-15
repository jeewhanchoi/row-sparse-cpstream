#include "stsort.hpp"

#include <algorithm>

#include <omp.h>

// intel parallel stl includes:
#if __INTEL_COMPILER
#include "pstl/algorithm"
#include "pstl/execution"
#endif


inline int ceil_log2(unsigned long long x)
{
  static const unsigned long long t[6] = {
    0xFFFFFFFF00000000ull,
    0x00000000FFFF0000ull,
    0x000000000000FF00ull,
    0x00000000000000F0ull,
    0x000000000000000Cull,
    0x0000000000000002ull
  };

  int y = (((x & (x - 1)) == 0) ? 0 : 1);
  int j = 32;
  int i;

  for (i = 0; i < 6; i++) {
    int k = (((x & t[i]) == 0) ? 0 : j);
    y += k;
    x >>= k;
    j >>= 1;
  }

  return y;
}

// most naive histogram prefix sum (columnwise with wrap around)
void colwise_prefix(idx_t* hist, idx_t n, idx_t nrows) {
    // expects a matrix of p*n

    //int nthreads = splatt_omp_get_num_threads();
    //int tid = splatt_omp_get_thread_num();

    idx_t* sums;
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int p = omp_get_num_threads();

        #pragma omp single
        {
            //sums = (idx_t*)malloc(nthreads*64);
            sums = (idx_t*)malloc(p*sizeof(idx_t));
        }

        idx_t local_sum = 0;

        // TODO: block by cachelines
        // TODO: figure out if we can use horizontal adds for this
        // TODO; use 32 bit impl. if total nnz (or # buckets) is < 2^31
        // TODO: or even 16 bit impl if small enough ...
        idx_t np = (n+p-1) / p;
        idx_t tbegin = tid*np;
        idx_t tend = std::min<idx_t>((tid+1)*np, n);

        for (idx_t j = tbegin; j < tend; ++j) {
          for (idx_t i = 0; i < nrows; ++i) {
            const idx_t idx = i*n + j;
            idx_t h = hist[idx];
            hist[idx] = local_sum;
            local_sum += h;
          }
        }

        // TODO: use cachelines per thread (false sharing!)
        //*(idx_t*)(((char*)sums) + 64*tid) = local_sum;
        sums[tid] = local_sum;
        #pragma omp barrier

        idx_t local_offset = 0;
        for (int i = 0; i < tid; ++i) {
            local_offset += sums[i];
        }

        for (idx_t j = tbegin; j < tend; ++j) {
          for (idx_t i = 0; i < nrows; ++i) {
            const idx_t idx = i*n + j;
            hist[idx] += local_offset;
          }
        }
    }
}


idx_t* tt_bucket_sort(sptensor_t * const tt, const idx_t mode, bool coarsen, idx_t& num_bins) {
  // TODO: chunking/coarsening of histogram bins 
#if 0
  const idx_t I = tt->dims[mode];

  // TODO: case if < 16*p*64 elements
  const idx_t approx_num_chunks = 16*omp_get_num_threads();
  const idx_t chunk_size_log = max(ceil_log2(I / approx_num_chunks), 4);
  const idx_t chunk_size = 1 << chunk_size_log;
  const idx_t num_chunks = (I + (chunk_size - 1)) / chunk_size;
#endif
  const idx_t I = tt->dims[mode];
  idx_t num_chunks = I;
  idx_t chunk_size_log = 0;


  // set max number of chunks depending on number of threads
  // TODO: do this dynamically based on load balance and sparseness etc
  int nthreads = omp_get_max_threads();

  if (coarsen) {
    while (num_chunks > 128*nthreads) {
      // pre-coarsening
      chunk_size_log++;
      num_chunks >>= 1;
    }
    const idx_t chunk_size =  1 << chunk_size_log;
    num_chunks = (I + (chunk_size - 1)) / chunk_size;
  }
  num_bins = num_chunks;


  // parallel bucketing into the num_chunks buckets
  idx_t* hists;
  {
    timed_section ts("par_hist");
    hists = my_par_hist(tt->ind[mode], tt->nnz, num_chunks, chunk_size_log);
  }


  // check if last row of `hists` is same as total prefix hist


  // apply permutation (ie, sort the tensor data)
  idx_t ** new_ind = (idx_t**) splatt_malloc(tt->nmodes*sizeof(idx_t*));
  for (int m = 0; m < tt->nmodes; ++m) {
    new_ind[m] = (idx_t*) splatt_malloc(tt->nnz*sizeof(idx_t));
  }
  val_t* new_vals = (val_t*) splatt_malloc(tt->nnz*sizeof(val_t));


  {
    timed_section ts("par-rearrange tt");
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int p = omp_get_num_threads();

    idx_t* my_hist = hists + num_chunks*tid;

    idx_t np = (tt->nnz + p - 1) / p;
    idx_t tstart = tid*np;
    idx_t tend = std::min<idx_t>((tid+1)*np, tt->nnz);

    for (idx_t i = tstart; i < tend; ++i) {
      const idx_t hidx = (tt->ind[mode][i] >> chunk_size_log);
      const idx_t oidx = my_hist[hidx];

      new_vals[oidx] = tt->vals[i];
      for (int m = 0; m < tt->nmodes; ++m) {
        new_ind[m][oidx] = tt->ind[m][i];
      }

      ++my_hist[hidx];
    }
  }
  }


  // replace tensor data with sorted data
  for (int m = 0; m < tt->nmodes; ++m) {
    splatt_free(tt->ind[m]);
    tt->ind[m] = new_ind[m];
  }
  splatt_free(new_ind);
  splatt_free(tt->vals);
  tt->vals = new_vals;

  return hists;
}


// tensor std::sort
void tensor_stdsort_inplace(sptensor_t* const tt, idx_t mode) {
  timed_section ts("tensor_stdsort");
  {
    timed_section t("create idx");
    std::vector<size_t> idx(tt->nnz);
    for (size_t i = 0; i < tt->nnz; ++i) {
      idx[i] = i;
    }

    t.new_section("sort idx");
#if __INTEL_COMPILER
    std::sort(std::execution::par, idx.begin(), idx.end(), [&](size_t x, size_t y) {
        return (tt->ind[mode][x] < tt->ind[mode][y]);});
#else
    // TODO: non intel parallel sort
    std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
        return (tt->ind[mode][x] < tt->ind[mode][y]);});
#endif

    t.new_section("permuted tt.ind");
    {
    std::vector<idx_t> tmp(tt->nnz);
    for (int m = 0; m < tt->nmodes; ++m) {
      std::copy(tt->ind[m],tt->ind[m] + tt->nnz, tmp.begin());
      for (idx_t i = 0; i < tt->nnz; ++i) {
        tt->ind[m][i] = tmp[idx[i]];
      }
    }
    }

    t.new_section("permuted tt.vals");
    {
    std::vector<val_t> tmp(tt->nnz);
    std::copy(tt->vals,tt->vals + tt->nnz, tmp.begin());
    for (idx_t i = 0; i < tt->nnz; ++i) {
      tt->vals[i] = tmp[idx[i]];
    }
    }
  }
}

// tensor std::sort
void tensor_stdsort(const sptensor_t* const tt, idx_t mode, sptensor_t** out_tt) {
  timed_section ts("tensor_stdsort");
  {
    timed_section t("create idx");
    std::vector<size_t> idx(tt->nnz);
    for (size_t i = 0; i < tt->nnz; ++i) {
      idx[i] = i;
    }

    t.new_section("sort idx");
#if __INTEL_COMPILER
    std::sort(std::execution::par, idx.begin(), idx.end(), [&](size_t x, size_t y) {
        return (tt->ind[mode][x] < tt->ind[mode][y]);});
#else
    // TODO: non-intel parallel sort
    std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
        return (tt->ind[mode][x] < tt->ind[mode][y]);});
#endif

    t.new_section("alloc new tensor");
    sptensor_t* stt = tt_alloc(tt->nnz, tt->nmodes);

    t.new_section("permuted tt.ind");
    for (int m = 0; m < tt->nmodes; ++m) {
      for (idx_t i = 0; i < tt->nnz; ++i) {
        stt->ind[m][i] = tt->ind[m][idx[i]];
      }
    }

    t.new_section("permuted tt.vals");
    for (idx_t i = 0; i < tt->nnz; ++i) {
      stt->vals[i] = tt->vals[idx[i]];
    }
    *out_tt = stt;
  }
}

