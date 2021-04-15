// C includes
#include <stdlib.h>
#include <stdio.h>

// C++ includes
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <string>
#include <random>


// Splatt includes
extern "C" {
#include <splatt.h>
#include <matrix.h>
#include <mttkrp.h>
#include <timer.h>
#include <io.h>
#include <sptensor.h>
#include <sort.h>
#include <ccp/ccp.h>
#include <cpd/cpd.h>
#include <splatt_debug.h>
#include <splatt_lapack.h>
#include <base.h>
}

#include "cpptimer.hpp"
#include "stsort.hpp"
#include "stmttkrp.hpp"

#if __INTEL_COMPILER
#include "tbb/task_scheduler_init.h"
#endif

#ifdef SPLATT_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

void printdims(const sptensor_t* const tt) {
    printf("(");
    for (idx_t i = 0; i < tt->nmodes; ++i) {
        if (i > 0)
            printf(", ");
        printf("%lu", tt->dims[i]);
    }
    printf(")");
}

double density(const sptensor_t* const tt) {
    double size = 1.;
    for (idx_t i = 0; i < tt->nmodes; ++i) {
        size *= tt->dims[i];
    }
    return tt->nnz / size;
}

void print_tensor_info(const sptensor_t* const tt) {
    printf("Tensor modes: %lu\n", tt->nmodes);
    printf("Tensor dims: ");
    printdims(tt);
    printf("\n");
    printf("Tensor nnz: %lu\n", tt->nnz);
    printf("Tensor density: %f %%\n", density(tt)*100.);
}

void print_tt_tuple(const sptensor_t* const tt, idx_t idx) {
    printf("(");
    for (idx_t i = 0; i < tt->nmodes; ++i) {
        printf("%lu, ", tt->ind[i][idx]);
    }
    printf("%f)", tt->vals[idx]);
}

void print_nzs(const sptensor_t* const tt, idx_t num) {
    for (idx_t i = 0; i < num; ++i) {
        print_tt_tuple(tt, i);
        printf("\n");
    }
    printf("...\n");
}

void print_nzs_short(const sptensor_t* tt) {
    printf("Tensor ");
    printdims(tt);
    printf(": [\n");
    for (idx_t i = 0; i < 3; ++i) {
        print_tt_tuple(tt, i);
        printf("\n");
    }
    printf("...\n");
    for (idx_t i = 0; i < 3; ++i) {
        print_tt_tuple(tt, tt->nnz - 3 + i);
        printf("\n");
    }
    printf("]\n");
}


sptensor_t* tt_copy(const sptensor_t* tt) {
    sptensor_t* ctt = tt_alloc(tt->nnz, tt->nmodes);
    memcpy(ctt->vals, tt->vals, tt->nnz*sizeof(val_t));
    for (idx_t m = 0; m < tt->nmodes; ++m) {
        memcpy(ctt->ind[m], tt->ind[m], tt->nnz*sizeof(idx_t));
    }
    return ctt;
}


/* permute the sparse tensor by the given permutation of 0:nnz */
void tt_permute(sptensor_t* const tt, const idx_t* const perm) {

  // TODO implement basic version at first
  // TODO (later)
  //    - [ ] optimized and compare inplace vs non-inplace
  //    - [ ] outer-loop over modes or nnz ?

  // alloc new arrays
  val_t* new_vals = (val_t*) splatt_malloc(tt->nnz*sizeof(val_t));
  idx_t** new_ind = (idx_t**) splatt_malloc(tt->nmodes*sizeof(idx_t*));
  for (idx_t m = 0; m < tt->nmodes; ++m) {
    new_ind[m] = (idx_t*) splatt_malloc(tt->nnz*sizeof(idx_t));
  }

  // apply permutation
  for (idx_t i = 0; i < tt->nnz; ++i) {
    new_vals[i] = tt->vals[perm[i]];
    for (idx_t m = 0; m < tt->nmodes; ++m) {
      new_ind[m][i] = tt->ind[m][perm[i]];
    }
  }

  // switch buffers and clean up
  splatt_free(tt->vals);
  tt->vals = new_vals;
  for (idx_t m = 0; m < tt->nmodes; ++m) {
    splatt_free(tt->ind[m]);
    tt->ind[m] = new_ind[m];
  }
  splatt_free(new_ind);
}

sptensor_t* tt_get_slice(const sptensor_t* const tt, idx_t mode, idx_t idx) {
  // Return new tensor containing the given slice: tt[...,:,idx,:,...] of `mode`
  // 1) count nnz having `ind[mode][i] == idx` (per thread and prefix sum)
  // 2) alloc new tensor
  // 3) scan again, copying via hist into new tensor


  // 1a) count nzz
  idx_t sl_nnz = 0;
  for (idx_t i = 0; i < tt->nnz; ++i) {
    if (tt->ind[mode][i] == idx)
      ++sl_nnz;
  }
  // 1b) (par) prefix sum

  // 2) alloc new tensor
  sptensor_t * slice =  tt_alloc(sl_nnz, tt->nmodes - 1);
  for (idx_t m = 0, sm = 0; m < tt->nmodes; ++m) {
    if (m != mode)
      slice->dims[sm++] = tt->dims[m];
  }

  // 3) scan `tt` again, copy all rows with `ind[mode][i] == idx` into slice
  for (idx_t i = 0, si = 0; i < tt->nnz; ++i) {
    if (tt->ind[mode][i] == idx) {
      for (idx_t m = 0, sm = 0; m < tt->nmodes; ++m) {
        if (m != mode) {
          slice->ind[sm++][si] = tt->ind[m][i];
        }
      }
      slice->vals[si] = tt->vals[i];
      ++si;
    }
  }
  return slice;
}

val_t maxabsdiff(const matrix_t * const A, const matrix_t * const B) {
  assert(A->I == B->I && A->J == B->J);

  val_t result = 0.;
  size_t argmax = 0;
  for (size_t i = 0; i < A->I*A->J; ++i) {
      const val_t absdiff = fabs(A->vals[i] - B->vals[i]);
      if (absdiff > result) {
        result = absdiff;
        argmax = i;
      }
  }
  std::cout << "maxabsdiff(A[" << argmax << "]=" << A->vals[argmax] << ",B[" << argmax << "]=" << B->vals[argmax] << ") = " << result << std::endl;
  return result;
}

val_t maxabsdiff(const matrix_t * const A, const rsp_matrix_t * const B) {
  assert(A->I == B->I && A->J == B->J);

  val_t result = 0.;
  size_t argmax = 0;
  size_t ra = 0;
  for (size_t ri = 0; ri < B->nnzr; ++ri) {
    while (ra < B->rowind[ri]) {
      // compare A[ra,j] to 0
      for (size_t j = 0; j < A->J; ++j) {
        val_t absdiff = fabs(A->vals[ra*A->J + j]);
        if (absdiff > result) {
          result = absdiff;
          argmax = ra*A->J + j;
        }
      }
      ++ra;
    }
    // compare A[ra,j] to B[ri,j]
    for (size_t j = 0; j < A->J; ++j) {
      val_t absdiff = fabs(A->vals[ra*A->J + j] - B->mat->vals[ri*A->J + j]);
      if (absdiff > result) {
        result = absdiff;
        argmax = ra*A->J + j;
      }
    }
    ++ra;
  }

  //std::cout << "maxabsdiff(A[" << argmax << "]=" << A->vals[argmax] << ",B[" << argmax << "]=" << B->vals[argmax] << ") = " << result << std::endl;
  return result;
}

// tests the different MTTKRP against the sequential version
// for the given tensor
void test_mttkrp(sptensor_t* tt) {
    int K = 10;

    // random init of matrices
    idx_t maxdim = 0;
    matrix_t** mats = (matrix_t**)malloc((MAX_NMODES+1)*sizeof(void*));
    for (idx_t m = 0; m < tt->nmodes; ++m) {
        mats[m] = mat_rand(tt->dims[m], K);
        if (tt->dims[m] > maxdim)
            maxdim = tt->dims[m];
    }
    mats[MAX_NMODES] = mat_alloc(maxdim, K);

    for (idx_t m = 0; m < tt->nmodes; ++m) {
      matrix_t* seq_out = mttkrp_seq(tt, mats, m);
      matrix_t* sort_out = mttkrp_stream_sort(tt, mats, m);
      //matrix_t* stream_out = mttkrp_stream
      // compare
      val_t maxdiff = maxabsdiff(seq_out, sort_out);
      std::cerr << "Maxdiff(seq_out, sort_out): " << maxdiff << std::endl;

      //mttkrp_stream(tt, mats, m);
      //maxdiff = maxabsdiff(seq_out, mats[MAX_NMODES]);
      //std::cerr << "Maxdiff(seq_out, stream_out): " << maxdiff << std::endl;

      matrix_t* idxsort_out = mttkrp_stream_idx_sort(tt, mats, m);
      maxdiff = maxabsdiff(seq_out, idxsort_out);
      std::cerr << "Maxdiff(seq_out, idxsort): " << maxdiff << std::endl;

      matrix_t* binsort_out = mttkrp_stream_sort_bin(tt, mats, m);
      maxdiff = maxabsdiff(seq_out, binsort_out);
      std::cerr << "Maxdiff(seq_out, binsort): " << maxdiff << std::endl;

      rsp_matrix_t* rsp_out = rsp_mttkrp_stream_idx_sort(tt, mats, m);
      maxdiff = maxabsdiff(seq_out, rsp_out);
      std::cerr << "Maxdiff(seq_out, rsp_out): " << maxdiff << std::endl;

      mat_free(idxsort_out);
      mat_free(binsort_out);
      mat_free(seq_out);
      mat_free(sort_out);
      rspmat_free(rsp_out);
    }

    // cleanup
    for (idx_t m = 0; m < tt->nmodes; ++m) {
      mat_free(mats[m]);
    }
    mat_free(mats[MAX_NMODES]);
    free(mats);
}

// benchmark the different mttkrp 
void mtt(sptensor_t* const tt, benchmark_timer& b) {

    int reps = 5;
    int K = 10;

    idx_t maxdim = 0;
    matrix_t** mats = (matrix_t**)malloc((MAX_NMODES+1)*sizeof(void*));
    for (idx_t m = 0; m < tt->nmodes; ++m) {
        mats[m] = mat_rand(tt->dims[m], K);
        if (tt->dims[m] > maxdim)
            maxdim = tt->dims[m];
    }
    mats[MAX_NMODES] = mat_alloc(maxdim, K);

    b.set_method("mttkrp_stream_seq");
    for (int m = 0; m < tt->nmodes; ++m) {
        printf("Benchmarking mttkrp seq mode %i\n", m);
        b.set_mode(m);
        for (int i = 0; i < reps; ++i) {
            b.tic();
            timed_section t("mttkrp stream");
	    mttkrp_seq(tt, mats, m);
            b.toc();
        }
    }


    b.set_method("mttkrp_stream");
    for (idx_t m = 0; m < tt->nmodes; ++m) {
        printf("Benchmarking mttkrp_stream mode %i\n", m);
        b.set_mode(m);
        for (int i = 0; i < reps; ++i) {
            b.tic();
            timed_section t("mttkrp stream");
            mttkrp_stream(tt, mats, m);  // choose mode
            b.toc();
        }
    }

    b.set_method("mttkrp_stream_idxsort", "sort");
    for (idx_t m = 0; m < tt->nmodes; ++m) {
        printf("Benchmarking mttkrp_stream_idxsort mode %i\n", m);
        b.set_mode(m);
        for (int i = 0; i < reps; ++i) {
            b.tic();
            timed_section t("mttkrp stream idxsort");
            mttkrp_stream_idx_sort(tt, mats, m);  // choose mode
            b.toc();
        }
    }

    b.set_method("rsp_mttkrp_stream_idxsort", "idxsort");
    for (idx_t m = 0; m < tt->nmodes; ++m) {
        printf("Benchmarking rsp_mttkrp_stream_idxsort mode %i\n", m);
        b.set_mode(m);
        for (int i = 0; i < reps; ++i) {
            b.tic();
            timed_section t("rsp mttkrp idxsort");
            rsp_matrix_t* out = rsp_mttkrp_stream_idx_sort(tt, mats, m);  // choose mode
            b.toc();
            rspmat_free(out);
        }
    }

    b.set_method("mttkrp_stream_sort", "sort");
    for (idx_t m = 0; m < tt->nmodes; ++m) {
        printf("Benchmarking mttkrp_stream_sort mode %i\n", m);
        b.set_mode(m);
        for (int i = 0; i < reps; ++i) {
            b.tic();
            timed_section t("mttkrp stream sort");
            mttkrp_stream_sort(tt, mats, m);  // choose mode
            b.toc();
        }
    }

    b.set_method("mttkrp_stream_sort", "bin64");
    for (idx_t m = 0; m < tt->nmodes; ++m) {
        printf("Benchmarking mttkrp_stream_bin64 mode %i\n", m);
        b.set_mode(m);
        for (int i = 0; i < reps; ++i) {
            b.tic();
            timed_section t("mttkrp stream bin64");
            mttkrp_stream_sort_bin(tt, mats, m);  // choose mode
            b.toc();
        }
    }

    /*
    // build CSF and benchmark csf
    double * csf_opts = splatt_default_opts();
    csf_opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
    csf_opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
    csf_opts[SPLATT_OPTION_VERBOSITY] = SPLATT_VERBOSITY_NONE;
    // construct CSF
    splatt_csf * csf;
    {
      // simplify API for single calls like this
        b.set_method("csf_alloc", "onemode_densetile");
        b.set_mode(-1);
        b.tic();
        csf = splatt_csf_alloc(tt, csf_opts);
        b.toc();
    }
    // prep mttkrp_csf workspace
    splatt_mttkrp_ws * mttkrp_ws = splatt_mttkrp_alloc_ws(csf, K, csf_opts);
    splatt_global_opts const * const global_opts = splatt_alloc_global_opts();
    splatt_cpd_opts * const cpd_opts = splatt_alloc_cpd_opts();
    cpd_ws * cpd_ws = cpd_alloc_ws_empty(tt->nmodes, K, cpd_opts, global_opts);


    b.set_method("mttkrp_csf");
    for (idx_t m = 0; m < tt->nmodes; ++m) {
        printf("Benchmarking mttkrp_csf mode %i\n", m);
        b.set_mode(m);
        for (int i = 0; i < reps; ++i) {
            b.tic();
            timed_section t("mttkrp csf");
            mttkrp_csf(csf, mats, m, cpd_ws->thds, mttkrp_ws, global_opts);
            b.toc();
        }
    }
    */

    // clean up
    for (idx_t m = 0; m < tt->nmodes; ++m) {
        mat_free(mats[m]);
    }
    free(mats);
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


int main(int argc, char** argv) {
  // Test matrix multiplication
  sp_timer_t m_timer;
  sp_timer_t mm_timer;
  timer_reset(&mm_timer);

  timer_start(&mm_timer);

  /* First type of MM */
  int I = 1000 * 1000 * 1;
  int K = 10;

  matrix_t A;
  matrix_t B;

  // A : I * K
  A.I = I;
  A.J = K;
  // B : K * K
  B.I = K;
  B.J = K;

  // Dest. matrix size: I * K
  matrix_t C;
  C.I = I;
  C.J = K;

  A.vals = (val_t*) splatt_malloc(I * K * sizeof(val_t));
  B.vals = (val_t*) splatt_malloc(K * K * sizeof(val_t));
  C.vals = (val_t*) splatt_malloc(I * K * sizeof(val_t));

  for (int i = 0; i < I * K; i++) {
    A.vals[i] = 1.0;
    C.vals[i] = 0.0;
  }

  for (int i = 0; i < K * K; i++) {
    B.vals[i] = 1.0;
  }

  my_matmul(
      &A, false,
      &B, false,
      &C
  );

  timer_stop(&mm_timer);

	// print_matrix_(&C);
  printf("Time elapsed: %f\n", mm_timer.seconds);


  timer_reset(&m_timer);
  timer_start(&m_timer);

  matrix_t AA;
  matrix_t BB;

  // A : I * K
  AA.I = I;
  AA.J = K;
  // B : K * K
  BB.I = K;
  BB.J = K;

  // Dest. matrix size: I * K
  matrix_t CC;
  CC.I = I;
  CC.J = K;

  AA.vals = (val_t*) splatt_malloc(I * K * sizeof(val_t));
  BB.vals = (val_t*) splatt_malloc(K * K * sizeof(val_t));
  CC.vals = (val_t*) splatt_malloc(I * K * sizeof(val_t));

  for (int i = 0; i < I * K; i++) {
    AA.vals[i] = 1.0;
    // CC.vals[i] = 0.0;
  }

  for (int i = 0; i < K * K; i++) {
    BB.vals[i] = 1.0;
  }

  my_matmul(
      &AA, false,
      &BB, false,
      &CC
  );

  timer_stop(&m_timer);

	// print_matrix_(&C);
  printf("Time elapsed: %f\n", m_timer.seconds);
  is_matrix_equal(&C, &CC);


  // Print
  /* Testing GELSS
	matrix_t A;
	matrix_t RHS;
	A.I = 10;
	A.J = 10;
	RHS.I = 10;
	RHS.J = 10;
	
	A.vals = (val_t*) splatt_malloc(A.I * A.J * sizeof(val_t));
	RHS.vals = (val_t*) splatt_malloc(RHS.I * RHS.J * sizeof(val_t));

	srand(234);
	
	for (idx_t i = 0; i < A.I * A.J; i++) {
		int r = rand();
		
		A.vals[i] = (r % 255) / 255.0;
	}

	for (idx_t i = 0; i < RHS.I * RHS.J; i++) {
		int r = rand();
		RHS.vals[i] = (r % 255) / 255.0;
	}

	print_matrix_(&A);
	printf("\n");

	bool is_spd = mat_cholesky_(&A);


	mat_solve_cholesky_with_fallback(&A, &RHS, is_spd);
  */


	/*
	printf("Before solve\n");
	print_matrix_(&A);
	printf("\n");
	print_matrix_(&RHS);
	printf("\n");
	

	char tri = 'L';
	splatt_blas_int N = 10;
	splatt_blas_int lda = N;
	splatt_blas_int ldb = N;
	splatt_blas_int info;
	splatt_blas_int nhrs = RHS.I;
	splatt_blas_int effective_rank;

	splatt_blas_int lwork = -1;
	val_t rcond = -1.0f;
	val_t work_query;
	val_t * conditions = (val_t *)splatt_malloc(N * sizeof(*conditions));
	
	// SPLATT_BLAS(gelss)();
	
	LAPACK_DGELSS(
		&N, &N, &nhrs,
		A.vals, &lda, RHS.vals, &ldb,
		conditions, &rcond, &effective_rank,
		&work_query, &lwork, &info);
	printf("Work query complete\n");
	lwork = (splatt_blas_int) work_query;

	printf("%d\n", lwork);

	val_t * work = (val_t*)splatt_malloc(lwork * sizeof(*work));

	LAPACK_DGELSS(
		&N, &N, &nhrs,
		A.vals, &lda, RHS.vals, &ldb,
		conditions, &rcond, &effective_rank,
		work, &lwork, &info);
    
	printf("Solve complete complete\n");
	printf("Effective rank: %d\n", effective_rank);

	print_matrix_(&A);
	printf("\n");
	print_matrix_(&RHS);
	printf("\n");
*/

    exit(1);
    /* End Testing GELSS  */

    if (argc < 2) {
        printf("Usage: ./%s <tensor>\n", argv[0]);
    }

    int nthreads = omp_get_max_threads();
    std::cout << "Number OMP Threads: " << nthreads << std::endl;

    // TBB threads
#if __INTEL_COMPILER
    int pstl_num_threads;
    if (auto pstl_num_threads_ca = std::getenv("PSTL_NUM_THREADS")) {
      std::istringstream iss(pstl_num_threads_ca);
      iss >> pstl_num_threads;
    }
    else {
      pstl_num_threads = tbb::task_scheduler_init::default_num_threads();
    }
    std::cout << "PSTL_NUM_THREADS: " << pstl_num_threads << '\n';
    tbb::task_scheduler_init init(pstl_num_threads);
#endif

    benchmark_timer b("timing.csv");
    b.set_tensor(argv[1]);

    timed_section t("read input");

    sptensor_t * tensor = tt_read_file(argv[1]);
    print_tensor_info(tensor);
    print_nzs_short(tensor);


    t.new_section("get_slice");
    printf("Getting slice (3,700)\n");

    sptensor_t * slice = tt_get_slice(tensor, 3, 700);
    print_tensor_info(slice);
    print_nzs_short(slice);

    t.new_section("permute_slice");
    // create random permutation
    std::vector<idx_t> perm(slice->nnz);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::default_random_engine());
    tt_permute(slice, perm.data());


    t.new_section("mtt");
    mtt(slice, b);

    test_mttkrp(slice);


    return 0;
}

