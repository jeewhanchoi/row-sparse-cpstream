
#include "StreamCPD.hxx"
#include "StreamMatrix.hxx"

// #define XING_DEBUG
#define ADMM_FUSION

extern "C" {
#include "../admm.h"
#include "../../mttkrp.h"
#include "../../timer.h"
#include "../../util.h"
#include "../../stats.h"
#include "../../splatt_debug.h"
#ifdef XING_DEBUG
#include "../../io.h"
#endif
}

#include <math.h>
#include <unistd.h>
#include <vector>

#ifdef SPLATT_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif


/* How often to check global factorization error. This is expensive! */
#ifndef CHECK_ERR_INTERVAL
#define CHECK_ERR_INTERVAL 1000
#endif

// #define DEBUG 1
#define USE_RSP_MTTKRP

#define SKIP_TEST 1
#define FIXED_NUM_IT 100

// Row-sparse mttkrp implementation
#include "../../stmttkrp.hpp"

#define DOTIME 0

static void p_copy_upper_tri(
    matrix_t * const M)
{
  timer_start(&timers[TIMER_ATA]);
  idx_t const I = M->I;
  idx_t const J = M->J;
  val_t * const restrict vals = M->vals;

  #pragma omp parallel for schedule(static, 1) if(I > 50)
  for(idx_t i=1; i < I; ++i) {
    for(idx_t j=0; j < i; ++j) {
      vals[j + (i*J)] = vals[i + (j*J)];
    }
  }
  timer_stop(&timers[TIMER_ATA]);
}

std::vector<size_t> zero_slices(const idx_t I, std::vector<size_t> &nz_rows)
{
  std::vector<size_t> zero_slices;

  idx_t cnt = 0;
  for (idx_t i = 0; i < I; i++) {
    if (cnt >= nz_rows.size()) {
      zero_slices.push_back(i);
    } else  {
      if (i == nz_rows.at(cnt)) {
        // If i is a index for a non-zero row, skip
        cnt += 1;
      } else {
        zero_slices.push_back(i);
      }
    }
  }
  return zero_slices;
}

/**
 * @brief computes aTa given a full matrix and row indices
 *
 * Is equivalent to a sequence of `convert_to_rspmat` -> `rsp_mataTb`
 */
static void mataTa_idx_based(
    matrix_t* A,
    std::vector<size_t>& idx,
    matrix_t* dest)
{
  assert(dest->I == dest->J);
  size_t _size = dest->I;
  #pragma omp parallel for schedule(static) 
  for (idx_t i = 0; i < _size; i ++) {
    for (idx_t j = 0; j < _size; j ++) {
      val_t tmp = 0.0;
      for (idx_t k = 0; k < idx.size(); k++) {
        idx_t row_idx = idx.at(k);
        tmp += A->vals[row_idx * _size + j] * A->vals[row_idx * _size + i];
      }
      dest->vals[i * _size + j] = tmp;
    }
  }
}

void StreamCPD::track_row(
    idx_t mode,
    idx_t orig_index,
    char * name)
{
  idx_t const num_rows = _stream_mats_new[mode]->num_rows();

  /* lookup stocks */
  idx_t const row_id = _source->lookup_ind(mode, orig_index);
  printf("tracking %s: %lu of %lu\n", name, row_id, num_rows);
  if(num_rows >= row_id) {
    val_t const * const track_row = &(_stream_mats_new[mode]->vals()[_rank * row_id]);
    val_t const * const time_row = _mat_ptrs[_stream_mode]->vals;
    val_t track_norm = 0.;
    val_t time_norm  = 0.;
    val_t inner = 0.;
    for(idx_t f=0; f < _rank; ++f) {
      inner += track_row[f] * time_row[f];

      track_norm += track_row[f] * track_row[f];
      time_norm += time_row[f] * time_row[f];
    }
    printf("%s: %e (track: %e time: %e)\n", name, inner, track_norm, time_norm);
  }
}


double StreamCPD::compute_errorsq(
    idx_t num_previous)
{
  splatt_kruskal * cpd = get_prev_kruskal(num_previous);
  sptensor_t * prev_tensor = _source->stream_prev(num_previous);
  double const err = cpd_error(prev_tensor, cpd);
  tt_free(prev_tensor);
  splatt_free_cpd(cpd);
  return err * err;
}


double StreamCPD::compute_cpd_errorsq(
    idx_t num_previous)
{
  sptensor_t * prev_tensor = _source->stream_prev(num_previous);

  double * csf_opts = splatt_default_opts();
  splatt_csf * csf = splatt_csf_alloc(prev_tensor, csf_opts);
  splatt_free(prev_tensor);

  splatt_kruskal * newcpd = splatt_alloc_cpd(csf, _rank);
  splatt_cpd(csf, _rank, NULL, NULL, newcpd);
  splatt_free_csf(csf, csf_opts);

  double const err = 1 - newcpd->fit;

  splatt_free_cpd(newcpd);
  splatt_free_opts(csf_opts);
  return err;
}

splatt_kruskal * StreamCPD::get_kruskal()
{
  /* store output */
  splatt_kruskal * cpd = (splatt_kruskal *) splatt_malloc(sizeof(*cpd));
  cpd->nmodes = _nmodes;
  cpd->lambda = (val_t *) splatt_malloc(_rank * sizeof(*cpd->lambda));
  cpd->rank = _rank;
  for(idx_t r=0; r < _rank; ++r) {
    cpd->lambda[r] = 1.;
  }
  for(idx_t m=0; m < _nmodes; ++m) {
    if(m == _stream_mode) {
      idx_t const nrows = _global_time->num_rows();
      cpd->dims[m] = nrows;
      cpd->factors[m] = (val_t *)
          splatt_malloc(nrows * _rank * sizeof(val_t));
      par_memcpy(cpd->factors[m], _global_time->vals(), nrows * _rank * sizeof(val_t));

    } else {
      idx_t const nrows = _stream_mats_new[m]->num_rows();
      cpd->dims[m] = nrows;

      cpd->factors[m] = (val_t *) splatt_malloc(nrows * _rank * sizeof(val_t));
      /* permute rows */
      #pragma omp parallel for schedule(static)
      for(idx_t i=0; i < nrows; ++i) {
        idx_t const new_id = i;
        memcpy(&(cpd->factors[m][i*_rank]),
               &(_stream_mats_new[m]->vals()[new_id * _rank]),
               _rank * sizeof(val_t));
      }
    }
  }

  return cpd;
}

void StreamCPD::print_kruskal()
{
  for (idx_t m = 0; m < _nmodes; ++m) {
    if(m == _stream_mode) {
      // print_matrix("streaming mode", _global_time->mat());
      print_matrix("streaming mode", _global_time->mat());

    } else {
      printf("%d\n", m);
      print_matrix("factor matrix", _stream_mats_new[m]->mat());
    }    
  }
}


splatt_kruskal * StreamCPD::get_prev_kruskal(idx_t previous)
{
  /* store output */
  splatt_kruskal * cpd = (splatt_kruskal *) splatt_malloc(sizeof(*cpd));
  cpd->nmodes = _nmodes;
  cpd->lambda = (val_t *) splatt_malloc(_rank * sizeof(*cpd->lambda));
  cpd->rank = _rank;
  for(idx_t r=0; r < _rank; ++r) {
    cpd->lambda[r] = 1.;
  }
  for(idx_t m=0; m < _nmodes; ++m) {
    if(m == _stream_mode) {
      idx_t const nrows = SS_MIN(previous, _global_time->num_rows());
      idx_t const startrow = _global_time->num_rows() - nrows;

      cpd->dims[m] = nrows;
      cpd->factors[m] = (val_t *)
          splatt_malloc(nrows * _rank * sizeof(val_t));
      par_memcpy(cpd->factors[m], &(_global_time->vals()[startrow * _rank]),
          nrows * _rank * sizeof(val_t));

    } else {
      idx_t const nrows = _stream_mats_new[m]->num_rows();
      cpd->dims[m] = nrows;

      cpd->factors[m] = (val_t *) splatt_malloc(nrows * _rank * sizeof(val_t));
      /* permute rows */
      #pragma omp parallel for schedule(static)
      for(idx_t i=0; i < nrows; ++i) {
        idx_t const new_id = i;
        memcpy(&(cpd->factors[m][i*_rank]),
               &(_stream_mats_new[m]->vals()[new_id * _rank]),
               _rank * sizeof(val_t));
      }
    }
  }

  return cpd;
}


void StreamCPD::grow_mats(
    idx_t const * const new_dims)
{
  for(idx_t m=0; m < _nmodes; ++m) {
    if(m != _stream_mode) {
      idx_t const new_rows = new_dims[m];

      _stream_mats_new[m]->grow_rand(new_rows);
      _mat_ptrs[m] = _stream_mats_new[m]->mat();
      mat_aTa(_mat_ptrs[m], _cpd_ws->aTa[m]);

      _stream_mats_old[m]->grow_zero(new_rows);
      _stream_duals[m]->grow_zero(new_rows);

      _stream_auxil->grow_zero(new_rows);
      _stream_init->grow_zero(new_rows);
      _mttkrp_buf->grow_zero(new_rows);
    } else {
      _stream_duals[m]->grow_zero(1); /* we only need 1 row for time dual */
      _global_time->grow_zero(_global_time->num_rows() + 1);
    }

    _cpd_ws->duals[m] = _stream_duals[m]->mat();
  }
  // Where we assign mttkrp results ?
  _mat_ptrs[SPLATT_MAX_NMODES] = _mttkrp_buf->mat();
  _cpd_ws->mttkrp_buf = _mttkrp_buf->mat();
  _cpd_ws->auxil = _stream_auxil->mat();
  _cpd_ws->mat_init = _stream_init->mat();
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


// Implementation of add_historical where
// we accept a row_sparse mttkrp result 
// instead of a full sized factor matrix
void StreamCPD::add_historical_rsp(
    idx_t const mode, 
    const rsp_matrix_t * const rsp_mat)
{
  matrix_t * ata_buf = _cpd_ws->aTa_buf;
  timer_start(&timers[TIMER_MATMUL]);

  /*
   * Construct Gram matrix.
   */

  /* Time info -- make sure to copy upper triangular to lower */
  par_memcpy(ata_buf->vals, _old_gram->vals,
      _rank * _rank * sizeof(*ata_buf->vals));
  p_copy_upper_tri(ata_buf);

  matrix_t * _historical = mat_zero(_rank, _rank);

  /* other factors: old^T * new */
  /* TODO: timer */
  for(idx_t m=0; m < _nmodes; ++m) {
    if((m == mode) || (m == _stream_mode)) {
      continue;
    }
    assert(_stream_mats_new[m]->num_rows() == _stream_mats_old[m]->num_rows());

    // H <- A_old^T * A_new
    // TODO: do this only once per mode (ie, update H[m] when A_new[m] is updated)
    // TODO: PAUL - Needed for performance enhancement!!!!
    my_matmul(_stream_mats_old[m]->mat(), true,
              _stream_mats_new[m]->mat(), false,
              _historical);

    /* incorporate into Gram */
    #pragma omp parallel for schedule(static)
    for(idx_t x=0; x < _rank * _rank; ++x) {
      ata_buf->vals[x] *= _historical->vals[x];
    }
  }

  /*
   * mttkrp_tmp = old * aTa_buf
   */
  matrix_t *mttkrp_tmp = (matrix_t *) splatt_malloc(sizeof(matrix_t));
  mttkrp_tmp->I = _stream_mats_old[mode]->mat()->I;
  mttkrp_tmp->J = _stream_mats_old[mode]->mat()->J;
  mttkrp_tmp->vals = (val_t*) splatt_malloc(mttkrp_tmp->I * mttkrp_tmp->J * sizeof(val_t));

  // A_prev * Q
  my_matmul(_stream_mats_old[mode]->mat(), false, ata_buf, false, mttkrp_tmp); 

  // Add updated rows to mttkrp_tmp->vals
  idx_t const num_rows = rsp_mat->nnzr;
  idx_t J = rsp_mat->J;

  #pragma omp parallel for schedule(static)
  for (idx_t i = 0; i < num_rows; i++) {
    // Sum i-th row in the row sparse matrix to the
    // rsp->rowind[i] -th row in the full size matrix
    for (idx_t j = 0; j < J; j++) {
      mttkrp_tmp->vals[rsp_mat->rowind[i] * J + j] += rsp_mat->mat->vals[i * J + j];
    }
  }

  // Final Write to _mttkrp_buf->mat()
  par_memcpy(_mttkrp_buf->mat()->vals, mttkrp_tmp->vals, mttkrp_tmp->I * mttkrp_tmp->J * sizeof(*mttkrp_tmp->vals));

  timer_stop(&timers[TIMER_MATMUL]);
  mat_free(_historical);
  mat_free(mttkrp_tmp);
}


void StreamCPD::add_historical(
    idx_t const mode)
{
  matrix_t * ata_buf = _cpd_ws->aTa_buf;
  timer_start(&timers[TIMER_MATMUL]);

  /*
   * Construct Gram matrix.
   */

  /* Time info -- make sure to copy upper triangular to lower */
  par_memcpy(ata_buf->vals, _old_gram->vals,
      _rank * _rank * sizeof(*ata_buf->vals));
  p_copy_upper_tri(ata_buf);

  matrix_t * _historical = mat_zero(_rank, _rank);

  /* other factors: old^T * new */
  /* TODO: timer */
  for(idx_t m=0; m < _nmodes; ++m) {
    if((m == mode) || (m == _stream_mode)) {
      continue;
    }
    assert(_stream_mats_new[m]->num_rows() == _stream_mats_old[m]->num_rows());

    // H <- A_old^T * A_new
    // TODO: do this only once per mode (ie, update H[m] when A_new[m] is updated)
    // TODO: PAUL - Needed for performance enhancement!!!!
    my_matmul(_stream_mats_old[m]->mat(), true,
              _stream_mats_new[m]->mat(), false,
              _historical);

    /* incorporate into Gram */
    #pragma omp parallel for schedule(static)
    for(idx_t x=0; x < _rank * _rank; ++x) {
      ata_buf->vals[x] *= _historical->vals[x];
    }
  }

  matrix_t * A_nz_prev_Q_ref = mat_alloc(_mttkrp_buf->mat()->I, _mttkrp_buf->mat()->J);
  my_matmul(_stream_mats_old[mode]->mat(), false, ata_buf, false, A_nz_prev_Q_ref, 0.0); 

  /*
  print_matrix("Q ref", ata_buf);
  */

  /*
  print_matrix("A_nz_prev_Q ref", A_nz_prev_Q_ref);
  */

  /*
   * mttkrp += old * aTa_buf
   */
  my_matmul(_stream_mats_old[mode]->mat(), false, ata_buf, false,
            _mttkrp_buf->mat(), 1.0);
  timer_stop(&timers[TIMER_MATMUL]);
  mat_free(_historical);
  mat_free(A_nz_prev_Q_ref);
}


StreamCPD::StreamCPD(
    ParserBase * source
) :
    _source(source)
{
}

StreamCPD::~StreamCPD()
{
}


inline val_t admm(
    idx_t mode,
    matrix_t * * mats,
    val_t * const restrict column_weights,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{ 
#ifdef ADMM_FUSION
  splatt_cpd_constraint *con = cpd_opts->constraints[mode];
  if (strcmp(con->description, "MAX-COL-NORM") == 0) {
    admm_stream(mode, mats, column_weights, ws, cpd_opts, global_opts);
  } else
#endif
  admm_(mode, mats, column_weights, ws, cpd_opts, global_opts);
}

/**
 * StreamCPD::compute_rowsparse
 */
splatt_kruskal *  StreamCPD::compute_rowsparse(
    splatt_idx_t const rank,
    double const forget,
    splatt_cpd_opts * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const stream_mode = _source->stream_mode();
  idx_t const num_modes = _source->num_modes();

  /* TODO fix constructor */
  _stream_mode = stream_mode;
  _rank = rank;
  _nmodes = num_modes;

  // sp_timer_t t_pre;
  // timer_reset(&t_pre);

  // timer_start(&t_pre);
  val_t* colnorms = (val_t*) splatt_malloc(rank*sizeof(val_t));
  _cpd_ws = cpd_alloc_ws_empty(_nmodes, _rank, cpd_opts, global_opts);
  matrix_t** mats_aTa =_cpd_ws->aTa;
  matrix_t* gram = mat_zero(rank, rank); // XXX the naming overlaps with the time gram G
  matrix_t* hgram = mat_zero(rank, rank); // historical gram G_t-1
  matrix_t** mats_haTa = (matrix_t**) splatt_malloc(num_modes * sizeof(matrix_t*));

  /* Hypersparse ALS */
  matrix_t** c_nz_prev = (matrix_t**) splatt_malloc(_nmodes * sizeof(matrix_t));
  matrix_t** c_z_prev = (matrix_t**) splatt_malloc(_nmodes * sizeof(matrix_t)); 

  matrix_t** c_nz = (matrix_t**) splatt_malloc(_nmodes * sizeof(matrix_t));
  matrix_t** c_z = (matrix_t**) splatt_malloc(_nmodes * sizeof(matrix_t));

  matrix_t** h_nz = (matrix_t**) splatt_malloc(_nmodes * sizeof(matrix_t));
  matrix_t** h_z = (matrix_t**) splatt_malloc(_nmodes * sizeof(matrix_t));

  // c = c_z + c_nz, h = h_z + h_nz
  matrix_t** h = (matrix_t**) splatt_malloc(_nmodes * sizeof(matrix_t));
  matrix_t** c = (matrix_t**) splatt_malloc(_nmodes * sizeof(matrix_t));

  // Used for line 29
  matrix_t** c_prev = (matrix_t**) splatt_malloc(_nmodes * sizeof(matrix_t));

  rsp_matrix_t** A_nz_prev = (rsp_matrix_t**) splatt_malloc(_nmodes * sizeof(rsp_matrix_t*));
  rsp_matrix_t** A_nz = (rsp_matrix_t**) splatt_malloc(_nmodes * sizeof(rsp_matrix_t*));

  /* Q * Phi^-1 is needed to update A_z[m] after convergence */
  matrix_t** Q_Phi_inv = (matrix_t**) splatt_malloc(_nmodes * sizeof(matrix_t));

  for (idx_t m = 0; m < num_modes; ++m) {
    mats_aTa[m] = mat_zero(rank, rank);
    mats_haTa[m] = mat_zero(rank, rank);

    /* Hypersparse ALS */
    c_nz_prev[m] = mat_zero(rank, rank);
    c_z_prev[m] = mat_zero(rank, rank);

    c_nz[m] = mat_zero(rank, rank);
    c_z[m] = mat_zero(rank, rank);

    h_nz[m] = mat_zero(rank, rank);
    h_z[m] = mat_zero(rank, rank);

    c[m] = mat_zero(rank, rank);
    h[m] = mat_zero(rank, rank);

    c_prev[m] = mat_zero(rank, rank);
    Q_Phi_inv[m] = mat_zero(rank, rank);
  }

  // Size increases as stream progresses
  _global_time = new StreamMatrix(rank);
  _mttkrp_buf = new StreamMatrix(rank);
  _stream_auxil = new StreamMatrix(rank);
  _stream_init = new StreamMatrix(rank);
  for(idx_t m=0; m < num_modes; ++m) {
    _stream_mats_new[m] = new StreamMatrix(rank);
    _stream_mats_old[m] = new StreamMatrix(rank);
    _stream_duals[m] = new StreamMatrix(rank);
  }

  // s_t
  _mat_ptrs[stream_mode] = mat_zero(1, rank);

  /* Only previous info -- just used for add_historical() */
  _old_gram = mat_zero(rank, rank);

  cpd_stats2(_rank, _source->num_modes(), cpd_opts, global_opts);
  // timer_stop(&t_pre);

  /*
   * Stream
   */
  sp_timer_t stream_time;

  sp_timer_t t_set;
  sp_timer_t t_Q;
  sp_timer_t t_vec;
  sp_timer_t t_first;
  sp_timer_t t_copy;
  sp_timer_t t_gram;
  sp_timer_t t_chol;
  sp_timer_t t_update;
  sp_timer_t t_q_inv;
  sp_timer_t t_ch;
  sp_timer_t t_full;
  sp_timer_t t_final_copy;
  sp_timer_t t_forget;
  sp_timer_t t_clean;
  sp_timer_t t_unknown;

  sp_timer_t t_pre;
  sp_timer_t t_inner;
  sp_timer_t t_post;
  
  sp_timer_t t_admm;
  sp_timer_t t_mttkrp;
  sp_timer_t t_hist;
  sp_timer_t t_cpderr;
  sp_timer_t t_mataTa;
  // Added to measure difference between sp and rsp mttkrp
  sp_timer_t rsp_mttkrp;
  sp_timer_t sp_mttkrp;
  timer_reset(&rsp_mttkrp);
  timer_reset(&sp_mttkrp);

  timer_reset(&t_set);
  timer_reset(&t_Q);
  timer_reset(&t_vec);
  timer_reset(&t_first);
  timer_reset(&t_copy);
  timer_reset(&t_gram);
  timer_reset(&t_chol);
  timer_reset(&t_update);
  timer_reset(&t_q_inv);
  timer_reset(&t_ch);
  timer_reset(&t_full);
  timer_reset(&t_final_copy);
  timer_reset(&t_forget);
  timer_reset(&t_clean);
  timer_reset(&t_unknown);
  timer_reset(&t_pre);
  timer_reset(&t_inner);
  timer_reset(&t_post);

  timer_reset(&stream_time);
  timer_reset(&t_admm);
  timer_reset(&t_mttkrp);
  timer_reset(&t_hist);
  timer_reset(&t_cpderr);
  timer_reset(&t_mataTa);

  idx_t it = 0;
  sptensor_t * batch = _source->next_batch();

  idx_t niter = 0;
  idx_t _niter;

  sp_timer_t t_one;
  timer_reset(&t_one);
  sp_timer_t t_two;
  timer_reset(&t_two);
  sp_timer_t t_three;
  timer_reset(&t_three);
  sp_timer_t t_four;
  timer_reset(&t_four);

  timer_start(&stream_time);
  /* batch start */
  while(batch != NULL) {
    // timer_start(&t_one);
    // if (it >= 500) break;
    // sp_timer_t batch_time;
    // timer_fstart(&batch_time);

    timer_start(&t_pre);
    #if DOTIME
    timer_start(&t_vec);
    #endif
    // Grow matrices that are used for computation for each batch
    grow_mats(batch->dims);

    /**
     * For each time slice tensor batch,
     * we cache the intermediate results used for hypersparse ALS
     * 
     * idx, buckets: used for rsp_mttkrp
     * nz_rows: corresponding non-zeros rows in factor matrix
     */
    std::vector<std::vector<size_t>> nz_rows((size_t)num_modes, 
                                             std::vector<size_t> (0, 0));
    std::vector<std::vector<size_t>> buckets((size_t)num_modes, 
                                             std::vector<size_t> (0, 0));
    std::vector<std::vector<size_t>> idx((size_t)num_modes, 
                                         std::vector<size_t> (0, 0));
    // For storing mappings of indices in I to indices in rowind
    // Without this we had to traverse through all rowind to find the 
    // matching index
    std::vector<std::vector<int>> ridx((size_t)num_modes, 
                                       std::vector<int> (0, 0));
    #if DOTIME
    timer_stop(&t_vec);
    #endif

    // Normalize factor matrices
    for (int m = 0; m < num_modes; ++m) {
        if (m == stream_mode) continue;
        mat_normalize(_stream_mats_new[m]->mat(), colnorms);
    }
    if (it == 0) {
        #pragma omp simd
        for (int r = 0; r < rank; ++r) {
            // Just normalize the columns and reset the lambda
            colnorms[r] = 1.0;
        }
    }

    // TODO: For every batch, compute the non zeros rows once and use it 
    // throughout 
    /* Compute non zero rows for each (batch, m) */

    #if DOTIME
    timer_start(&t_set);
    #endif
    for (idx_t m=0; m < num_modes; m++) {
      nonzero_slices(batch, m, nz_rows[m], idx[m], ridx[m], buckets[m]);

      /* Commenting this out breaks the algo */
      if (m == stream_mode) continue;

      /* Reuse throughout current time slice */
      size_t nnzr = nz_rows[m].size();
      idx_t * rowind = &nz_rows[m][0];
      A_nz_prev[m] = convert_to_rspmat(_stream_mats_old[m]->mat(), nnzr, 
                                       rowind);
      /* Before each inner iteration, c and h are updated accordingly using 
         full factor matrix */
      my_matmul(_stream_mats_old[m]->mat(), true, _stream_mats_new[m]->mat(), 
                false, h[m]);

      mat_aTa(_stream_mats_new[m]->mat(), c[m]);
    }
    #if DOTIME
    timer_stop(&t_set);
    #endif

    #if DOTIME
    timer_start(&t_set);
    #endif
    /* Operations required for t > 1 */
    if (it > 0) {
      for (idx_t m = 0; m < _nmodes; m++) {
        if (m == _stream_mode) continue;
        /* compute C_t-1 based on non zeros slices for current batch */
        mataTa_idx_based(_stream_mats_old[m]->mat(), nz_rows[m],
            c_nz_prev[m]);

        for (idx_t i = 0; i < rank * rank; i++) {
          c_z_prev[m]->vals[i] = c_prev[m]->vals[i] - c_nz_prev[m]->vals[i];
        }
      }
    }

    /* Before inner convergence loop, populate A_nz based on _stream_mats_new */
    for (idx_t m = 0; m < num_modes; m++) {
      if (m == stream_mode) {
        A_nz[m] = rspmat_alloc(1, rank, 1);
        memcpy(A_nz[m]->mat->vals, _mat_ptrs[m]->vals, sizeof(val_t) * rank * 1);
        A_nz[m]->rowind[0] = 0;
      }
      else {
        A_nz[m] = convert_to_rspmat(
            _mat_ptrs[m], 
            nz_rows[m].size(), 
            &nz_rows[m][0]);
      }
    }
    #if DOTIME
    timer_stop(&t_set);
    #endif

    // #if DOTIME
    // timer_start(&t_Q);
    // #endif
    val_t prev_delta = 0.;
    val_t _prev_delta = 0.;


    // #if DOTIME
    // timer_stop(&t_copy);
    // #endif
    /* Complete - Compute new time slice */

    // timer_stop(&t_one);
    timer_stop(&t_pre);

    /* Compute new time slice - TODO: Need to move outside of inner 
        iteration loop */
    _mat_ptrs[SPLATT_MAX_NMODES]->I = 1;

    matrix_t * Phi = mat_zero(rank, rank);
    /* Compute element-wise product for Q and Phi - Q is not symmetric */
    mat_form_gram(c, Phi, _nmodes, _stream_mode);
    mat_add_diag(Phi, 1e-12); // Do this manually

    /* Scratchpad method */
    #if 1
    timer_start(&t_mttkrp);
    #endif

    // mttkrp_stream_wo_lock(batch, _mat_ptrs, stream_mode);
    // mttkrp_stream(batch, _mat_ptrs, stream_mode);
    // A_nz[_stream_mode] = rsp_mttkrp_stream_rsp(batch, A_nz)
    rsp_mttkrp_stream_rsp_streaming_mode(batch, A_nz, stream_mode, 
                          stream_mode, idx[stream_mode], 
                          ridx, buckets[stream_mode]);

    #if 1
    timer_stop(&t_mttkrp);
    #endif

    #if DOTIME
    timer_start(&t_admm);
    #endif
    memcpy(_mat_ptrs[_stream_mode]->vals, A_nz[_stream_mode]->mat->vals, 
            sizeof(val_t) * rank);
    
    // closedform_solve(_mat_ptrs[_stream_mode], Phi, _cpd_ws);
    closedform_solve(A_nz[_stream_mode]->mat, Phi, _cpd_ws);
    // print_matrix("streaming mode", _mat_ptrs[_stream_mode]);
    // print_kruskal();
    // exit(1);

    #if DOTIME
    timer_stop(&t_admm);
    #endif

    // #if DOTIME
    // timer_start(&t_copy);
    // #endif
    /* Accumulate new time slice into temporal Gram matrix */
    val_t       * const restrict ata_vals = _cpd_ws->aTa[stream_mode]->vals;
    // val_t const * const restrict new_slice = _mat_ptrs[stream_mode]->vals;
    val_t const * const restrict new_slice = A_nz[stream_mode]->mat->vals;    
    p_copy_upper_tri(_cpd_ws->aTa[stream_mode]);
    /* save old Gram matrix and update h */
    par_memcpy(_old_gram->vals, ata_vals, rank * rank * sizeof(*ata_vals));
    par_memcpy(h[stream_mode]->vals, ata_vals, rank * rank * sizeof(val_t));
    // #if DOTIME
    // timer_stop(&t_copy);
    // #endif

    timer_start(&timers[TIMER_ATA]);
    #if DOTIME
    timer_start(&t_gram);
    #endif
    #pragma omp parallel for schedule(static) if(rank > 50)
    for(idx_t i=0; i < rank; ++i) {
      for(idx_t j=0; j < rank; ++j) {
        ata_vals[j + (i*rank)] += new_slice[i] * new_slice[j];
        c[stream_mode]->vals[j + (i*rank)] += new_slice[i] * new_slice[j];
      }
    }
    #if DOTIME
    timer_stop(&t_gram);
    #endif
    timer_stop(&timers[TIMER_ATA]);

    /* Update A_nz version of stream mode as well 
      * This is a discrepency between the actual implementation and the pseudo code
      * In pseudo code, this is done out size of the convergence loop
      * */
    // #if DOTIME
    // timer_start(&t_copy);
    // #endif

    memcpy(
      _mat_ptrs[stream_mode]->vals, 
      A_nz[stream_mode]->mat->vals,
      sizeof(val_t) * rank);

    // timer_start(&t_two);
    /* Inner iteration - until convergence  */
    timer_start(&t_inner);

#if SKIP_TEST == 1
    for(idx_t outer=0; outer < FIXED_NUM_IT; ++outer) {
#else
    for(idx_t outer=0; outer < cpd_opts->max_iterations; ++outer) {
#endif
      val_t delta = 0.;
      val_t _delta = 0.; // Used for cross examination


      // timer_start(&t_four);
      /* Update remaining modes */
      for(idx_t m=0; m < num_modes; ++m) {
        if(m == stream_mode) {
          continue;
        }

        // ------------------------------------------------------------
        #if DOTIME
        timer_start(&t_gram);
        #endif
        matrix_t * Q = mat_zero(rank, rank);
        matrix_t * Phi = mat_zero(rank, rank);

        /* Compute element-wise product for Q and Phi - Q is not symmetric, therefore the full version */
        mat_form_gram(c, Phi, _nmodes, m);
        // Add frob reg so that it can be used throughout 
        mat_add_diag(Phi, 1e-12);

        mat_form_gram_full(h, Q, _nmodes, m);
        #if DOTIME
        timer_stop(&t_gram);
        #endif
        // ------------------------------------------------------------


        // ------------------------------------------------------------
        /* SpMTTKRP */
        #if 1
        timer_start(&t_mttkrp);
        #endif
        // Provide rsp_mttkrp with additional parameters: ridx, stream_mode
        rsp_matrix_t * rsp_mat = rsp_mttkrp_stream_rsp(batch, A_nz, m, 
                                                       stream_mode, idx[m], 
                                                       ridx, buckets[m]);
        #if 1
        timer_stop(&t_mttkrp);
        #endif
        // ------------------------------------------------------------


        // ------------------------------------------------------------
        #if 1
        timer_start(&t_hist);
        #endif
        idx_t I = rsp_mat->I;
        idx_t J = rsp_mat->J;
        idx_t nnzr = rsp_mat->nnzr;
        idx_t * rowind = rsp_mat->rowind;

        /* Add historical */
        rsp_matrix_t * A_nz_prev_Q = rsp_mat_mul(A_nz_prev[m], Q);

        rsp_mat_add(rsp_mat, A_nz_prev_Q);
        rspmat_free(A_nz_prev_Q);

        #if 1
        timer_stop(&t_hist);
        #endif
        // ------------------------------------------------------------


        // ------------------------------------------------------------
        #if DOTIME
        timer_start(&t_chol);
        #endif
        // A_nz = temp * Phi^-1
        matrix_t * Phi_cpy = mat_alloc(rank, rank);
        par_memcpy(Phi_cpy->vals, Phi->vals, sizeof(val_t) * rank * rank);

        mat_cholesky(Phi_cpy);
        mat_solve_cholesky(Phi_cpy, rsp_mat->mat);
        #if DOTIME
        timer_stop(&t_chol);
        #endif

        #if 1
        timer_start(&t_update);
        #endif
        rsp_mataTb(rsp_mat, rsp_mat, c_nz[m]);
        rsp_mataTb(A_nz_prev[m], rsp_mat, h_nz[m]);
        #if 1
        timer_stop(&t_update);
        #endif
        // ------------------------------------------------------------


        // ------------------------------------------------------------
        /* Compute Q_Phi_inv */
        #if DOTIME
        timer_start(&t_q_inv);
        #endif
        matrix_t * temp_Phi = mat_alloc(rank, rank);
        matrix_t * temp_Q = mat_alloc(rank, rank);

        par_memcpy(temp_Phi->vals, Phi->vals, sizeof(val_t) * rank * rank);
        par_memcpy(temp_Q->vals, Q->vals, sizeof(val_t) * rank * rank);

        mat_cholesky(temp_Phi);
        mat_solve_cholesky(temp_Phi, temp_Q);

        par_memcpy(Q_Phi_inv[m]->vals, temp_Q->vals, sizeof(val_t) * rank * rank);
        #if DOTIME
        timer_stop(&t_q_inv);
        #endif
        // ------------------------------------------------------------


        // ------------------------------------------------------------
        #if DOTIME
        timer_start(&t_ch);
        #endif
        /* Compute h_z[m] */
        mat_matmul(c_z_prev[m], Q_Phi_inv[m], h_z[m]);
        /* Compute c_z[m] */
        my_matmul(Q_Phi_inv[m], true, h_z[m], false, c_z[m]);

        par_memcpy(A_nz[m]->mat->vals, rsp_mat->mat->vals, sizeof(val_t) * nnzr * J);
        rspmat_free(rsp_mat);

        /* Update c, h */
        c[m] = mat_add(c_nz[m], c_z[m]);
        h[m] = mat_add(h_nz[m], h_z[m]);

        timer_start(&t_cpderr);
        val_t tr_c = mat_trace(c[m]);
        val_t tr_h = mat_trace(h[m]);
        val_t tr_c_prev = mat_trace(c_prev[m]);
        timer_stop(&t_cpderr);
        #if DOTIME
        timer_stop(&t_ch);
        #endif
        // ------------------------------------------------------------


        // ------------------------------------------------------------
        /* Computed by new formulation */
        // #if DOTIME
        // timer_start(&t_unknown);
        // #endif
        _delta += sqrt(fabs(((tr_c + tr_c_prev - 2.0 * tr_h) / (tr_c + 1e-12))));
        // #if DOTIME
        // timer_stop(&t_unknown);
        // #endif
        // ------------------------------------------------------------


      } /* foreach mode */
      
      printf("it: %d delta: %e prev_delta: %e (%e diff)\n", it, _delta, _prev_delta, fabs(_delta - _prev_delta));
      
      _niter = outer + 1;
      // timer_stop(&t_four);

      cpd_opts->tolerance = 1e-3;

#if SKIP_TEST == 1
#else
      if(outer > 0 && fabs(_delta - _prev_delta) < cpd_opts->tolerance) {
        printf("it: %d: converged in: %lu\n", it, outer+1);
        prev_delta = 0.;
        _prev_delta = 0.;
        break;
      }
      prev_delta = delta;
      _prev_delta = _delta;
#endif

    } /* foreach outer max iterations */
    timer_stop(&t_inner);
    niter += _niter;
    // timer_stop(&t_two);

    timer_start(&t_post);
    // timer_start(&t_three);
    timer_start(&t_full);
    /* Update full factor matrices */
    for (idx_t m = 0; m < num_modes; m++) {
      if (m == stream_mode) continue;

      idx_t I = A_nz[m]->I;
      idx_t J = A_nz[m]->J;
      idx_t * rowind = A_nz[m]->rowind;
      size_t nnzr = A_nz[m]->nnzr;

      /* Update based on A_nz[m] */
      #pragma omp parallel for schedule(static)
      for (idx_t i = 0; i < nnzr; i++) {
        idx_t ridx = rowind[i];
        /*
        par_memcpy(
            &_stream_mats_new[m]->mat()->vals[ridx * rank], 
            &A_nz[m]->mat->vals[i * rank], 
            sizeof(val_t) * rank);
        */
        for (idx_t j = 0; j < J; j++) {
          _stream_mats_new[m]->mat()->vals[ridx * rank + j] = A_nz[m]->mat->vals[i * rank + j];
        }
      }

      /* Update based on A_z */
      std::vector<size_t> z_rows = zero_slices(batch->dims[m], nz_rows[m]);
      idx_t nzr = z_rows.size();

      rsp_matrix_t * A_z = convert_to_rspmat(_stream_mats_old[m]->mat(), nzr, &z_rows[0]); 
      rsp_matrix_t * prev_A_z_Q_Phi_inv = rspmat_alloc(A_z->I, A_z->J, A_z->nnzr);
      // We need Q_Phi_inv
      prev_A_z_Q_Phi_inv = rsp_mat_mul(A_z, Q_Phi_inv[m]);

      #pragma omp parallel for schedule(static)
      for (idx_t i=0; i < nzr; i++) {
        idx_t ridx = z_rows.at(i);
        /*
        par_memcpy(
            &_stream_mats_new[m]->mat()->vals[ridx * rank], 
            &prev_A_z_Q_Phi_inv->mat->vals[i * rank], 
            sizeof(val_t) * rank);
        */
        for (idx_t j=0; j < rank; j++) {
          _stream_mats_new[m]->mat()->vals[ridx * rank + j] = prev_A_z_Q_Phi_inv->mat->vals[i * rank + j];
        }
      }
      rspmat_free(A_z);
      rspmat_free(prev_A_z_Q_Phi_inv);
    }
    timer_stop(&t_full);

    // timer_start(&t_clean);
    /* Clean up */
    for (idx_t m = 0; m < num_modes; m++) {
      rspmat_free(A_nz[m]);
    }
    // timer_stop(&t_clean);

    // timer_start(&t_forget);
    /* Incorporate forgetting factor */
    // #pragma omp parallel for schedule(static)
    for(idx_t x=0; x < _rank * _rank; ++x) {
      _cpd_ws->aTa[_stream_mode]->vals[x] *= forget;
      c[_stream_mode]->vals[x] *= forget;
      h[_stream_mode]->vals[x] *= forget;
    }
    // timer_stop(&t_forget);

    // timer_start(&t_clean);
    /* Clean up */
    for(idx_t m=0; m < num_modes; m++) {
      if(m == stream_mode) {
        continue;
      }
      rspmat_free(A_nz_prev[m]);
    }
    // timer_stop(&t_clean);

    // timer_start(&t_final_copy);
    /*
     * Copy new factors into old
     */
    for(idx_t m=0; m < num_modes; ++m) {
      if(m == stream_mode) {
        continue;
      }
      par_memcpy(_stream_mats_old[m]->vals(), _stream_mats_new[m]->vals(),
          _stream_mats_new[m]->num_rows() * rank * sizeof(val_t));

      mat_aTa(_stream_mats_new[m]->mat(), c[m]);
      p_copy_upper_tri(c[m]);

      my_matmul(_stream_mats_old[m]->mat(), true,
          _stream_mats_new[m]->mat(), false,
          h[m]);

      par_memcpy(c_prev[m]->vals, c[m]->vals, sizeof(val_t) * rank * rank);
    }

    /* save time vector */
    par_memcpy(&(_global_time->vals()[it*rank]),
      _mat_ptrs[stream_mode]->vals, rank * sizeof(val_t));
    // timer_stop(&t_final_copy);

    /*
     * Batch stats
     */
    // timer_stop(&batch_time);
    ++it;

    /*
    double local_err   = compute_errorsq(1);
    double global_err  = -1.;
    double local10_err = -1.;
    double cpd_err     = -1.;
    if((it > 0) && ((it % CHECK_ERR_INTERVAL == 0) || _source->last_batch())) {
      global_err  = compute_errorsq(it);
      local10_err = compute_errorsq(10);
      cpd_err     = compute_cpd_errorsq(it);
      if(isnan(cpd_err)) {
        cpd_err = -1.;
      }
    }

    printf("batch %5lu: %7lu nnz (%0.5fs) (%0.3e NNZ/s) "
           "cpd: %+0.5f global: %+0.5f local-1: %+0.5f local-10: %+0.5f\n",
        it, batch->nnz, batch_time.seconds,
        (double) batch->nnz / batch_time.seconds,
        cpd_err, global_err, local_err, local10_err);
     */

    /* prepare for next batch */
    // timer_start(&t_clean);
    tt_free(batch);
    batch = _source->next_batch();
    // timer_stop(&t_clean);
    /* XXX */
    // timer_stop(&t_three);
    timer_stop(&t_post);
  } /* while batch != NULL */
  timer_stop(&stream_time);


  // Report time for all measurements
  /*
  printf("stream-time: %0.3fs\n", stream_time.seconds);
  printf("admm-time: %0.3fs\n", t_admm.seconds);
  printf("mttkrp-time: %0.3fs\n", t_mttkrp.seconds);
  printf("mataTa-time: %0.3fs\n", t_mataTa.seconds);
  printf("historical-time: %0.3fs\n", t_hist.seconds);
  printf("cpderr-time: %0.3fs\n", t_cpderr.seconds);

  printf("stream-time: %0.3fs\n", stream_time.seconds);
   */

  /*
  printf("--tone--time: %0.3f s\n", t_one.seconds);
  printf("--ttwo--time: %0.3f s\n", t_two.seconds);
  printf("--tthree--time: %0.3f s\n", t_three.seconds);
  printf("--tfour--time: %0.3f s\n", t_four.seconds);
   */

  // printf("--unknown--time: %0.3f s\n", t_unknown.seconds);
  // printf("--clean-time: %0.3f s\n", t_clean.seconds);
  // printf("--forget-time: %0.3f s\n", t_forget.seconds);
  // printf("--fnlcpy-time: %0.3f s\n", t_final_copy.seconds);
  printf("--pre-time: %0.3f s\n", t_pre.seconds);
  printf("--inner-time: %0.3f s\n", t_inner.seconds);
  printf("--post--time: %0.3f s\n", t_post.seconds);
  // printf("--full-time: %0.3f s\n", t_full.seconds);
  // printf("--ch-time: %0.3f s\n", t_ch.seconds);
  // printf("--qinv-time: %0.3f s\n", t_q_inv.seconds);
  // printf("--update-time: %0.3f s\n", t_chol.seconds);
  printf("--update-time: %0.3f s\n", t_update.seconds);
  // printf("--gram-time: %0.3f s\n", t_gram.seconds);
  // printf("--copy-time: %0.3f s\n", t_copy.seconds);
  // printf("--first-time: %0.3f s\n", t_first.seconds);
  // printf("--vec-time: %0.3f s\n", t_vec.seconds);
  // printf("--pre-time: %0.3f s\n", t_pre.seconds);
  // printf("--set-time: %0.3f s\n", t_set.seconds);
  // printf("--Q-time: %0.3f s\n", t_Q.seconds);
  // printf("--admm-time: %0.3f s\n", t_admm.seconds);
  printf("--mttkrp-time: %0.3f s\n", t_mttkrp.seconds);
  // printf("--mataTa-time: %0.3f s\n", t_mataTa.seconds);
  printf("--historical-time: %0.3f s\n", t_hist.seconds);
  printf("--cpderr-time: %0.3f s\n", t_cpderr.seconds);
  printf("--stream-time: %0.3f s\n", stream_time.seconds);
  printf("--final-err: %0.5f\n",  1.0);
  //double accum = t_admm.seconds + t_mttkrp.seconds + t_mataTa.seconds + t_hist.seconds + t_cpderr.seconds + t_pre.seconds + t_set.seconds + t_Q.seconds + t_vec.seconds + t_first.seconds + t_copy.seconds + t_gram.seconds + t_update.seconds + t_q_inv.seconds + t_ch.seconds + t_full.seconds + t_final_copy.seconds + t_forget.seconds + t_clean.seconds + t_unknown.seconds + t_chol.seconds;
  double accum = t_pre.seconds + t_inner.seconds + t_post.seconds;
  printf("--Misc: %0.3f s\n", stream_time.seconds - accum);
  printf("--Total: %0.3f s\n", accum);
  printf("--outer: %lu\n", niter);
  printf("--Batches: %lu\n", it);

  /* compute quality assessment */
  splatt_kruskal * cpd = get_kruskal();
  print_kruskal();
  double const final_err = cpd_error(_source->full_stream(), cpd);
  printf("\n");
  printf("final-err: %0.5f\n",  final_err);

  mat_free(_old_gram);

  for(idx_t m=0; m < num_modes; ++m) {
    delete _stream_mats_new[m];
    delete _stream_mats_old[m];
  }
  mat_free(_mat_ptrs[stream_mode]);
  delete _mttkrp_buf;
  delete _stream_init;

  /* XXX */
  //splatt_cpd_free_ws(_cpd_ws);

  return cpd;
}

/**
 * StreamCPD::compute
 */
splatt_kruskal *  StreamCPD::compute(
    splatt_idx_t const rank,
    double const forget,
    splatt_cpd_opts * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const stream_mode = _source->stream_mode();
  idx_t const num_modes = _source->num_modes();

  /* TODO fix constructor */
  _stream_mode = stream_mode;
  _rank = rank;
  _nmodes = num_modes;
#if 1
  /* register constraints */
  for(idx_t m=0; m < num_modes; ++m) {
    if(m != stream_mode) {
      /* convert ntf to norm-constrained ntf */
      if(strcmp(cpd_opts->constraints[m]->description, "non-negative") == 0) {
        splatt_register_maxcolnorm_nonneg(cpd_opts, &m, 1);

        /* just column norm constraints */
      } else if(strcmp(cpd_opts->constraints[m]->description, "unconstrained") == 0) {
        splatt_register_maxcolnorm(cpd_opts, &m, 1);
      }
    }
  }
  val_t *colnorms = NULL;
#else
  val_t* colnorms = (val_t*) splatt_malloc(rank*sizeof(val_t));
#endif

  _cpd_ws = cpd_alloc_ws_empty(_nmodes, _rank, cpd_opts, global_opts);
  matrix_t** mats_aTa =_cpd_ws->aTa;
  matrix_t* gram = mat_zero(rank, rank); // xxx the naming overlaps with the time gram g
  matrix_t* hgram = mat_zero(rank, rank);
  matrix_t** mats_haTa = (matrix_t**) splatt_malloc(num_modes * sizeof(matrix_t*));
  for (idx_t m = 0; m < num_modes; ++m) {
    mats_aTa[m] = mat_zero(rank, rank);
    mats_haTa[m] = mat_zero(rank, rank);
  }

  // size increases as stream progresses
  _global_time = new StreamMatrix(rank);
  _mttkrp_buf = new StreamMatrix(rank);
  _stream_auxil = new StreamMatrix(rank);
  _stream_init = new StreamMatrix(rank);
  for(idx_t m=0; m < num_modes; ++m) {
    _stream_mats_new[m] = new StreamMatrix(rank);
    _stream_mats_old[m] = new StreamMatrix(rank);
    _stream_duals[m] = new StreamMatrix(rank);
  }

  _mat_ptrs[stream_mode] = mat_zero(1, rank);

  /* only previous info -- just used for add_historical() */
  _old_gram = mat_zero(rank, rank);

#if use_csf == 1
  double * csf_opts = splatt_default_opts();
  csf_opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  csf_opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
  csf_opts[SPLATT_OPTION_VERBOSITY] = SPLATT_VERBOSITY_NONE;
#endif

  // cpd_opts->tolerance = 5e-2;
  cpd_opts->max_inner_iterations = 25;
  cpd_opts->inner_tolerance = 7e-2;
  cpd_stats2(_rank, _source->num_modes(), cpd_opts, global_opts);

  /*
   * stream
   */
  sp_timer_t stream_time;
  sp_timer_t t_admm;
  sp_timer_t t_mttkrp;
  sp_timer_t t_hist;
  sp_timer_t t_cpderr;
  sp_timer_t t_matata;
  // added to measure difference between sp and rsp mttkrp
  sp_timer_t rsp_mttkrp;
  sp_timer_t sp_mttkrp;
  timer_reset(&rsp_mttkrp);
  timer_reset(&sp_mttkrp);
  
  timer_reset(&stream_time);
  timer_reset(&t_admm);
  timer_reset(&t_mttkrp);
  timer_reset(&t_hist);
  timer_reset(&t_cpderr);
  timer_reset(&t_matata);

  idx_t it = 0;
  sptensor_t * batch = _source->next_batch();

  /* batch start */
  while(batch != NULL) {
    sp_timer_t batch_time;
    timer_start(&stream_time);
    timer_fstart(&batch_time);

    // grow matrices that are used for computation for each batch
    grow_mats(batch->dims);
    /* normalize factors on the first batch */
    if(it == 0) {
      val_t * tmp = (val_t *) splatt_malloc(_rank * sizeof(*tmp));
      for(idx_t m=0; m < num_modes; ++m) {
        if(m == stream_mode) {
          continue;
        }
        // tmp is for lambda for discarding for it==0?
        mat_normalize(_mat_ptrs[m], tmp);
        // This was added later
        mat_aTa(_mat_ptrs[m], _cpd_ws->aTa[m]);

      }
      splatt_free(tmp);
    }


    /*
      * compute new time slice.
      */
    timer_start(&t_mttkrp);

    _mat_ptrs[SPLATT_MAX_NMODES]->I = 1;

    // print_matrix("before mttkrp", _mttkrp_buf->mat());
    mttkrp_stream_wo_lock(batch, _mat_ptrs, stream_mode);
    // print_matrix("after mttkrp", _mttkrp_buf->mat());

    timer_stop(&t_mttkrp);

    timer_start(&t_admm);
    admm(_stream_mode, _mat_ptrs, colnorms, _cpd_ws, cpd_opts, global_opts);
    timer_stop(&t_admm);
    // print_matrix("after admm", _mat_ptrs[_stream_mode]);

    // exit(1);
    /* accumulate new time slice into temporal gram matrix */
    val_t       * const restrict ata_vals = _cpd_ws->aTa[stream_mode]->vals;
    val_t const * const restrict new_slice = _mat_ptrs[stream_mode]->vals;
    p_copy_upper_tri(_cpd_ws->aTa[stream_mode]);
    /* save old gram matrix */
    par_memcpy(_old_gram->vals, ata_vals, rank * rank * sizeof(*ata_vals));

    timer_start(&timers[TIMER_ATA]);
#pragma omp parallel for schedule(static) if(rank > 50)
    for(idx_t i=0; i < rank; ++i) {
      for(idx_t j=0; j < rank; ++j) {
        ata_vals[j + (i*rank)] += new_slice[i] * new_slice[j];
      }
    }
    timer_stop(&timers[TIMER_ATA]);



    val_t prev_delta = 0.;
#if SKIP_TEST
    for(idx_t outer=0; outer < FIXED_NUM_IT; ++outer) {
#else
    for(idx_t outer=0; outer < cpd_opts->max_iterations; ++outer) {
#endif
      val_t delta = 0.;
      /*
       * update the remaining modes
       */
      for(idx_t m=0; m < num_modes; ++m) {
        if(m == stream_mode) {
          continue;
        }

        /* mttkrp */
        _mat_ptrs[SPLATT_MAX_NMODES]->I = batch->dims[m];

#ifdef non_init_version
        /* start non-init version */
        // matrix_t * _temp_mat = mat_alloc(batch->dims[m], rank);

        timer_start(&t_hist);
        add_historical_non_init(_mttkrp_buf->mat(), m);
        timer_stop(&t_hist);

        timer_start(&t_mttkrp);
        mttkrp_stream_non_init(_mttkrp_buf->mat(), batch, _mat_ptrs, m);
        timer_stop(&t_mttkrp);

        // par_memcpy(_mttkrp_buf->mat()->vals, _temp_mat->vals, sizeof(val_t) * _temp_mat->I * _temp_mat->j);

        // mat_free(_temp_mat);
#else
        timer_start(&t_mttkrp);
        mttkrp_stream(batch, _mat_ptrs, m);
        // mttkrp_stream_wo_lock(batch, _mat_ptrs, m);
        timer_stop(&t_mttkrp);
        /* add historical data to mttkrp */
        timer_start(&t_hist);
        add_historical(m);
        timer_stop(&t_hist);
#endif


        /* reset dual matrix. todo: necessary? */
        memset(_cpd_ws->duals[m]->vals, 0,
            _cpd_ws->duals[m]->I * _rank * sizeof(val_t));
        timer_start(&t_admm);     
        admm(m, _mat_ptrs, colnorms, _cpd_ws, cpd_opts, global_opts);
        timer_stop(&t_admm);

        /* lastly, update ata */
        timer_start(&t_matata);
        mat_aTa(_stream_mats_new[m]->mat(), _cpd_ws->aTa[m]);
        timer_stop(&t_matata);

        delta +=
          mat_norm_diff(_stream_mats_old[m]->mat(), _stream_mats_new[m]->mat())
          / (mat_norm(_stream_mats_new[m]->mat()) + 1e-12);

      } /* foreach mode */

      // printf("it: %d delta: %e prev_delta: %e (%e diff)\n", it, delta, prev_delta, fabs(delta - prev_delta));

      /* check convergence */
      // cpd_opts->tolerance = 1e-3;
#if SKIP_TEST == 1
#else
      if(outer > 0 && fabs(delta - prev_delta) < cpd_opts->tolerance) {
        printf("it: %d: converged in: %lu\n", it, outer+1);
        prev_delta = 0.;
        break;
      }
      prev_delta = delta;
#endif
    } /* foreach outer max iterations */

    /* incorporate forgetting factor */
    for(idx_t x=0; x < _rank * _rank; ++x) {
      _cpd_ws->aTa[_stream_mode]->vals[x] *= forget;
    }

    /*
     * copy new factors into old
     */
    for(idx_t m=0; m < num_modes; ++m) {
      if(m == stream_mode) {
        continue;
      }
      par_memcpy(_stream_mats_old[m]->vals(), _stream_mats_new[m]->vals(),
          _stream_mats_new[m]->num_rows() * rank * sizeof(val_t));
    }

    /* save time vector */
    par_memcpy(&(_global_time->vals()[it*rank]),
        _mat_ptrs[stream_mode]->vals, rank * sizeof(val_t));

    /*
     * batch stats
     */
    timer_stop(&batch_time);
    timer_stop(&stream_time);
    ++it;

    /*
       double local_err   = compute_errorsq(1);
       double global_err  = -1.;
       double local10_err = -1.;
       double cpd_err     = -1.;
       if((it > 0) && ((it % check_err_interval == 0) || _source->last_batch())) {
       global_err  = compute_errorsq(it);
       local10_err = compute_errorsq(10);
       cpd_err     = compute_cpd_errorsq(it);
       if(isnan(cpd_err)) {
       cpd_err = -1.;
       }
       }

       printf("batch %5lu: %7lu nnz (%0.5fs) (%0.3e nnz/s) "
       "cpd: %+0.5f global: %+0.5f local-1: %+0.5f local-10: %+0.5f\n",
       it, batch->nnz, batch_time.seconds,
       (double) batch->nnz / batch_time.seconds,
       cpd_err, global_err, local_err, local10_err);
       */

    /* prepare for next batch */
    tt_free(batch);
#if use_csf == 1
    splatt_free_csf(csf, csf_opts);
    splatt_mttkrp_free_ws(mttkrp_ws);
#endif
    batch = _source->next_batch();
    /* xxx */
  } /* while batch != NULL */

  // report time for all measurements
  printf("stream-time: %0.3fs\n", stream_time.seconds);
  printf("admm-time: %0.3fs\n", t_admm.seconds);
  printf("mttkrp-time: %0.3fs\n", t_mttkrp.seconds);
  printf("matata-time: %0.3fs\n", t_matata.seconds);
  printf("historical-time: %0.3fs\n", t_hist.seconds);
  printf("cpderr-time: %0.3fs\n", t_cpderr.seconds);

  // for comparing rsp and sp mttkrp
#if use_csf == 1
  splatt_free_opts(csf_opts);
#endif

  /* compute quality assessment */
  splatt_kruskal * cpd = get_kruskal();
  double const final_err = cpd_error(_source->full_stream(), cpd);
  printf("\n");
  printf("final-err: %0.5f\n",  final_err * final_err);

  mat_free(_old_gram);
  /*
     mat_free(m2);
     mat_free(m3);
     mat_free(m_ref);
     */

  for(idx_t m=0; m < num_modes; ++m) {
    delete _stream_mats_new[m];
    delete _stream_mats_old[m];
  }
  mat_free(_mat_ptrs[stream_mode]);
  delete _mttkrp_buf;
  delete _stream_init;

  /* xxx */
  //splatt_cpd_free_ws(_cpd_ws);

  return cpd;
}

