



/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "admm.h"
#include "../util.h"
#include "../splatt_debug.h"

#include <omp.h>
#include <math.h>

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Compute the auxiliary matrix before the Cholesky solve. This function
*        computes: mat_mttkrp + (penalty .* (mat_primal - mat_dual)).
*
* @param mat_primal The primal variable.
* @param mat_mttkrp The latest MTTKRP result.
* @param mat_dual The dual variable.
* @param penalty The penalty parameter, 'rho'. This could also be used during
*                l2 (Tikhonov) regularization.
* @param[out] mat_auxil The auxiliary matrix.
* @param should_parallelize Whether we should parallelize.
*/
static void p_setup_auxiliary(
    matrix_t const * const mat_primal,
    matrix_t const * const mat_mttkrp,
    matrix_t const * const mat_dual,
    val_t const penalty,
    matrix_t * const mat_auxil,
    bool const should_parallelize)
{
  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict aux    = mat_auxil->vals;

  val_t const * const restrict mttkrp = mat_mttkrp->vals;
  val_t const * const restrict primal = mat_primal->vals;
  val_t const * const restrict dual   = mat_dual->vals;

  #pragma omp parallel for schedule(static) if(should_parallelize)
  for(idx_t x=0; x < I * J; ++x) {
    aux[x] = mttkrp[x] + penalty * (primal[x] + dual[x]);
  }
}



/**
* @brief Update the dual variable after updating the primal and auxiliary
*        variables. The squared Frobenius norm of the new dual is returned.
*        This function performs: mat_dual += mat_primal - mat_auxil.
*
* @param mat_primal The newest primal variable.
* @param mat_auxil The newest auxiliary variable.
* @param[out] mat_dual The dual variable to update.
* @param should_parallelize Whether we should parallelize.
*
* @return The norm of the new dual; || mat_dual ||_F^2.
*/
static val_t p_update_dual(
    matrix_t const * const mat_primal,
    matrix_t const * const mat_auxil,
    matrix_t * const mat_dual,
    bool const should_parallelize)
{
  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict dual = mat_dual->vals;
  val_t const * const restrict matv = mat_primal->vals;
  val_t const * const restrict auxl = mat_auxil->vals;

  val_t norm = 0.;

  #pragma omp parallel for schedule(static) reduction(+:norm) \
      if(should_parallelize)
  for(idx_t x=0; x < I * J; ++x) {
    dual[x] += matv[x] - auxl[x];
    norm += dual[x] * dual[x];
  }

  return norm;
}


/**
* @brief Initialize the primal matrix with (auxil - dual).
* 
* @param[out] mat_primal The primal matrix to initialize.
* @param mat_auxil The auxiliary matrix.
* @param mat_dual The dual matrix.
* @param should_parallelize Whether we should parallelize.
*/
static void p_setup_proximity(
    matrix_t       * const mat_primal,
    matrix_t const * const mat_auxil,
    matrix_t const * const mat_dual,
    bool const should_parallelize)
{
  val_t       * const restrict primal = mat_primal->vals;
  val_t const * const restrict auxil = mat_auxil->vals;
  val_t const * const restrict dual = mat_dual->vals;

  idx_t const N = mat_primal->I * mat_primal->J;

  #pragma omp parallel for schedule(static) if(should_parallelize)
  for(idx_t x=0; x < N; ++x) {
    primal[x] = auxil[x] - dual[x];
  }
}


/**
* @brief Calculate the primal and dual residuals before the ADMM convergence
*        check.
*
* @param mat_primal The primal variable (the factor we are updating).
* @param mat_auxil The auxiliary matrix; ideally mat_auxil^T = mat_primal.
* @param mat_init The initial matrix factor (at the start of this iteration).
* @param[out] primal_norm The norm of the primal variable; norm(mat_primal)^2.
* @param[out] primal_resid The residual of the primal variable;
*             norm(mat_primal - mat_auxil)^2.
* @param[out] dual_resid The dual residual; norm(mat_primal - mat_init)^2.
* @param should_parallelize Whether we should parallelize.
*/
static void p_calc_residual(
    matrix_t const * const mat_primal,
    matrix_t const * const mat_auxil,
    matrix_t const * const mat_init,
    val_t * primal_norm,
    val_t * primal_resid,
    val_t * dual_resid,
    bool const should_parallelize)
{
  val_t const * const restrict matv = mat_primal->vals;
  val_t const * const restrict auxv = mat_auxil->vals;
  val_t const * const restrict init = mat_init->vals;

  idx_t const nrows = mat_primal->I;
  idx_t const ncols = mat_primal->J;

  val_t p_norm  = 0;
  val_t p_resid = 0;
  val_t d_resid = 0;

  /*
   * Converge based on max row movement.
   */
#if SPLATT_ADMM_ROW_CONVERGE
  #pragma omp parallel for reduction(max:p_norm, p_resid, d_resid) \
      if(should_parallelize)
  for(idx_t i=0; i < nrows; ++i) {
    val_t row_p_norm  = 0;
    val_t row_p_resid = 0;
    val_t row_d_resid = 0;

    for(idx_t j=0; j < ncols; ++j) {
      idx_t const index = j + (i*ncols);
      val_t const pdiff = matv[index] - auxv[index];
      val_t const ddiff = matv[index] - init[index];
      row_p_norm  += matv[index] * matv[index];
      row_p_resid += pdiff * pdiff;
      row_d_resid += ddiff * ddiff;
    }

    /* save the row with the largest primal residual */
    if(row_p_resid > p_resid) {
      p_norm  = row_p_norm;
      p_resid = row_p_resid;
      d_resid = row_d_resid;
    }
  }

#else

  /*
   * Converge based on aggregate row movement.
   */
  #pragma omp parallel for reduction(+:p_norm, p_resid, d_resid) \
      if(should_parallelize)
  for(idx_t i=0; i < nrows; ++i) {
    for(idx_t j=0; j < ncols; ++j) {
      idx_t const index = j + (i*ncols);
      val_t const pdiff = matv[index] - auxv[index];
      val_t const ddiff = matv[index] - init[index];

      p_norm  += matv[index] * matv[index];
      p_resid += pdiff * pdiff;
      d_resid += ddiff * ddiff;
    }
  }
#endif

  *primal_norm  = p_norm;
  *primal_resid = p_resid;
  *dual_resid   = d_resid;
}



/**
* @brief Optimally update the primal variable using a closed-form solution.
*
* @param[out] primal The matrix to update.
* @param ws CPD workspace.
* @param con The constraint we are enforcing.
*/
static void p_constraint_closedform(
    matrix_t * const primal,
    cpd_ws * const ws,
    splatt_cpd_constraint * con)
{
  /* Modify primal/Gram matrices if necessary. */
  if(con->clsd_func != NULL) {
    idx_t const nrows = primal->I;
    idx_t const ncols = primal->J;
    con->clsd_func(primal->vals, nrows, ncols, con->data);
  }

  mat_cholesky(ws->gram);

  /* Copy and then solve directly against MTTKRP */
  size_t const bytes = primal->I * primal->J * sizeof(*primal->vals);
  par_memcpy(primal->vals, ws->mttkrp_buf->vals, bytes);
  mat_solve_cholesky(ws->gram, primal);
}


static idx_t p_admm_iterate_chunk(
    matrix_t * primal,
    matrix_t * auxil,
    matrix_t * dual,
    matrix_t * cholesky,
    matrix_t * mttkrp_buf,
    matrix_t * init_buf,
    idx_t mode,
    splatt_cpd_constraint * const con,
    val_t const rho,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts,
    bool const should_parallelize)
{
  idx_t const rank = primal->J;

  bool is_spd = mat_cholesky_(ws->gram);

  /* for checking convergence */
  val_t primal_norm     = 0.;
  val_t dual_norm       = 0.;
  val_t primal_residual = 0.;
  val_t dual_residual   = 0.;

  /* foreach inner iteration */
  idx_t it;
  for(it=0; it < cpd_opts->max_inner_iterations; ++it) {
    /* save starting point for convergence check */
    size_t const bytes = primal->I * rank * sizeof(*primal->vals);
    if(should_parallelize) {
      par_memcpy(init_buf->vals, primal->vals, bytes);
    } else {
      memcpy(init_buf->vals, primal->vals, bytes);
    }

    /* auxiliary = MTTKRP + (rho .* (primal + dual)) */
    p_setup_auxiliary(primal, mttkrp_buf, dual, rho, auxil,
        should_parallelize);

    /* Cholesky against auxiliary */
    // mat_solve_cholesky(ws->gram, auxil);
    mat_solve_cholesky_with_fallback(ws->gram, auxil, is_spd); 
   
    p_setup_proximity(primal, auxil, dual, should_parallelize);

    /* APPLY CONSTRAINT / REGULARIZATION */
    if(con->prox_func != NULL) {
      con->prox_func(primal->vals, primal->I, rank, 0, con->data, rho,
          should_parallelize);
    } else {
      fprintf(stderr, "SPLATT: WARNING no proximity operator specified for "
                      "constraint '%s'\n.", con->description);
    }

    /* update dual: U += (primal - auxiliary) */
    dual_norm = p_update_dual(primal, auxil, dual, should_parallelize);

    /* check ADMM convergence */
    p_calc_residual(primal, auxil, init_buf, &primal_norm, &primal_residual,
        &dual_residual, should_parallelize);

    /* converged? */
    if((primal_residual <= cpd_opts->inner_tolerance * primal_norm) &&
       (dual_residual   <= cpd_opts->inner_tolerance * dual_norm)) {
      ++it;
      break;
    }
  }

  return it;
}

val_t admm_stream_inner_maxcolnorm(
    matrix_t * primal_mat,
    matrix_t * auxil_mat,
    matrix_t * dual_mat,
    matrix_t * cholesky_mat,
    matrix_t * mttkrp_buf,
    matrix_t * init_buf,
    idx_t chunk_size,
    splatt_cpd_constraint * const con,
    val_t const rho,
    splatt_cpd_opts const * const cpd_opts)
{
  idx_t rank = primal_mat->J;
  idx_t niter;

  /* for checking convergence */
  val_t p_norm = 0.;
  val_t d_norm = 0.;
  val_t p_res  = 0.;
  val_t d_res  = 0.;

  chunk_size = 64;
  idx_t num_chunks =  (primal_mat->I / chunk_size);
  if(primal_mat->I % chunk_size > 0) {
    ++num_chunks;
  }

  bool is_spd = mat_cholesky_(cholesky_mat);

  #pragma omp parallel shared(p_norm,d_norm,p_res,d_res,niter)
  {
    int tid = omp_get_thread_num();

    val_t * restrict norms = (val_t*) splatt_malloc(rank*sizeof(val_t));
    val_t * restrict colnorms = (val_t*) splatt_malloc(rank*sizeof(val_t));
    memset(norms, 0, rank*sizeof(val_t));
    memset(colnorms, 0, rank*sizeof(val_t));

    /*
    __assume_aligned(norms, 64);
    __assume_aligned(colnorms, 64);
    */

    #pragma omp for
    for(idx_t c=0; c < num_chunks; ++c) {
      idx_t const start = c * chunk_size;
      idx_t const stop = (c == num_chunks-1) ? primal_mat->I : (c+1)*chunk_size;
      idx_t const offset = start * rank;
      idx_t const nrows = stop - start;
      idx_t const ncols = rank;

      /* extract all the workspaces per chunk */
      val_t * const restrict primal = primal_mat->vals + offset;
      val_t * const restrict auxil = auxil_mat->vals + offset;
      val_t * const restrict dual = dual_mat->vals + offset;
      val_t * const restrict mttkrp = mttkrp_buf->vals + offset;
      val_t * const restrict init = init_buf->vals + offset;
      /*
      __assume_aligned(primal, 64);
      __assume_aligned(auxil, 64);
      __assume_aligned(dual, 64);
      __assume_aligned(mttkrp, 64);
      __assume_aligned(init, 64);
      */

      matrix_t auxil_chunk_mat;
      mat_fillptr(&auxil_chunk_mat, auxil, nrows, rank, auxil_mat->rowmajor);
      // row-wise/vector-wise fused formation of rhs
      #pragma simd
      #pragma vector aligned
      for (idx_t idx = 0; idx < ncols*nrows; ++idx) {
        auxil[idx] = mttkrp[idx] + rho*(primal[idx] + dual[idx]);
      }

      // chunk solve chol
      // mat_solve_cholesky(cholesky_mat, &auxil_chunk_mat);
      mat_solve_cholesky_with_fallback(cholesky_mat, &auxil_chunk_mat, is_spd);

      // form prox and compute new norm
      for (idx_t i = 0; i < nrows; ++i) {
        for (idx_t j = 0; j < ncols; ++j) {
          idx_t idx = j + i*ncols;
          val_t x = auxil[idx] - dual[idx];
          init[idx] = x; // primal
          // TODO: compute colnorm and perform possible thresholding (non-neg)
          colnorms[j] += x * x;
        }
      }
    }

    /* reduce norms */
    #pragma omp barrier
    thread_allreduce(colnorms, rank, SPLATT_REDUCE_SUM);

    for (idx_t j=0; j < rank; ++j) {
      colnorms[j] = sqrt(colnorms[j]);
      colnorms[j] = (colnorms[j] > 1.) ? colnorms[j] : 1.;
    }

    memcpy(norms, colnorms, rank*sizeof(val_t));
    memset(colnorms, 0, rank*sizeof(val_t));

    idx_t it;
    int do_break = 0;
    for(it=0; it < cpd_opts->max_inner_iterations; ++it) {
      { p_res = 0; d_res = 0; p_norm = 0; d_norm = 0; }

      #pragma omp for reduction(+:p_norm,p_res,d_norm,d_res)
      for(idx_t c=0; c < num_chunks; ++c) {
        idx_t const start = c * chunk_size;
        idx_t const stop = (c == num_chunks-1) ? primal_mat->I : (c+1)*chunk_size;
        idx_t const offset = start * rank;
        idx_t const nrows = stop - start;
        idx_t const ncols = rank;

        /* extract all the workspaces per chunk */
        val_t * const restrict primal = primal_mat->vals + offset;
        val_t * const restrict auxil = auxil_mat->vals + offset;
        val_t * const restrict dual = dual_mat->vals + offset;
        val_t * const restrict mttkrp = mttkrp_buf->vals + offset;
        val_t * const restrict init = init_buf->vals + offset;
        /*
        __assume_aligned(primal, 64);
        __assume_aligned(auxil, 64);
        __assume_aligned(dual, 64);
        __assume_aligned(mttkrp, 64);
        __assume_aligned(init, 64);
        */

        matrix_t auxil_chunk_mat;
        mat_fillptr(&auxil_chunk_mat, auxil, nrows, rank, auxil_mat->rowmajor);

        // vectorized loop?
        // form prox and compute new norm
          // TODO:
          // instead do inner loop of some vector blocksize (64 bytes)
          // - duplicate norms to at least (B + ncols)
          // - ncols % blocksize  remainder, how to handle?
          // - block b is element b*B, which is column: b*B % ncols
        for (idx_t i = 0; i < nrows; ++i) {
          for (idx_t j = 0; j < ncols; ++j) {
            idx_t idx = j + i*ncols;
            init[idx] /= norms[j];
          }
        }

        const idx_t cs = nrows*ncols;
        #pragma simd
        #pragma vector aligned
        for (idx_t idx = 0; idx < cs; ++idx) {

          // compute new primal and dual residual
          val_t x = init[idx];
          val_t pdiff = x - primal[idx];
          d_res += pdiff*pdiff;
          primal[idx] = x;

          // update primal norm
          p_norm += x*x;

          // update dual U <- U + (pri - aux)
          val_t y =  x - auxil[idx];
          val_t di = dual[idx] + y;
          dual[idx] = di;

          // update dual norm and primal residual
          d_norm += di*di;
          p_res += y*y;

          // form next RHS for cholesky
          auxil[idx] = mttkrp[idx] + rho*(x + di);
        }

        // chunk solve chol
        // mat_solve_cholesky(cholesky_mat, &auxil_chunk_mat);
        mat_solve_cholesky_with_fallback(cholesky_mat, &auxil_chunk_mat, is_spd);

        // form prox and compute new norm
        for (idx_t i = 0; i < nrows; ++i) {
          for (idx_t j = 0; j < ncols; ++j) {
            idx_t idx = j + i*ncols;
            val_t x = auxil[idx] - dual[idx];
            init[idx] = x;
            // TODO: compute colnorm and perform possible thresholding (non-neg)
            colnorms[j] += x * x;
          }
        }
      }

      #pragma omp barrier

      /* check convergence */
      if((p_res <= cpd_opts->inner_tolerance * p_norm) &&
         (d_res   <= cpd_opts->inner_tolerance * d_norm)) {
        ++it;
        break;
      }

      /* reduce norms */
      thread_allreduce(colnorms, rank, SPLATT_REDUCE_SUM);

      for (idx_t j=0; j < rank; ++j) {
        colnorms[j] = sqrt(colnorms[j]);
        colnorms[j] = (colnorms[j] > 1.) ? colnorms[j] : 1.;
      }


      memcpy(norms, colnorms, rank*sizeof(val_t));
      memset(colnorms, 0, rank*sizeof(val_t));

    } /* admm iteration */

    #pragma omp master
    {
      niter = it;
    }

  } /* omp parallel */

  return niter;
}

val_t admm_stream(
    idx_t mode,
    matrix_t * * mats,
    val_t * const restrict column_weights,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const rank = mats[mode]->J;
  splatt_cpd_constraint * con = cpd_opts->constraints[mode];
  /* (A^T * A) .* (B^T * B) .* .... ) */
  mat_form_gram(ws->aTa, ws->gram, ws->nmodes, mode);
  if(con->gram_func != NULL) {
    con->gram_func(ws->gram->vals, rank, con->data);
  }

  /* these can be solved optimally without ADMM iterations */
  if(con->solve_type == SPLATT_CON_CLOSEDFORM) {
    p_constraint_closedform(mats[mode], ws, con);

    /* Absorb columns into column_weights if no constraints are applied */
    if(ws->unconstrained) {
      mat_normalize(mats[mode], column_weights);
    }
    return 0.;
  }

  /* Add penalty to diagonal -- value taken from AO-ADMM paper */
  val_t const rho = mat_trace(ws->gram) / (val_t) rank;
  mat_add_diag(ws->gram, rho);

  /* Compute Cholesky factorization to use for forward/backward solves each
   * ADMM iteration */
  // mat_cholesky(ws->gram);

  /* Compute number of chunks */
  idx_t const chunk_size = cpd_opts->chunk_sizes[mode];

  idx_t niter = admm_stream_inner_maxcolnorm(
                    mats[mode], ws->auxil, ws->duals[mode], ws->gram,
                    ws->mttkrp_buf, ws->mat_init, chunk_size, con, rho, cpd_opts);

  /* return #iterations */
  return (val_t) niter;
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void closedform_solve(
    matrix_t * const primal,
    matrix_t * const gram,
    cpd_ws * const ws)
{

  mat_cholesky(gram);

  /* Copy and then solve directly against MTTKRP */
  size_t const bytes = primal->I * primal->J * sizeof(*primal->vals);
  par_memcpy(primal->vals, ws->mttkrp_buf->vals, bytes);
  mat_solve_cholesky(gram, primal);
}

val_t admm_(
    idx_t mode,
    matrix_t * * mats,
    val_t * const restrict column_weights,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const rank = mats[mode]->J;
  splatt_cpd_constraint * con = cpd_opts->constraints[mode];
  /* (A^T * A) .* (B^T * B) .* .... ) */
  mat_form_gram(ws->aTa, ws->gram, ws->nmodes, mode);
  
  if(con->gram_func != NULL) {
    con->gram_func(ws->gram->vals, rank, con->data);
  }
  /* these can be solved optimally without ADMM iterations */ 
  if(con->solve_type == SPLATT_CON_CLOSEDFORM) {
    p_constraint_closedform(mats[mode], ws, con);

    /* Absorb columns into column_weights if no constraints are applied */
    if(ws->unconstrained) {
      mat_normalize(mats[mode], column_weights);
    }
    return 0.;
  }
  /* Add penalty to diagonal -- value taken from AO-ADMM paper */
  val_t const rho = mat_trace(ws->gram) / (val_t) rank;
  mat_add_diag(ws->gram, rho);

  /* Compute Cholesky factorization to use for forward/backward solves each
   * ADMM iteration */
  // mat_cholesky(ws->gram);

  /* Compute number of chunks */
  idx_t num_chunks = 1;
  idx_t const chunk_size = cpd_opts->chunk_sizes[mode];
  if(con->hints.row_separable && chunk_size > 0) {
    num_chunks =  (mats[mode]->I / chunk_size);
    if(mats[mode]->I % chunk_size > 0) {
      ++num_chunks;
    }
  }

  idx_t it = 0;
  #pragma omp parallel for schedule(dynamic) reduction(+:it) if(num_chunks > 1)
  for(idx_t c=0; c < num_chunks; ++c) {
    idx_t const start = c * chunk_size;
    idx_t const stop = (c == num_chunks-1) ? mats[mode]->I : (c+1)*chunk_size;
    idx_t const offset = start * rank;
    idx_t const nrows = stop - start;

    /* sub-matrix chunks */
    matrix_t primal;
    matrix_t auxil;
    matrix_t dual;
    matrix_t mttkrp;
    matrix_t init_buf;

    /* extract all the workspaces */
    mat_fillptr(&primal, mats[mode]->vals + offset, nrows, rank,
        mats[mode]->rowmajor);
    mat_fillptr(&auxil, ws->auxil->vals + offset, nrows, rank,
        ws->auxil->rowmajor);
    mat_fillptr(&dual, ws->duals[mode]->vals + offset, nrows, rank,
        ws->duals[mode]->rowmajor);
    mat_fillptr(&mttkrp, ws->mttkrp_buf->vals + offset, nrows, rank,
        ws->mttkrp_buf->rowmajor);
    mat_fillptr(&init_buf, ws->mat_init->vals + offset, nrows, rank,
        ws->mat_init->rowmajor);

    /* should the ADMM kernels parallelize themselves? */
    bool const should_parallelize = (num_chunks == 1);

    /* Run ADMM until convergence and record total ADMM its per row. */
    idx_t const chunk_iters =  p_admm_iterate_chunk(&primal, &auxil, &dual,
        ws->gram, &mttkrp, &init_buf, mode, con, rho, ws, cpd_opts,
        global_opts, should_parallelize);
    it += chunk_iters * nrows;
  } /* foreach chunk */
  /* return average # iterations */
  return (val_t) it / (val_t) mats[mode]->I;
}
