#ifndef SPLATT_STMTTKRP_H
#define SPLATT_STMTTKRP_H

// Splatt includes
extern "C" {
#include "matrix.h"
#include "mttkrp.h"
#include "sptensor.h"
}

#include <vector>

/**********************************************************************
*                  Row-Sparse Matrix and Operations                  *
**********************************************************************/

/**
 * @brief Row-Sparse Matrix
 *
 * row-sparse matrix I x J where `nnzr` is the number of nonzero rows
 * represented by a `nnzr x J` matrix and a `nnzr x 1` vector of row indeces
 */
struct rsp_matrix_t {
  idx_t I; // number of rows
  idx_t J; // number of columns (== mat.J)
  idx_t nnzr; // number of nonzero rows (== mat.I)
  matrix_t* mat; // dense matrix (nnzr x J) containing the values
  idx_t* rowind; // array of row indeces
};

/**********************************************************************
*                     Row-Spare Matrix Operations                    *
**********************************************************************/

/**
 * @brief Allocate a (uninitialized) Row-Sparse Matrix
 *
 * @param nrows     Number of rows
 * @param ncols     Number of columns
 * @param nnzr      Number of nonzero rows (only these are saved)
 *
 * @return The newly allocated Row SParse Matrix
 */
rsp_matrix_t* rspmat_alloc(idx_t nrows, idx_t ncols, idx_t nnzr);

/**
* @brief Free a row-sparse matrix allocated with rspmat_alloc().
*
* This also frees the given matrix pointer!
*
* @param mat The row-sparse matrix to be freed.
 */
void rspmat_free(rsp_matrix_t* mat);

/**
 * @brief converts a full matrix to rsp_matrix format
 */
rsp_matrix_t* convert_to_rspmat(
    matrix_t* fm, idx_t nnzr, idx_t* rowind);

/**
 * @brief computes aTa for row sparse matrices
 */
void rsp_mataTb(rsp_matrix_t* A, rsp_matrix_t* B, matrix_t* dest);

/**
 * @brief sum (A+=B) of two row sparse matrices
 */
void rsp_mat_add(rsp_matrix_t* A, rsp_matrix_t* B);

/**
 * @brief product (A*=B) of two row sparse matrices
 */
rsp_matrix_t* rsp_mat_mul(rsp_matrix_t* A, matrix_t* B);


/**********************************************************************
*                       MTTKRP implementations                       *
**********************************************************************/

matrix_t* mttkrp_seq(sptensor_t * const tt, matrix_t ** mats, const idx_t mode);

matrix_t* mttkrp_stream_sort_bin(sptensor_t * const tt, matrix_t ** mats, const idx_t mode);

matrix_t* mttkrp_stream_sort(const sptensor_t * tt, matrix_t ** mats, const idx_t mode);

matrix_t*  mttkrp_stream_idx_sort(sptensor_t * const tt, matrix_t ** mats, const idx_t mode);

rsp_matrix_t* rsp_mttkrp_stream_idx_sort(sptensor_t * const tt, matrix_t ** mats, const idx_t mode);

rsp_matrix_t* rsp_mttkrp_stream(sptensor_t * const tt, matrix_t ** mats, const idx_t mode);
rsp_matrix_t* rsp_mttkrp_stream_with_idx(
    sptensor_t * const tt, matrix_t ** mats, const idx_t mode, 
    std::vector<size_t>& idx, std::vector<size_t>& buckets);

rsp_matrix_t* rsp_mttkrp_stream_rsp(
    sptensor_t * const tt, rsp_matrix_t ** rsp_mats, const idx_t mode, 
    const idx_t stream_mode,
    std::vector<size_t>& idx, 
    std::vector<std::vector<int>>& ridx, 
    std::vector<size_t>& buckets);

rsp_matrix_t* rsp_mttkrp_stream_with_idx(
    sptensor_t * const tt, matrix_t ** mats, const idx_t mode, 
    std::vector<size_t>& idx, std::vector<size_t>& buckets);

void nonzero_slices(
    sptensor_t * const tt, const idx_t mode,
    std::vector<size_t>& nz_rows,
    std::vector<size_t>& idx,
    std::vector<int>& ridx,
    std::vector<size_t>& buckets);

#endif // SPLATT_STMTTKRP_H
