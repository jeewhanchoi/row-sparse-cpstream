/**
* @file include/splatt/mpi.h
* @brief Functions for distributed-memory SPLATT (MPI).
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-10
*/


#ifndef SPLATT_INCLUDE_MPI_H
#define SPLATT_INCLUDE_MPI_H


#include "cpd.h"


/**
* @brief Tensor decomposition schemes.
*/
typedef enum
{
  /** @brief Coarse-grained decomposition is using a separate 1D decomposition
   *         for each mode. */
  SPLATT_DECOMP_COARSE,
  /** @brief Medium-grained decomposition is an 'nmodes'-dimensional
   *         decomposition. */
  SPLATT_DECOMP_MEDIUM,
  /** @brief Fine-grained decomposition distributes work at the nonzero level.
   *         NOTE: requires a partitioning on the nonzeros. */
  SPLATT_DECOMP_FINE
} splatt_decomp_type;


/**
* @brief Communication pattern type. We support point-to-point, and all-to-all
*        (vectorized).
*/
typedef enum
{
  /**
  * @brief Use point-to-point communications (`MPI_Isend`/`MPI_Irecv`) during
  *        major exchanges.
  */
  SPLATT_COMM_POINT2POINT,

  /**
  * @brief Use personalized all-to-all communications (`MPI_Alltoallv`) during
  *        major exchanges.
  */
  SPLATT_COMM_ALL2ALL
} splatt_comm_type;




/**
* @brief Opaque type for MPI communication.
*/
typedef struct _splatt_comm_info splatt_comm_info;



#ifdef __cplusplus
extern "C" {
#endif


/**
\defgroup api_mpi_list List of functions for \splatt MPI.
@{
*/




/**
* @brief Free the memory allocated by `splatt_alloc_comm_info()`.
*
*        NOTE: this function exists whether MPI is enabled or not -- it mildly
*        simplifies SPLATT internals.
*
* @param comm_info The object to free.
*/
void splatt_free_comm_info(
    splatt_comm_info * comm_info);



/*
 * Dummy MPI implementations for when MPI is not enabled.
 */
#ifndef SPLATT_USE_MPI
/**
* @brief Allocate a `splatt_comm_info` structure. This data must be freed with
*        `splatt_free_comm_info()`.
*
*        NOTE: this overloaded API function is only exposed when MPI is *not*
*        enabled.  Otherwise this function accepts an `MPI_Comm` parameter.
*
* @return A new `splatt_comm_info` object.
*/
splatt_comm_info * splatt_alloc_comm_info();
#endif



#ifdef SPLATT_USE_MPI


/**
* @brief Allocate a `splatt_comm_info` structure. This data must be freed with
*        `splatt_free_comm_info()`.
*
* @param comm The MPI communicator to work from (the communicator will be
*             duplicated).
*
* @return A new `splatt_comm_info` object.
*/
splatt_comm_info * splatt_alloc_comm_info(
    MPI_Comm comm);




/**
* @brief Read a tensor from a file and distribute. The distribution will be
*        chosen in order to optimize CPD computation.
*
* @param fname The file to read.
* @param cpd_opts CPD factorization options. Constraints and rank may be used
*                 to optimize the data distribution.
* @param[out] comm_info MPI communication data which will store information
*                       about the distribution. This should have been allocated
*                       by `splatt_alloc_comm_info()`.
*
* @return  A distributed tensor which is optimized for CPD computation.
*/
splatt_coord * splatt_mpi_distribute_cpd(
    char const * const fname,
    splatt_cpd_opts const * const cpd_opts,
    splatt_comm_info * const comm_info);



/**
* @brief Rearrange a distributed tensor in order to optimize computation for a
*        CPD.
*
* @param coord The distributed tensor to rearrange. This may be modified during
*              rearrangement (e.g., sorted) but will not be destroyed.
* @param cpd_opts CPD factorization options. Constraints and rank may be used
*                 to optimize the data distribution.
* @param[out] comm_info MPI communication data which will store information
*                       about the distribution. This should have been allocated
*                       by `splatt_alloc_comm_info()`.
*
* @return  A rearranged 'coord' which is optimized for CPD computation.
*/
splatt_coord * splatt_mpi_rearrange_cpd(
    splatt_coord * const coord,
    splatt_cpd_opts const * const cpd_opts,
    splatt_comm_info * const comm_info);



/**
* @brief Rearrange a distribute tensor into a medium-grained decomposition.
*
* @param coord The distributed tensor to rearrange. This may be modified during
*              rearrangement (e.g., sorted) but will not be destroyed.
* @param rank_dims The dimensions of the Cartesian grid in terms of MPI ranks.
*                  If this is NULL, SPLATT will find a high-quality grid for
*                  you.
* @param[out] comm_info MPI communication data which will store information
*                       about the distribution. This should have been allocated
*                       by `splatt_alloc_comm_info()`.
*
* @return  A rearranged 'coord' which is optimized for CPD computation.
*/
splatt_coord * splatt_mpi_rearrange_medium(
    splatt_coord * const coord,
    int const * const rank_dims,
    splatt_comm_info * const comm_info);




/**
* @brief Load a coordinate tensor from a file. This is a fast-but-simple load
*        which in which Load balance is based on non-zero count. No
*        communication or other heuristics used.
*
* @param fname The file to read.
* @param comm_info The communication structure previously allocated by
*                  `splatt_alloc_comm_info()`.
*
* @return A distributed tensor. Each MPI rank will have roughly the same number
*         of non-zeros.
*/
splatt_coord * splatt_coord_load_mpi(
    char const * const fname,
    splatt_comm_info * const comm_info);




/**
* @brief Read a tensor from a file, distribute among an MPI communicator, and
*        convert to CSF format.
*
* @param fname The filename to read from.
* @param[out] nmodes SPLATT will fill in the number of modes found.
splatt_mpi_coord_loasplatt_mpi_coord_loasplatt_mpi_coord_load @param[out] tensors An array of splatt_csf structure(s). Allocation scheme
*                follows opts[SPLATT_OPTION_CSF_ALLOC].
* @param options An options array allocated by splatt_default_opts(). The
*                distribution scheme follows opts[SPLATT_OPTION_DECOMP].
* @param comm The MPI communicator to distribute among.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_mpi_csf_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_csf ** tensors,
    double const * const options,
    MPI_Comm comm);



/**
* @brief Load a tensor in coordinate from from a file and distribute it among
*        an MPI communicator.
*
* @param fname The file to read from.
* @param[out] nmodes The number of modes in the tensor.
* @param[out] nnz The number of nonzeros in my portion.
* @param[out] inds An array of indices for each mode.
* @param[out] vals The tensor nonzero values.
* @param options SPLATT options array. Currently unused.
* @param comm Which communicator to distribute among.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_mpi_coord_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_idx_t * nnz,
    splatt_idx_t *** inds,
    splatt_val_t ** vals,
    double const * const options,
    MPI_Comm comm);




#endif /* if mpi */

/** @} */

#ifdef __cplusplus
}
#endif



#endif
