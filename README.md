The Surprisingly ParalleL spArse Tensor Toolkit - STREAMED
==========================================================

**NOTE**: this is the development repository for the _streaming_ version of
SPLATT. Run `splatt cpd --help` for streaming instructions. In short, you can
add the `--stream=<MODE> --reg=frob,1e-2,<MODE>` flag to stream one tensor mode
with regularization (strongly recommended). This is a prototype implementation
of the streaming algorithm, and thus the tensor is still read from disk in its
entirety and then factored in a streaming manner.

The streaming algorithm implemented in this library was published in SDM'18:
"Streaming Tensor Factorization for Infinite Data Sources". If this contributes
to your research, please cite with the following BibTeX entry:

    @inproceedings{smith2018streaming,
      title={Streaming Tensor Factorization for Infinite Data Sources},
      author={Smith, Shaden and Huang, Kejun and Sidiropoulos, Nicholas D. and Karypis, George},
      booktitle={Proceedings of the 2018 SIAM International Conference on Data Mining},
      year={2018},
      organization={SIAM}
    }


SPLATT is a library and C API for sparse tensor factorization. SPLATT supports
shared-memory parallelism with OpenMP and distributed-memory parallelism with
MPI.


Tensor Format
-------------
SPLATT expects tensors to be stored in 0- or 1-indexed coordinate format with
nonzeros separated by newlines. Each line of of the file has the coordinates of
the nonzero  followed by the value, all separated by spaces.  The following is
an example 2x2x3 tensor with 5 nonzeros:

    # This is a comment
    1 1 2 1.5
    1 2 2 2.5
    2 1 1 3.7
    1 2 3 0.5
    2 1 2 4.1


Building & Installing
---------------------
SPLATT requires CMake and working BLAS/LAPACK libraries to run. In short,

    $ ./configure && make

will build the SPLATT library and its executable. SPLATT also provides options
to automatically download the CMake and BLAS/LAPACK dependencies on Linux/OSX
systems:

    $ ./configure --download-cmake --download-blas-lapack

You can also run

    $ ./configure --help

to see additional build options.  After compilation, the executable will be
found in `build/<arch>/bin/`. To install,

    $ make install

will suffice. The installation prefix can be chosen by adding a
'--prefix=DIR' flag to configure.


Executable
----------
After building, an executable will found in the `build/` directory (or the
installation prefix if SPLATT was installed). SPLATT builds a single executable
which features a number of sub-commands:

* cpd
* check
* convert
* reorder
* stats

All SPLATT commands are executed in the form

    $ splatt CMD [OPTIONS]

You can execute

    $ splatt CMD --help

for usage information of each command.

### Example 1

    $ splatt check mytensor.tns  --fix=fixed.tns

This runs `splatt-check` on 'mytensor.tns' and writes the fixed tensor to
'fixed.tns'. The `splatt-check` routine finds empty slices and duplicate
nonzero entries. Empty slices are indices in any mode which do not have any
nonzero entries associated with them. Some SPLATT routines (including CPD)
expect there to be no empty slices, so running `splatt-check` on a new tensor
is recommended.

### Example 2

    $ splatt cpd mytensor.tns -r 25 -t 4

This runs `splatt-cpd` on 'mytensor.tns' and finds a rank-25 CPD of the tensor.
Adding '-t 4' instructs SPLATT to use four OpenMP threads during the
computation. SPLATT will use all available CPU cores by default.  The matrix
factors are written to `modeN.mat` and lambda, the vector for scaling, is
written to `lambda.mat`.

### Example 3 - Streaming CPD

    $ splatt cpd --stream=<MODE> --reg=frob,1e-2,<MODE> -r 25 -t 4 mytensor.tns

Providing the streaming mode as an additional flag will compute Streaming CPD as published in SDM'18 "Streaming Tensor Factorization for Infinite Data Sources". 
As the example command suggests, it is strongly recommended to impose regularization to the streaming mode as it brings more stability. The output factor matrices are named using `mode%n.mat` format in the working directory
    
### Example 4 - Row-sparse Streaming CPD

    $ splatt cpd --stream=<MODE> --reg=frob,1e-2,<MODE> -r 25 -t 4 --use_rsp=true mytensor.tns

Providing an additonal "--use_rsp" flag will execute the spCP-stream as proposed in IPDPS'21 "High Performance Streaming Tensor Decomposition".
Compared to the previously mentioned regular streaming CPD, the row-sparse version is recommended for usage in cases where tensors are extremely large in dimensions and the number of entries are hyper-sparse.

Distributed-Memory Computation
------------------------------
SPLATT can optionally be built with support for distributed-memory systems via
MPI. To add MPI support, simply add "--with-mpi" to the configuration step:

    $ ./configure --with-mpi && make

After building with MPI, `splatt-cpd` can be used as before. Careful
consideration should be given to the mapping of MPI ranks, because each SPLATT
process will by default use all available CPU cores (`$OMP_MAX_THREADS`). We
recommend mapping one rank per CPU socket. The necessary parameters to `mpirun`
vary based on the MPI implementation. For example, OpenMPI supports:

### Example 3

    $ mpirun --map-by ppr:1:socket -np 16 splatt cpd mytensor.tns -r 25 -t 8

This would fully utilize 16 sockets, each with 8 cores to compute a rank-25 CPD
of `mytensor.tns`. To alternatively use one MPI rank per core:

### Example 4

    $ mpirun -np 128 splatt cpd mytensor.tns -r 25 -t 1

This would use 128 processes, with each using only one OpenMP thread.


C/C++ API
---------
SPLATT provides a C API which is callable from C and C++. Installation not only
installs the SPLATT executable, but also the shared library `libsplatt.so` and
the header `splatt.h`. To use the C API, include `splatt.h` and link against
the SPLATT library.

### IO
Unless otherwise noted, SPLATT expects tensors to be stored in the compressed
sparse fiber (CSF) format. SPLATT provides two functions for forming a tensor
in CSF:

* `splatt_csf_load` reads a tensor from a file
* `splatt_csf_convert` converts a tensor from coordinate format to CSF


### Computation
* `splatt_cpd` computes the CPD and returns a Kruskal tensor
* `splatt_default_opts` allocates and returns an options array with defaults


### Cleanup
All memory allocated by the SPLATT API should be freed by these functions:

* `splatt_free_csf` deallocates a list of CSF tensors
* `splatt_free_opts` deallocates a SPLATT options array
* `splatt_free_kruskal` deallocates a Kruskal tensor

### Example
The following is an example usage of the SPLATT API:

    #include <splatt.h>

    /* allocate default options */
    double * cpd_opts = splatt_default_opts();

    /* load the tensor from a file */
    int ret;
    splatt_idx_t nmodes;
    splatt_csf_t * tt;
    ret = splatt_csf_load("mytensor.tns", &nmodes, &tt, cpd_opts);

    /* do the factorization! */
    splatt_kruskal_t factored;
    ret = splatt_cpd_als(tt, 10, cpd_opts, &factored);

    /* do some processing */
    for(splatt_idx_t m = 0; m < nmodes; ++m) {
      /* access factored.lambda and factored.factors[m] */
    }

    /* cleanup */
    splatt_free_csf(tt, cpd_opts);
    splatt_free_kruskal(&factored);
    splatt_free_opts(cpd_opts);


Please see `splatt.h` for further documentation of SPLATT structures and call
signatures.


Octave/Matlab API
-----------------
SPLATT also provides an API callable from Octave and Matlab that wraps the C
API. To compile the interface just enter the `matlab/` directory from either
Octave or Matlab and call `make`.

    >> cd matlab
    >> make

**NOTE:** Matlab uses a version of LAPACK/BLAS with 64-bit integers. Most
LAPACK/BLAS libraries use 32-bit integers, and so SPLATT by default provides
32-bit integers. You should either instruct Matlab to link against a matching
library, or configure SPLATT to also use 64-bit integers during configuration:

    $ ./configure --blas-int=64

Note that this may break usability of the SPLATT executable or API.

Some Matlab versions have issues with linking to applications which use OpenMP
(e.g., SPLATT) due to a limited amount of thread-local storage. This is a
system limitation, not necessarily a software limitation. When calling SPLATT
from Matlab, you may receive an error message:

    dlopen: cannot load any more object with static TLS

Two workarounds for this issue are:
1. Ensure that your OpenMP library is loaded first when starting Matlab.  The
most common OpenMP library is `libgomp.so.1`:

        $ LD_PRELOAD=libgomp.so.1 matlab 

2. Disable OpenMP (at the cost of losing multi-threaded execution):

        $ ./configure --no-openmp


After compilation, the MEX files will be found in the current directory. You
can now call those functions directly:

    >> KT = splatt_cpd('mytensor.tns', 25);

`splatt_cpd` returns a structure with three fields:

* `U` a cell array of the factor matrices
* `lambda` the factor column norms absorbed into a vector
* `fit` quality of the CPD defined by: 1 - (norm(residual) / norm(X))

SPLATT also supports explicitly storing tensors in CSF form to avoid IO times
during successive factorizations,

    >> X = splatt_load('mytensor.tns');
    >> K25 = splatt_cpd(X, 25);
    >> K50 = splatt_cpd(X, 50);

SPLATT accepts non-default parameters via structures:

    >> opts = struct('its', 100, 'tol', 1e-8);
    >> XT = splatt_cpd(X, 25, opts);

Finally, there are several SPLATT routines exposed for developing other tensor
operations. SPLATT provides:

* `splatt_mttkrp`
* `splatt_norm`
* `splatt_dim`
* `splatt_innerprod`

Please see `help <cmd>` from an Octave/Matlab terminal or read <cmd>.m in the
`matlab/` directory for more usage information.


Licensing
---------
SPLATT is released under the MIT License. Please see the 'LICENSE' file for
details.
