#ifndef SPLATT_STSORT_HPP
#define SPLATT_STSORT_HPP

extern "C" {
#include "sptensor.h"
}

#include <vector>

#include <omp.h>
#include "cpptimer.hpp"

void colwise_prefix(idx_t* hist, idx_t n, idx_t nrows);
idx_t* tt_bucket_sort(sptensor_t * const tt, const idx_t mode, bool coarsen, idx_t& num_bins);

// tensor std::sort inplace
void tensor_stdsort_inplace(sptensor_t* const tt, idx_t mode);

// tensor std::sort (returns new sorted tensor without modifying the given tensor tt)
void tensor_stdsort(const sptensor_t* const tt, idx_t mode, sptensor_t** out_tt);

template <typename T>
idx_t* my_par_hist(const T* const values, const size_t n, const size_t num_bins, int bin_size_log = 0) {

  // parallel bucketing into the num_chunks buckets
  idx_t * hists;
  int p = omp_get_max_threads();
  timed_section t("alloc hist");
  hists = (idx_t*) splatt_malloc(num_bins*(p+1)*sizeof(idx_t));
  memset(hists+num_bins*p, 0, num_bins*sizeof(idx_t));

  idx_t max_hist = 0; // TODO returnb the max hist bin
  t.new_section("par_hist_count");
  #pragma omp parallel reduction(max:max_hist)
  {
    int tid = omp_get_thread_num();
    assert(p == omp_get_num_threads());
    idx_t* my_hist = hists + num_bins*tid;
    memset(my_hist, 0, num_bins*sizeof(idx_t));

    idx_t np = (n + p - 1) / p;
    idx_t tstart = tid*np;
    idx_t tend = std::min<idx_t>((tid+1)*np, n);

    for (idx_t i = tstart; i < tend; ++i) {
      ++my_hist[values[i] >> bin_size_log];
    }
  }

  t.new_section("colwise prefix");
  colwise_prefix(hists, num_bins, p+1);

  return hists;
}

template <typename T>
std::vector<T> seq_hist(const T* const values, const size_t n, const T max_val, int bin_size_log = 0) {
    size_t num_bins = (max_val >> bin_size_log) + 1;
    std::vector<T> hist(num_bins, 0);
    for (size_t i = 0; i < n; ++i) {
      ++hist[values[i] >> bin_size_log];
    }
    return hist;
}

template <typename T>
void xprefix(std::vector<T>& v) {
  T x = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    x += v[i];
    v[i] = x;
  }
}

template <typename T>
void eprefix(std::vector<T>& v) {
  T x = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    T y = v[i];
    v[i] = x;
    x += y;
  }
}
#endif // SPLATT_STSORT_HPP
