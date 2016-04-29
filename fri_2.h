// declaration of generic Fast Randomized Iteration
// subroutines
// (c) Jonathan Weare 2015

#ifndef _fri_2_h_
#define _fri_2_h_

#include <iostream> 
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>
// #include "fri_index.h"

//---------------------------------------------------------
// Sparse vector entry definition and helper routines
// Declaration
//---------------------------------------------------------

// An entry of a sparse matrix consists of
// an index and a value. Both must have a
// less than operator, and ValType must also
// support multiplication, division, addition,
// subtraction, assignment to zero, and an 
// absolute value function named 'abs'
// with a return value that casts to 'double'
// unambiguously.
template <typename IdxType, typename ValType>
struct SparseEntry {
  IdxType idx;
  ValType val;
};

// Compare two sparse vector entries by value.
// Relies on ValType having a less than comparator.
struct spcomparebyval;

// Compare two sparse vector entries by index.
// Relies on IdxType having a less than comparator.
struct spcomparebyidx;

//---------------------------------------------------------
// Sparse vector class definition and routines
// Declaration
//---------------------------------------------------------

// SparseVector represents a sparse vector implemented
// as an array of SparseEntry values. The underlying
// array is fixed size to avoid costly allocations, 
// this is max_size_, but the size of the actual vector 
// is dynamic and equal to curr_size_. There are no 
// guarantees on SparseEntry values past curr_size_.
// They can be  subscripted as if they were a vector of
// SparseEntry structs and also provide begin() and end()
// based on curr_size_.

// SparseVectors are added using sp_axpy and multiplied
// by sparse matrices using sp_gemv. They can be
// normalized using normalize(vec), cleaned of zero entries
// using remove_zeros(vec), and printed using 
// print_vector(vec). The sum of entry values is available
// through .norm().

template <typename IdxType, typename ValType>
class SparseVector {
public:
  // Number of nonzero entries actually in the vector.
  size_t curr_size_;
  // Number of nonzero entries that could be in the vector.
  // Must not change.
  const size_t max_size_;

  // norm() returns the sum of the current entry magnitudes.
  ValType norm() const;

  // sum() returns the sum of the current entry values.
  ValType sum() const;

  // Constructor taking max_size as an argument and 
  // allocating space for that many SparseEntry structs.
  SparseVector(size_t max_size);

  // Assignment by value up to current size, leaving
  // max size & other entries unchanged.
  inline SparseVector<IdxType, ValType>& operator=(const SparseVector<IdxType, ValType> &other);

  // Accessors for the underlying vector so one can do
  // some of the normal vector manipulation, but not all.
  // Specifically, subscripting and iterator begin and end
  // are provided. Incremental queue and stack function is 
  // intentionally not provided.
  inline SparseEntry<IdxType, ValType>& operator[](const size_t idx) {return entries_[idx];}
  inline const SparseEntry<IdxType, ValType>& operator[](const size_t idx) const {return entries_[idx];}
  typedef typename std::vector<SparseEntry<IdxType, ValType>>::iterator sp_iterator;
  inline sp_iterator begin() {return entries_.begin();}
  inline sp_iterator end() {return entries_.begin() + curr_size_;}

private:
  // Do not allow manual resizing and pushing/popping of the entries vector.
  std::vector<SparseEntry<IdxType, ValType>> entries_;
};

// Normalize a sparse vector by the sum of its entry vals.
template <typename IdxType, typename ValType>
inline void normalize(SparseVector<IdxType, ValType> &vec);

// Remove all zero elements from a sparse vector.
template <typename IdxType, typename ValType>
inline void remove_zeros(SparseVector<IdxType, ValType> &vec);

// Print a vector to cout by printing the number 
// of nonzero entries and then printing each entry
// as a val idx pair. 
template <typename IdxType, typename ValType>
inline void print_vector(SparseVector<IdxType, ValType> &vec);

// Perform  y <- α x + y.
// y must be large enough to contain 
// all entries of x and y; no allocation
// will be performed by this routine.
template <typename IdxType, typename ValType>
inline int sparse_axpy(ValType alpha,
         const SparseVector<IdxType, ValType> &x,
         SparseVector<IdxType, ValType> &y);

// sparse_gemv computes y <-- alpha A x + beta y
// where x and y are sparse vectors and the matrix 
// A is specified by a routine Acolumn that returns 
// a single column of A.  bw is an
// upper bound on the number of non-zero entries in
// any column of A.  y is assumed to be of max_size
// at least y.curr_size_ + bw x.curr_size_ if beta 
// is nonzero, and bw x.curr_size_ otherwise.
// y is overwritten upon output.
template <typename IdxType, typename ValType>
inline int sparse_gemv(ValType alpha,
        int (*Acolumn)(SparseVector<IdxType, ValType> &col, const IdxType jj),
        size_t max_nz_col_entries,
        const SparseVector<IdxType, ValType> &x, ValType beta,
        SparseVector<IdxType, ValType> &y);

template <typename IdxType, typename ValType>
inline int sparse_gemv_cmp(ValType alpha,
        int (*Acolumn)(SparseVector<IdxType, ValType> &col, const IdxType jj),
        size_t max_nz_col_entries,
        const SparseVector<IdxType, ValType> &x, ValType beta,
        SparseVector<IdxType, ValType> &y, std::vector<size_t> &col_locs);

//---------------------------------------------------------
// Sparse vector compression helper class and routines
// Declaration
//---------------------------------------------------------

// Sparse vector compression class.
template <typename IdxType, typename ValType>
class Compressor {
private:
  
  // Temporary storage for an intermediate.
  SparseVector<size_t, double> xabs_;
  
  // Random number generator.
  std::mt19937_64 gen_;
  std::uniform_real_distribution<> uu_;
  
  // Do not allow copying or assignment for compressors.
  Compressor<IdxType, ValType>(Compressor<IdxType, ValType> &);
  Compressor<IdxType, ValType>& operator= (Compressor<IdxType, ValType> &);
  
  // Helper function: compress the internal 
  // vector of moduli.
  inline void compress_xabs_sys(size_t target_nnz);
  inline void compress_xabs_dmc(size_t target_nnz);

public:

  // Constructor based on maximum size of the
  // modulus vector and a random seed.
  inline Compressor<IdxType, ValType>(size_t max_size, size_t seed) : 
    xabs_(SparseVector<size_t, double>(max_size)) {
    // Set up the pseudorandom number generator.
    gen_ = std::mt19937_64(seed);
    uu_ = std::uniform_real_distribution<>(0,1);
  }
  
  // Compress a SparseVector using the stored
  // temp vector and pseudorandom number generator.
  void compress(SparseVector<IdxType, ValType> &x, size_t target_nnz);
};

//---------------------------------------------------------
// Sparse vector entry definition and helper routines
// Implementation
//---------------------------------------------------------

// Compare two sparse vector entries by value.
// Relies on ValType having a less than comparator.
struct spcomparebyval {
  template <typename IdxType, typename ValType>
  inline bool operator () (const SparseEntry<IdxType, ValType> &a, const SparseEntry<IdxType, ValType> &b) {
    return a.val < b.val;
  }
};

// Compare two sparse vector entries by index.
// Relies on IdxType having a less than comparator.
struct spcomparebyidx {
  template <typename IdxType, typename ValType>
  inline bool operator () (const SparseEntry<IdxType, ValType> &a, const SparseEntry<IdxType, ValType> &b) {
    return a.idx < b.idx;
  }
};

//---------------------------------------------------------
// Sparse vector class definition and routines
// Implementation
//---------------------------------------------------------

  // norm() returns the sum of the magnitudes of the current entry values.
template <typename IdxType, typename ValType>
inline ValType SparseVector<IdxType, ValType>::norm() const {
  ValType norm = 0;
  for (size_t i = 0; i < curr_size_; i++) {
    norm += abs(entries_[i].val);
  }
  return norm;
}

  // sum() returns the sum of the current entry values.
template <typename IdxType, typename ValType>
inline ValType SparseVector<IdxType, ValType>::sum() const {
  ValType sum = 0;
  for (size_t i = 0; i < curr_size_; i++) {
    sum += entries_[i].val;
  }
  return sum;
}

  // Constructor taking max_size as an argument and 
  // allocating space for that many SparseEntry structs.
template <typename IdxType, typename ValType>
inline SparseVector<IdxType, ValType>::SparseVector(size_t max_size) : max_size_(max_size) {
  curr_size_ = 0;
  entries_ = std::vector<SparseEntry<IdxType, ValType>>(max_size);
}

  // Assignment by value up to current size, leaving
  // max size & other entries unchanged.
template <typename IdxType, typename ValType>
inline SparseVector<IdxType, ValType>& SparseVector<IdxType, ValType>::operator=(const SparseVector<IdxType, ValType> &other) {
  assert(other.curr_size_ <= max_size_);
  curr_size_ = other.curr_size_;
  for (size_t i = 0; i < other.curr_size_; i++) {
    entries_[i] = other.entries_[i];
  }
  return *this;
}

// Normalize a sparse vector by the sum of its entry vals.
template <typename IdxType, typename ValType>
inline void normalize(SparseVector<IdxType, ValType> &vec) {
  ValType norm = vec.norm();
  for (size_t i = 0; i < vec.curr_size_; i++) {
    vec[i].val /= norm;
  }
}

// Remove all zero elements from a sparse vector.
template <typename IdxType, typename ValType>
inline void remove_zeros(SparseVector<IdxType, ValType> &vec) {
  size_t nnz = 0;
  for(size_t jj = 0; jj < vec.curr_size_; jj++){
    if (vec[jj].val != 0){
      vec[nnz] = vec[jj];
      nnz++;
    }
  }
  vec.curr_size_ = nnz;
}

// Print a vector to cout by printing the number 
// of nonzero entries and then printing each entry
// as a val idx pair. 
template <typename IdxType, typename ValType>
inline void print_vector(SparseVector<IdxType, ValType> &vec) {
  std::cout << vec.curr_size_ << std::endl;
  for (size_t jj = 0; jj < vec.curr_size_; jj++) {
    std::cout << vec[jj].val << "\t " << vec[jj].idx << std::endl;
  }
  std::cout << std::endl;
}

// Perform  y <- α x + y.
// y must be large enough to contain 
// all entries of x and y; no allocation
// will be performed by this routine.
template <typename IdxType, typename ValType>
inline int sparse_axpy(ValType alpha,
         const SparseVector<IdxType, ValType> &x,
         SparseVector<IdxType, ValType> &y) {
  // If alpha is not zero, then ensure the
  // result vector y can store the entire sum.
  if (alpha != 0) {
    assert(x.curr_size_ + y.curr_size_ <= y.max_size_);
  // If alpha is zero, this is a do-nothing operation.
  // Short circuit it.
  } else {
    return 0;
  }

  // Shift the entries of y to past where they could possibly
  // reside in the new vector.
  for (size_t jj = y.curr_size_; jj > 0; jj--)
      y[x.curr_size_ + jj - 1] = y[jj];
  // Add in entries of either x or y with smallest indices. 
  size_t y_entries_begin = x.curr_size_;
  size_t y_entries_end = x.curr_size_ + y.curr_size_;
  size_t n_result_entries = 0;
  // Move over the entries in each vector from least
  // index to highest, iterating on the x vector as
  // the driver loop and y as a following iterator.
  for (size_t jj = 0; jj < x.curr_size_; jj++){
    // If there are y entries left and the y entry has lower
    // or equal index, move it in.
    while (y_entries_begin < y_entries_end && y[y_entries_begin].idx <= x[jj].idx){
      y[n_result_entries] = y[y_entries_begin];
      y_entries_begin++;
      n_result_entries++;
    }
    // If the x and y entries had equal index, add the x 
    // entry times alpha to the y entry.
    if (n_result_entries > 0 && x[jj].idx == y[n_result_entries - 1].idx){
      y[n_result_entries - 1].val += alpha * x[jj].val;
    }
    // Otherwise, just move the x entry times alpha in. 
    else {
      y[n_result_entries].idx = x[jj].idx;
      y[n_result_entries].val = alpha * x[jj].val;
      n_result_entries++;
    }
  }
  // If all x entries are handled and y entries remain, move
  // them all in.
  while (y_entries_begin < y_entries_end){
    y[n_result_entries] = y[y_entries_begin];
    y_entries_begin++;
    n_result_entries++;
  }
  // Now all entries are set; the new size is known.
  y.curr_size_ = n_result_entries;
  return 0;
}

// sparse_gemv computes y <-- alpha A x + beta y
// where x and y are sparse vectors and the matrix 
// A is specified by a routine Acolumn that returns 
// a single column of A.  bw is an
// upper bound on the number of non-zero entries in
// any column of A.  y is assumed to be of max_size
// at least y.curr_size_ + bw x.curr_size_ if beta 
// is nonzero, and bw x.curr_size_ otherwise.
// y is overwritten upon output.
template <typename IdxType, typename ValType>
inline int sparse_gemv(ValType alpha,
        int (*Acolumn)(SparseVector<IdxType, ValType> &col, const IdxType jj),
        size_t max_nz_col_entries,
        const SparseVector<IdxType, ValType> &x, ValType beta,
        SparseVector<IdxType, ValType> &y)
{
  // Check for correct size; multiplication must not overflow.
  if (beta != 0) {
    assert(y.curr_size_ + max_nz_col_entries * x.curr_size_ <= y.max_size_);
  } else {
    assert(max_nz_col_entries * x.curr_size_ <= y.max_size_);
  }

  // First find what to add to the result SparseVector
  // from the beta multiplication.
  size_t n_entry_adds;
  if (beta != 0) {
    for (size_t ii = 0; ii < y.curr_size_; ii++) {
      y[ii].val *= beta;
    } 
    n_entry_adds = y.curr_size_;
  } else {
    n_entry_adds = 0;
  }

  // Make a list of all the entries in A scaled
  // by the entry of x corresponding to the column
  // containing the particular entry of A; these
  // will all be added to the result.
  SparseVector<IdxType, ValType> single_row_by_column_adds(max_nz_col_entries);
  for (size_t jj = 0; jj < x.curr_size_; jj++) {
    Acolumn(single_row_by_column_adds, x[jj].idx);
    for(size_t ii = 0; ii < single_row_by_column_adds.curr_size_; ii++){
      y[n_entry_adds].val = x[jj].val * single_row_by_column_adds[ii].val;
      y[n_entry_adds].idx = single_row_by_column_adds[ii].idx;
      n_entry_adds++;
    }
  }
  y.curr_size_ = n_entry_adds;

  // Now take all of those entry additions and resolve
  // them into a single SparseVector by adding up all
  // with the same indices.

  // Sort the list of additions according to their indices
  std::make_heap(y.begin(), y.begin() + y.curr_size_, spcomparebyidx());
  std::sort_heap(y.begin(), y.begin() + y.curr_size_, spcomparebyidx());

  // Sum additions corresponding to like indices,
  // collapsing the list so the indices are unique
  // and all additions are resolved.
  size_t new_num_entries = 0;
  size_t curr_entry_add = 0;
  size_t next_entry_add = 0;
  while (curr_entry_add < y.curr_size_) {
    y[new_num_entries] = y[curr_entry_add];
    next_entry_add = curr_entry_add + 1;
    while (next_entry_add < y.curr_size_ &&
           y[next_entry_add].idx == y[curr_entry_add].idx) {
      y[new_num_entries].val += y[next_entry_add].val;
      next_entry_add++;
    }
    curr_entry_add = next_entry_add;
    new_num_entries++;
  }
  y.curr_size_ = new_num_entries;

  return 0;
}

template <typename IdxType, typename ValType>
inline int sparse_gemv_cmp(ValType alpha,
        int (*Acolumn)(SparseVector<IdxType, ValType> &col, const IdxType jj),
        size_t max_nz_col_entries,
        const SparseVector<IdxType, ValType> &x, ValType beta,
        SparseVector<IdxType, ValType> &y, std::vector<size_t> &col_locs)
{
  // Check for correct size; multiplication must not overflow.
  if (beta != 0) {
    assert(y.curr_size_ + max_nz_col_entries * x.curr_size_ <= y.max_size_);
  } else {
    assert(max_nz_col_entries * x.curr_size_ <= y.max_size_);
  }

  // First find what to add to the result SparseVector
  // from the beta multiplication.
  SparseVector<IdxType, size_t> entry_to_Idx(y.max_size_);
  size_t n_entry_adds;
  if (beta != 0) {
    for (size_t ii = 0; ii < y.curr_size_; ii++) {
      y[ii].val *= beta;
      entry_to_Idx[ii].val = ii;
      entry_to_Idx[ii].idx = y[ii].idx;
    } 
    n_entry_adds = y.curr_size_;
  } else {
    n_entry_adds = 0;
  }

  size_t kk, n_entry_selected;
  double w, U, column_sum, s;
  std::mt19937_64 gen_;
  std::uniform_real_distribution<> uu_;

  // Make a list of all the entries in A scaled
  // by the entry of x corresponding to the column
  // containing the particular entry of A; these
  // will all be added to the result.
  SparseVector<IdxType, ValType> single_row_by_column_adds(max_nz_col_entries);
  col_locs.clear();
  cout << col_locs.size() << "\n";
  for (size_t jj = 0; jj < x.curr_size_; jj++) {
    col_locs.push_back(n_entry_adds);
    Acolumn(single_row_by_column_adds, x[jj].idx);
    for(size_t ii = 0; ii < single_row_by_column_adds.curr_size_; ii++){
      entry_to_Idx[n_entry_adds].val = n_entry_adds;
      entry_to_Idx[n_entry_adds].idx = single_row_by_column_adds[ii].idx;
      y[n_entry_adds].val = x[jj].val * single_row_by_column_adds[ii].val;
      y[n_entry_adds].idx = single_row_by_column_adds[ii].idx;
      n_entry_adds++;
    }
  }
  y.curr_size_ = n_entry_adds;
  entry_to_Idx.curr_size_ = n_entry_adds;
  col_locs.push_back(n_entry_adds);

  // Now take all of those entry additions and resolve
  // them into a single SparseVector by adding up all
  // with the same indices.

  // Sort the list of additions according to their indices
  std::make_heap(entry_to_Idx.begin(), entry_to_Idx.begin() + entry_to_Idx.curr_size_, spcomparebyidx());
  std::sort_heap(entry_to_Idx.begin(), entry_to_Idx.begin() + entry_to_Idx.curr_size_, spcomparebyidx());

  // Sum additions corresponding to like indices,
  // collapsing the list so the indices are unique
  // and all additions are resolved.
  size_t curr_entry_add = 0;
  size_t next_entry_add = 0;
  size_t ii;
  ValType temp_val; 
  while (curr_entry_add < y.curr_size_) {
    temp_val = y[entry_to_Idx[curr_entry_add].val].val;
    next_entry_add = curr_entry_add + 1;
    while (next_entry_add < y.curr_size_ &&
           entry_to_Idx[next_entry_add].idx == entry_to_Idx[curr_entry_add].idx) {
      temp_val += y[entry_to_Idx[next_entry_add].val].val;
      next_entry_add++;
    }
    for( ii=curr_entry_add;ii<next_entry_add;ii++){
      y[entry_to_Idx[ii].val].val = temp_val/(next_entry_add-curr_entry_add);
    }
    curr_entry_add = next_entry_add;
  }

  return 0;
}

//---------------------------------------------------------
// Sparse vector compression helper class and routines
// Implementation
//---------------------------------------------------------

using std::abs;

// Compress a SparseVector using the stored
// temp vector and pseudorandom number generator.
template <typename IdxType, typename ValType>
inline void Compressor<IdxType, ValType>::compress(SparseVector<IdxType, ValType> &x, size_t target_nnz) {
  // Copy the modulus of each entry into xabs_.
  assert(x.curr_size_ <= xabs_.max_size_);
  size_t ii = 0;
  for (size_t jj = 0; jj< x.curr_size_; jj++){
    xabs_[ii].val = abs(x[jj].val);
    xabs_[ii].idx = jj;
    if (xabs_[ii].val>0) ii++;
  }
  xabs_.curr_size_ = ii;
  // Compress the moduli vector.
  compress_xabs_sys(target_nnz);
  // Translate the compression of the moduli vector to
  // a compression of the input vector. For each entry
  // of the compressed xabs,
  for(size_t jj = 0; jj < xabs_.curr_size_; jj++){
    // Find the corresponding member of x and
    // set its modulus according to the modulus
    // of xabs.
    ii = xabs_[jj].idx;
    assert(x[ii].val != 0);
    x[ii].val = (x[ii].val / abs(x[ii].val)) * xabs_[jj].val;
  }
  // Remove the entries set to zero.
  remove_zeros(x);
  // The vector is now compressed.
}

template <typename IdxType, typename ValType>
inline void Compressor<IdxType, ValType>::compress_xabs_sys(size_t target_nnz) {
  // First count the number of actually
  // non-zero entries and check that
  // all entries are non-negative.
  double Tol = 1e-12;
  size_t nnz = xabs_.curr_size_;
  double initial_sum = xabs_.norm();
  for (size_t ii = 0; ii < xabs_.curr_size_; ii++) {
    assert(xabs_[ii].val>= 0);
    if (xabs_[ii].val < Tol * initial_sum) {
      nnz -= 1;
      xabs_[ii].val = 0.0;
    }
  }

  // cout << nnz << "\t" << target_nnz << "\n";
    
  // If there are already fewer than n nonzero
  // entries, no compression is needed.
  if (nnz <= target_nnz) {
    return;
  // Otherwise, perform compression.
  } else {
      
    // Find the maximum and storage position of the maximum.
    double dmax = 0;
    size_t imax;
    for (size_t ii = 0; ii < xabs_.curr_size_; ii++) {
      if (xabs_[ii].val > dmax) {
        imax = ii;
        dmax = xabs_[ii].val;
      }
    }
    // Place it at the end of the stored vector;
    // we are building a new vector in place by
    // transferring entries within the old one.
    std::swap( xabs_[xabs_.curr_size_ - 1], xabs_[imax]);
    
    // Check if there are any elements large
    // enough to be preserved exactly.  If so
    // heapify and pull large entries until remaining
    // entries are not large enough to be preserved
    // exactly.
    size_t nnz_large = 0;
    double sum_unprocessed;
    initial_sum = sum_unprocessed = xabs_.sum();
    if (target_nnz * dmax > sum_unprocessed) {
      nnz_large = 1;
      sum_unprocessed -= dmax;
      std::make_heap(xabs_.begin(),
                     xabs_.begin() + xabs_.curr_size_ - nnz_large, 
                     spcomparebyval());
    
      while (( (target_nnz - nnz_large) * xabs_[0].val > sum_unprocessed) 
               && (nnz_large < target_nnz)) {
        sum_unprocessed -= xabs_[0].val;
        assert(sum_unprocessed >0);
        std::pop_heap(xabs_.begin(),
                     xabs_.begin() + xabs_.curr_size_ - nnz_large, 
                     spcomparebyval());
        nnz_large++;
      }
    }
    nnz = nnz_large;

    // cout << nnz_large << "\t" << target_nnz << "\n";

    assert(nnz_large < target_nnz);

    // Compute the norm of the vector of
    // remaining small entries.
    size_t n_small_entries = xabs_.curr_size_ - nnz_large;
    double sum_small = 0;
    for (size_t ii = 0; ii < n_small_entries; ii++) {
      sum_small += xabs_[ii].val;
    }


    // Compress remaining small entries, assuming more
    // nonzeros are needed and small entries are not 
    // too small in sum. Entries that are not set to 
    // zero are set to the norm of the vector of remaining
    // small entries divided by the difference between 
    // target_nnz and the number of large entries preserved 
    // exactly.
    size_t target_nnz_from_small = target_nnz - nnz_large;
    double target_small_entry_average = sum_small / (double) target_nnz_from_small;
    size_t jj = 0;
    double w = xabs_[0].val/target_small_entry_average;
    xabs_[0].val = 0;
    double U = uu_(gen_);
    for(size_t ii=0; ii<target_nnz_from_small; ii++ ){
      // double U = uu_(gen_);
      while( w < ii + U ){
        jj++;
        w += xabs_[jj].val / target_small_entry_average;
        xabs_[jj].val = 0;
      }
      xabs_[jj].val += target_small_entry_average;
      assert(xabs_[jj].val <= 2*target_small_entry_average);
    }
    jj++;
    while( jj < n_small_entries){
      xabs_[jj].val = 0;
      jj++; 
    }



    // size_t target_nnz_from_small = target_nnz - nnz_large;
    // double target_small_entry_average = sum_small / (double) target_nnz_from_small;
    // size_t ii;
    // for (ii=0; ii<n_small_entries; ii++){
    //   xabs_[ii].val = target_small_entry_average*floor(xabs_[ii].val/target_small_entry_average + uu_(gen_));
    //   assert( xabs_[ii].val <= target_small_entry_average );
    // }



    // //double Tol = 1e-15;
    // if (nnz_large < target_nnz) {
    //   double w = -uu_(gen_);
    //   size_t nnz_from_small = 0;
    //   size_t target_nnz_from_small = target_nnz - nnz_large;
    //   double target_small_entry_average = sum_small / (double) target_nnz_from_small;
    //   // For every small entry,
    //   size_t ii;
    //   for (ii = 0; ii < n_small_entries; ii++) {
    //     // If there are still more nonzeros needed
    //     // for the compression,
    //     if (nnz < target_nnz) {
    //       // For each entry, add to w an amount equal to the
    //       // current entry value divided by the target average
    //       // entry value.
    //       w += xabs_[ii].val / target_small_entry_average;
    //       // If enough small nonzeros are already 
    //       // kept to exceed the current w, set this
    //       // entry's value to zero.
    //       if (nnz_from_small > (size_t) floor(w)) {
    //         xabs_[ii].val = 0;
    //       // Otherwise, use this small entry to make
    //       // a new nonzero in the compressed vector.
    //       } else {
    //         xabs_[ii].val = target_small_entry_average;
    //         nnz_from_small++;
    //         nnz++;
    //       }
    //     // Otherwise set all remaining entries to zero.
    //     } else {
    //       xabs_[ii].val = 0;
    //     }      
    //   }
    // // If enough large entries were kept exactly or 
    // // the sum of small entries is not large enough,
    // // set all remaining (small) entries to zero.
    // } else {
    //   for (size_t ii = 0; ii < n_small_entries; ii++) {
    //     xabs_[ii].val = 0;
    //   }
    // }
    // if (nnz > target_nnz) {
    //   std::cerr << "Too many nonzeros in compress\n";
    // }

  nnz = 0;
  for (size_t ii = 0; ii < xabs_.curr_size_; ii++) {
    if( xabs_[ii].val > 0) nnz++;
  }

  // cout << nnz_large << "\t" << nnz << "\t" << target_nnz << "\n";
  // assert( nnz == target_nnz );
  }
}

template <typename IdxType, typename ValType>
inline void Compressor<IdxType, ValType>::compress_xabs_dmc(size_t target_nnz) {
  // First count the number of actually
  // non-zero entries and check that
  // all entries are non-negative.
  size_t nnz = xabs_.curr_size_;
  for (size_t ii = 0; ii < xabs_.curr_size_; ii++) {
    assert(xabs_[ii].val >= 0);
    if (xabs_[ii].val == 0) {
      nnz -= 1;
    }
  }
    
  // If there are already fewer than n nonzero
  // entries, no compression is needed.
  // if (nnz <= target_nnz) {
  //   return;
  // // Otherwise, perform compression.
  // } else {
  if (false) {
    return;
  // Otherwise, perform compression.
  } else {

    double target_entry_average = xabs_.norm()/(double) target_nnz;
    // For every small entry,
    for (size_t ii = 0; ii < xabs_.curr_size_; ii++) {
      // For each entry, add to w an amount equal to the
      // current entry value divided by the target average
      // entry value.
      xabs_[ii].val = target_entry_average * floor(xabs_[ii].val/target_entry_average + uu_(gen_));
    }

    // cout << xabs_.norm()*target_nnz << "\n";
    // cout << xabs_[0].val*target_nnz << "\n";

    // double U = uu_(gen_);
    // double w = 0;
    // size_t nnz_added = 0;
    // size_t nnz_before_entry;
    // double target_entry_average = xabs_.norm()/(double) target_nnz;
    // // For every small entry,
    // for (size_t ii = 0; ii < xabs_.curr_size_; ii++) {
    //   // For each entry, add to w an amount equal to the
    //   // current entry value divided by the target average
    //   // entry value.
    //   w += xabs_[ii].val / target_entry_average;
    //   // If enough small nonzeros are already 
    //   // kept to exceed the current w, set this
    //   // entry's value to zero.
    //   nnz_before_entry = nnz_added;
    //   nnz_added = (size_t)ceil(w-U);
    //   // while ((double)nnz_added <= w) nnz_added++;
    //   xabs_[ii].val = target_entry_average * (double)(nnz_added-nnz_before_entry);

    //   // if( xabs_[ii].val>0 ) U = uu_(gen_);  
    // }

    assert(xabs_.norm()>0);
  }
}

#endif