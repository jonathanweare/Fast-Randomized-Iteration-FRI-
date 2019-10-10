// declaration of generic Fast Randomized Iteration
// subroutines
// (c) Jonathan Weare 2015

#ifndef _fri_4_h_
#define _fri_4_h_

#include <iostream> 
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>
#include <valarray>
// #include "fri_index.h"

//---------------------------------------------------------
// Sparse vector entry definition and helper routines
// Declaration
//---------------------------------------------------------

// An entry of a sparse vector consists of
// an index and a value. Both must have a
// less than operator, and ValType must also
// support multiplication, division, addition,
// subtraction, assignment to zero, and an 
// absolute value function named 'abs'
// with a return value that casts to 'double'
// unambiguously.
template <typename IdxType, typename ValType>
struct SparseVectorEntry {
  IdxType idx;
  ValType val;
};

// Compare two sparse vector entries by value.
// Relies on ValType having a less than comparator.
struct spveccomparebyval;

// Compare two sparse vector entries by index.
// Relies on IdxType having a less than comparator.
struct spveccomparebyidx;


//---------------------------------------------------------
// Sparse matrix entry definition and helper routines
// Declaration
//---------------------------------------------------------

// An entry of a sparse matrix consists of
// two indices and a value. Each index
// and the value must have a
// less than operator, and ValType must also
// support multiplication, division, addition,
// subtraction, assignment to zero, and an 
// absolute value function named 'abs'
// with a return value that casts to 'double'
// unambiguously.
template <typename IdxType, typename ValType>
struct SparseMatrixEntry {
  IdxType rowidx;
  IdxType colidx;
  ValType val;
};

// Compare two sparse matrix entries by value.
// Relies on ValType having a less than comparator.
struct spmatcomparebyval;

// Compare two sparse matrix entries by lexicographic
// ordering of indices with rowidx first.
// Relies on IdxType having a less than comparator.
struct spmatcomparebyrowidxfirst;

// Compare two sparse matrix entries by lexicographic
// ordering of indices with colidx first.
// Relies on IdxType having a less than comparator.
struct spmatcomparebycolidxfirst;

//---------------------------------------------------------
// Sparse vector class definitions and routines
// Declaration
//---------------------------------------------------------

// SparseVector represents a sparse vector implemented
// as an array of SparseVectorEntry values. The underlying
// array is fixed size to avoid costly allocations, 
// this is max_size_, but the size of the actual vector 
// is dynamic and equal to curr_size_. There are no 
// guarantees on SparseEntry values past curr_size_.
// They can be  subscripted as if they were a vector of
// SparseVectorEntry structs and also provide begin() and end()
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
  // Number of nonzero entries that could be in the vector.
  // Must not change.
  const size_t max_size_;

  // norm() returns the sum of the current entry magnitudes.
  inline const ValType norm() const;

  // sum() returns the sum of the current entry values.
  inline const ValType sum() const;

  // Constructor taking max_size as an argument and 
  // allocating space for that many SparseEntry structs.
  SparseVector(size_t max_size);

  void remove_zeros();

  inline void clear();

  inline const size_t size() const {return curr_size_;}

  //void normalize();

  inline void print_vector();

  // Assignment by value up to current size, leaving
  // max size & other entries unchanged.
  inline SparseVector<IdxType, ValType>& operator=(const SparseVector<IdxType, ValType> &other);

  inline SparseVector<IdxType, ValType>& operator*=(const ValType alpha);
  inline SparseVector<IdxType, ValType>& operator/=(const ValType alpha);
  inline SparseVector<IdxType, ValType>& operator+=(const ValType alpha);
  inline SparseVector<IdxType, ValType>& operator-=(const ValType alpha);

  inline SparseVector<IdxType, ValType>& operator+=(const SparseVector<IdxType, ValType> &other);
  inline SparseVector<IdxType, ValType>& operator-=(const SparseVector<IdxType, ValType> &other);

  // Accessors for the underlying vector so one can do
  // some of the normal vector manipulation, but not all.
  // Specifically, subscripting and iterator begin and end
  // are provided. Incremental queue and stack function is 
  // intentionally not provided.
  //inline ValType& operator[](const size_t idx) {return entries_[idx].val;}
  inline const SparseVectorEntry<IdxType,ValType>& operator[](const size_t idx) const {return entries_[idx];}
  inline void set_entry(const SparseVectorEntry<IdxType,ValType>& other);
  inline void set_entry(const IdxType idx, const ValType val);
  inline void set_value(const size_t idx, const ValType val);
  //inline const SparseVectorEntry<IdxType, ValType>& get_entry(const size_t idx) const {return entries_[idx];}
  typedef typename std::vector<SparseVectorEntry<IdxType, ValType>>::iterator spvec_iterator;
  inline spvec_iterator begin() {return entries_.begin();}
  inline spvec_iterator end() {return entries_.begin() + curr_size_;}

private:
  // Number of nonzero entries actually in the vector.
  size_t curr_size_;
  // Do not allow manual resizing and pushing/popping of the entries vector.
  std::vector<SparseVectorEntry<IdxType, ValType>> entries_;
};

// // Normalize a sparse vector by the sum of its entry vals.
// template <typename IdxType, typename ValType>
// inline void normalize(SparseVector<IdxType, ValType> &vec);

// // Remove all zero elements from a sparse vector.
// template <typename IdxType, typename ValType>
// inline void remove_zeros(SparseVector<IdxType, ValType> &vec);

// Print a vector to cout by printing the number 
// of nonzero entries and then printing each entry
// as a val idx pair. 
// template <typename IdxType, typename ValType>
// inline void print_vector(SparseVector<IdxType, ValType> &vec);

// Perform  y <- α x + y.
// y must be large enough to contain 
// all entries of x and y; no allocation
// will be performed by this routine.
// template <typename IdxType, typename ValType>
// inline int sparse_axpy(const ValType alpha,
//          const SparseVector<IdxType, ValType> &x,
//          SparseVector<IdxType, ValType> &y);

// Perform  y <- α x + y.
// y must be large enough to contain 
// all entries of x and y; no allocation
// will be performed by this routine.
// template <typename IdxType, typename ValType>
// inline int sparse_axpy_v2(ValType alpha,
//          const SparseVector<IdxType, ValType> &x,
//          SparseVector<IdxType, ValType> &y);

// sparse_gemv computes y <-- alpha A x + beta y
// where x and y are sparse vectors and the matrix 
// A is specified by a routine Acolumn that returns 
// a single column of A.  bw is an
// upper bound on the number of non-zero entries in
// any column of A.  y is assumed to be of max_size
// at least y.curr_size_ + bw x.curr_size_ if beta 
// is nonzero, and bw x.curr_size_ otherwise.
// y is overwritten upon output.
// template <typename IdxType, typename ValType>
// inline int sparse_gemv(ValType alpha,
//         int (*Acolumn)(SparseVector<IdxType, ValType> &col, const IdxType jj),
//         size_t max_nz_col_entries,
//         const SparseVector<IdxType, ValType> &x, ValType beta,
//         SparseVector<IdxType, ValType> &y);













//---------------------------------------------------------
// Sparse vector entry definition and helper routines
// Implementation
//---------------------------------------------------------

// Compare two sparse vector entries by value.
// Relies on ValType having a less than comparator.
struct spveccomparebyval {
  template <typename IdxType, typename ValType>
  inline bool operator () (const SparseVectorEntry<IdxType, ValType> &a, const SparseVectorEntry<IdxType, ValType> &b) {
    return a.val < b.val;
  }
};

// Compare two sparse vector entries by index.
// Relies on IdxType having a less than comparator.
struct spveccomparebyidx {
  template <typename IdxType, typename ValType>
  inline bool operator () (const SparseVectorEntry<IdxType, ValType> &a, const SparseVectorEntry<IdxType, ValType> &b) {
    return a.idx < b.idx;
  }
};









//---------------------------------------------------------
// Sparse vector class definition and routines
// Implementation
//---------------------------------------------------------

template <typename IdxType, typename ValType>
inline void SparseVector<IdxType, ValType>::set_entry(const SparseVectorEntry<IdxType,ValType>& other){

  assert(curr_size_<max_size_);

  if ( curr_size_ == 0){
    entries_[0] = other;
    curr_size_++;
    return;
  }

  if( other.idx > entries_[curr_size_-1].idx){
    entries_[curr_size_] = other;
    curr_size_++;
    return;
  }

  spvec_iterator pos;

  pos = std::lower_bound(entries_.begin(), entries_.begin()+curr_size_, other, spveccomparebyidx());

  if( (*pos).idx == other.idx ){
    (*pos).val = other.val;
  }
  else{
    std::move_backward(pos, entries_.begin()+curr_size_, entries_.begin()+curr_size_+1);
    *pos = other;
    curr_size_++;
  }

return;
}

template <typename IdxType, typename ValType>
inline void SparseVector<IdxType, ValType>::set_entry(const IdxType idx, const ValType val){

  assert(curr_size_<max_size_);

  if ( curr_size_ == 0){
    entries_[0].idx = idx;
    entries_[0].val = val;
    curr_size_++;
    return;
  }

  if( idx > entries_[curr_size_-1].idx){
    entries_[curr_size_].idx = idx;
    entries_[curr_size_].val = val;
    curr_size_++;
    return;
  }

  spvec_iterator pos;

  SparseVectorEntry<IdxType,ValType> e;
  e.idx = idx;
  e.val = val;

  pos = std::lower_bound(entries_.begin(), entries_.begin()+curr_size_, e, spveccomparebyidx());

  if( (*pos).idx == idx ){
    (*pos).val = val;
  }
  else{
    std::move_backward(pos, entries_.begin()+curr_size_, entries_.begin()+curr_size_+1);
    *pos = e;
    curr_size_++;
  }

return;
}


template <typename IdxType, typename ValType>
inline void SparseVector<IdxType, ValType>::set_value(const size_t idx, const ValType val){
  assert(idx<curr_size_);

  entries_[idx].val = val;
  return;
}


  // norm() returns the sum of the magnitudes of the current entry values.
template <typename IdxType, typename ValType>
inline const ValType SparseVector<IdxType, ValType>::norm() const {
  ValType norm = 0;
  for (size_t i = 0; i < curr_size_; i++) {
    norm += abs(entries_[i].val);
  }
  return norm;
}

  // sum() returns the sum of the current entry values.
template <typename IdxType, typename ValType>
inline const ValType SparseVector<IdxType, ValType>::sum() const {
  ValType sum = 0;
  for (size_t i = 0; i < curr_size_; i++) {
    sum += entries_[i].val;
  }
  return sum;
}

  // Constructor taking max_size as an argument and 
  // allocating space for that many SparseVectorEntry structs.
template <typename IdxType, typename ValType>
inline SparseVector<IdxType, ValType>::SparseVector(size_t max_size) : max_size_(max_size) {
  curr_size_ = 0;
  entries_ = std::vector<SparseVectorEntry<IdxType, ValType>>(max_size);
}

template <typename IdxType, typename ValType>
inline void SparseVector<IdxType, ValType>::clear() {
  curr_size_ = 0;
  return;
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

template <typename IdxType, typename ValType>
inline SparseVector<IdxType, ValType>& SparseVector<IdxType, ValType>::operator*=(const ValType alpha){
  for (size_t ii=0; ii< curr_size_; ii++ ){
    entries_[ii].val *= alpha;
  }
  return *this;
}

template <typename IdxType, typename ValType>
inline SparseVector<IdxType, ValType>& SparseVector<IdxType, ValType>::operator/=(const ValType alpha){
  assert(alpha!=0);
  for (size_t ii=0; ii< curr_size_; ii++ ){
    entries_[ii].val /= alpha;
  }
  return *this;
}

template <typename IdxType, typename ValType>
inline SparseVector<IdxType, ValType>& SparseVector<IdxType, ValType>::operator-=(const ValType alpha){
  for (size_t ii=0; ii< curr_size_; ii++ ){
    entries_[ii].val -= alpha;
  }
  return *this;
}

template <typename IdxType, typename ValType>
inline SparseVector<IdxType, ValType>& SparseVector<IdxType, ValType>::operator+=(const ValType alpha){
  for (size_t ii=0; ii< curr_size_; ii++ ){
    entries_[ii].val += alpha;
  }
  return *this;
}



// Normalize a sparse vector by the sum of its entry vals.
// template <typename IdxType, typename ValType>
// inline void SparseVector<IdxType, ValType>::normalize() {
//   ValType norm = norm();
//   this/=norm;
// }

// Remove all zero elements from a sparse vector.
template <typename IdxType, typename ValType>
inline void SparseVector<IdxType, ValType>::remove_zeros() {

  size_t jj = 0;
  while (jj<curr_size_){
    if (entries_[jj].val == 0){
      std::rotate(entries_.begin()+jj,entries_.begin()+jj+1,entries_.begin()+curr_size_);
      curr_size_--;
    }
    else{
      jj++;
    }
  }
  return;
}

// Print a vector to cout by printing the number 
// of nonzero entries and then printing each entry
// as a val idx pair. 
template <typename IdxType, typename ValType>
inline void SparseVector<IdxType, ValType>::print_vector() {
  std::cout << curr_size_ << std::endl;
  for (size_t jj = 0; jj < curr_size_; jj++) {
    std::cout << entries_[jj].idx <<"\t"<<entries_[jj].val << std::endl;
  }
  std::cout << std::endl;
}

// Perform  y <- α x + y.
// y must be large enough to contain 
// all entries of x and y; no allocation
// will be performed by this routine.
// template <typename IdxType, typename ValType>
// inline int sparse_axpy(ValType alpha,
//          const SparseVector<IdxType, ValType> &x,
//          SparseVector<IdxType, ValType> &y) {
//   // If alpha is not zero, then ensure the
//   // result vector y can store the entire sum.
//   if (alpha != 0) {
//     assert(x.curr_size_ + y.curr_size_ <= y.max_size_);
//   // If alpha is zero, this is a do-nothing operation.
//   // Short circuit it.
//   } else {
//     return 0;
//   }

//   // Shift the entries of y to past where they could possibly
//   // reside in the new vector.
//   for (size_t jj = y.curr_size_; jj > 0; jj--)
//       y[x.curr_size_ + jj - 1] = y[jj-1];
//   // Add in entries of either x or y with smallest indices. 
//   size_t y_entries_begin = x.curr_size_;
//   size_t y_entries_end = x.curr_size_ + y.curr_size_;
//   size_t n_result_entries = 0;
//   // Move over the entries in each vector from least
//   // index to highest, iterating on the x vector as
//   // the driver loop and y as a following iterator.
//   for (size_t jj = 0; jj < x.curr_size_; jj++){
//     // If there are y entries left and the y entry has lower
//     // or equal index, move it in.
//     while (y_entries_begin < y_entries_end && y[y_entries_begin].idx <= x[jj].idx){
//       y[n_result_entries] = y[y_entries_begin];
//       y_entries_begin++;
//       n_result_entries++;
//     }
//     // If the x and y entries had equal index, add the x 
//     // entry times alpha to the y entry.
//     if (n_result_entries > 0 && x[jj].idx == y[n_result_entries - 1].idx){
//       y[n_result_entries - 1].val += alpha * x[jj].val;
//     }
//     // Otherwise, just move the x entry times alpha in. 
//     else {
//       y[n_result_entries].idx = x[jj].idx;
//       y[n_result_entries].val = alpha * x[jj].val;
//       n_result_entries++;
//     }
//   }
//   // If all x entries are handled and y entries remain, move
//   // them all in.
//   while (y_entries_begin < y_entries_end){
//     y[n_result_entries] = y[y_entries_begin];
//     y_entries_begin++;
//     n_result_entries++;
//   }
//   // Now all entries are set; the new size is known.
//   y.curr_size_ = n_result_entries;
//   return 0;
// }



// Perform  y <- α x + y.
// y must be large enough to contain 
// all entries of x and y; no allocation
// will be performed by this routine.
// *** UNTESTED ****
// template <typename IdxType, typename ValType>
// inline int sparse_axpy(const ValType alpha,
//          const SparseVector<IdxType, ValType> &x,
//          SparseVector<IdxType, ValType> &y) {
//   // If alpha is not zero, then ensure the
//   // result vector y can store the entire sum.
//   if (alpha != 0) {
//     assert(x.curr_size_ + y.curr_size_ <= y.max_size_);
//   // If alpha is zero, this is a do-nothing operation.
//   // Short circuit it.
//   } else {
//     return 0;
//   }

//   // Shift the entries of y to past where they could possibly
//   // reside in the new vector.
//   for (size_t jj = y.curr_size_; jj > 0; jj--)
//       y[x.curr_size_ + jj - 1] = y[jj-1];
//   // Add in entries of either x or y with smallest indices. 
//   size_t y_entries_begin = x.curr_size_;
//   size_t y_entries_end = x.curr_size_ + y.curr_size_;
//   size_t n_result_entries = 0;
//   // Move over the entries in each vector from least
//   // index to highest, iterating on the x vector as
//   // the driver loop and y as a following iterator.
//   size_t jj = 0;
//   while (jj < x.curr_size_){
//     // If there are y entries left and the y entry has lower
//     // index, move it in.
//     while (y_entries_begin < y_entries_end and y[y_entries_begin].idx < x[jj].idx){
//       y[n_result_entries] = y[y_entries_begin];
//       y_entries_begin++;
//       n_result_entries++;
//       assert(1<0);
//     }
//     // If the x and y entries had equal index, add the x 
//     // entry times alpha to the y entry.
//     // std::cout<<jj<<std::endl;
//     // std::cout <<y_entries_begin<<"\t"<<y_entries_end<<std::endl;
//     // std::cout << x[jj].idx<<"\t"<<y[y_entries_begin].idx<<std::endl;
//     while ((jj<x.curr_size_) and (y_entries_begin < y_entries_end) and (x[jj].idx == y[y_entries_begin].idx)){
//       y[n_result_entries] = y[y_entries_begin];
//       // std::cout <<jj<<"\t"<<n_result_entries<<std::endl;
//       // std::cout<<y[n_result_entries].val<<std::endl;
//       y[n_result_entries].val += alpha * x[jj].val;
//       // std::cout<<alpha*x[jj].val<<std::endl;
//       // std::cout << std::endl;
//       jj++;
//       y_entries_begin++;
//       n_result_entries++;
//     }
//     // Otherwise, just move the x entry times alpha in. 
//     while (jj<x.curr_size_ && y[y_entries_begin].idx > x[jj].idx){
//       y[n_result_entries].idx = x[jj].idx;
//       y[n_result_entries].val = alpha * x[jj].val;
//       jj++;
//       n_result_entries++;
//     }
//   }
//   // If all x entries are handled and y entries remain, move
//   // them all in.
//   while (y_entries_begin < y_entries_end){
//     y[n_result_entries] = y[y_entries_begin];
//     y_entries_begin++;
//     n_result_entries++;
//   }
//   // Now all entries are set; the new size is known.
//   y.curr_size_ = n_result_entries;
//   return 0;
// }



template <typename IdxType, typename ValType>
inline SparseVector<IdxType, ValType>& SparseVector<IdxType, ValType>::operator+=(const SparseVector<IdxType, ValType>& x){

  assert(x.curr_size_ + curr_size_<= max_size_);

  std::vector<SparseVectorEntry<IdxType,ValType>> work(x.curr_size_+curr_size_);

  size_t y_entries_begin = 0;
  size_t y_entries_end = curr_size_;
  size_t n_result_entries = 0;
  // Move over the entries in each vector from least
  // index to highest, iterating on the x vector as
  // the driver loop and y as a following iterator.
  size_t jj = 0;
  while (jj < x.curr_size_){
    // If there are y entries left and the y entry has lower
    // index, move it in.
    while (y_entries_begin < y_entries_end and entries_[y_entries_begin].idx < x[jj].idx){
      work[n_result_entries] = entries_[y_entries_begin];
      y_entries_begin++;
      n_result_entries++;
      assert(1<0);
    }
    // If the x and y entries had equal index, add the x 
    // entry times alpha to the y entry.
    // std::cout<<jj<<std::endl;
    // std::cout <<y_entries_begin<<"\t"<<y_entries_end<<std::endl;
    // std::cout << x[jj].idx<<"\t"<<y[y_entries_begin].idx<<std::endl;
    while ((jj<x.curr_size_) and (y_entries_begin < y_entries_end) and (x[jj].idx == entries_[y_entries_begin].idx)){
      work[n_result_entries] = entries_[y_entries_begin];
      // std::cout <<jj<<"\t"<<n_result_entries<<std::endl;
      // std::cout<<y[n_result_entries].val<<std::endl;
      work[n_result_entries].val += x[jj].val;
      // std::cout<<alpha*x[jj].val<<std::endl;
      // std::cout << std::endl;
      jj++;
      y_entries_begin++;
      n_result_entries++;
    }
    // Otherwise, just move the x entry times alpha in. 
    while (jj<x.curr_size_ && entries_[y_entries_begin].idx > x[jj].idx){
      work[n_result_entries].val = x[jj].val;
      work[n_result_entries].idx = x[jj].idx;
      jj++;
      n_result_entries++;
    }
  }
  // If all x entries are handled and y entries remain, move
  // them all in.
  while (y_entries_begin < y_entries_end){
    work[n_result_entries].val = entries_[y_entries_begin].val;
    work[n_result_entries].idx = entries_[y_entries_begin].idx;
    y_entries_begin++;
    n_result_entries++;
  }

  for(jj=0;jj<n_result_entries;jj++){
    entries_[jj].val = work[jj].val;
    entries_[jj].idx = work[jj].idx;
  }
  // Now all entries are set; the new size is known.
  curr_size_ = n_result_entries;

  return *this;
}

template <typename IdxType, typename ValType>
inline SparseVector<IdxType, ValType>& SparseVector<IdxType, ValType>::operator-=(const SparseVector<IdxType, ValType>& x){
  assert(x.curr_size_ + curr_size_<= max_size_);

  std::vector<SparseVectorEntry<IdxType, ValType>> work(x.curr_size_+curr_size_);

  size_t y_entries_begin = 0;
  size_t y_entries_end = curr_size_;
  size_t n_result_entries = 0;
  // Move over the entries in each vector from least
  // index to highest, iterating on the x vector as
  // the driver loop and y as a following iterator.
  size_t jj = 0;
  while (jj < x.curr_size_){
    // If there are y entries left and the y entry has lower
    // index, move it in.
    while (y_entries_begin < y_entries_end and entries_[y_entries_begin].idx < x[jj].idx){
      work[n_result_entries] = entries_[y_entries_begin];
      y_entries_begin++;
      n_result_entries++;
      assert(1<0);
    }
    // If the x and y entries had equal index, add the x 
    // entry times alpha to the y entry.
    // std::cout<<jj<<std::endl;
    // std::cout <<y_entries_begin<<"\t"<<y_entries_end<<std::endl;
    // std::cout << x[jj].idx<<"\t"<<y[y_entries_begin].idx<<std::endl;
    while ((jj<x.curr_size_) and (y_entries_begin < y_entries_end) and (x[jj].idx == entries_[y_entries_begin].idx)){
      work[n_result_entries] = entries_[y_entries_begin];
      // std::cout <<jj<<"\t"<<n_result_entries<<std::endl;
      // std::cout<<y[n_result_entries].val<<std::endl;
      work[n_result_entries].val -= x[jj].val;
      // std::cout<<alpha*x[jj].val<<std::endl;
      // std::cout << std::endl;
      jj++;
      y_entries_begin++;
      n_result_entries++;
    }
    // Otherwise, just move the x entry times alpha in. 
    while (jj<x.curr_size_ && entries_[y_entries_begin].idx > x[jj].idx){
      work[n_result_entries].val = x[jj].val;
      work[n_result_entries].idx = x[jj].idx;
      jj++;
      n_result_entries++;
    }
  }
  // If all x entries are handled and y entries remain, move
  // them all in.
  while (y_entries_begin < y_entries_end){
    work[n_result_entries].val = entries_[y_entries_begin].val;
    work[n_result_entries].idx = entries_[y_entries_begin].idx;
    y_entries_begin++;
    n_result_entries++;
  }

  for(jj=0;jj<n_result_entries;jj++){
    entries_[jj].val = work[jj].val;
    entries_[jj].idx = work[jj].idx;
  }
  // Now all entries are set; the new size is known.
  curr_size_ = n_result_entries;

  return *this;
}







//---------------------------------------------------------
// Sparse matrix class definitions and routines
// Declaration
//---------------------------------------------------------

// SparseMatrix represents a sparse matrix implemented
// as an array of SparseMatrixEntry values. The underlying
// array is fixed size to avoid costly allocations, 
// this is max_size_, but the size of the actual matrix 
// is dynamic and equal to curr_size_. There are no 
// guarantees on SparseMatrixEntry values past curr_size_.
// They can be  subscripted as if they were a vector of
// SparseMatrixEntry structs and also provide begin() and end()
// based on curr_size_.
template <typename IdxType, typename ValType>
class SparseMatrix {
public:
  
  // Number of nonzero entries that could be in the vector.
  // Must not change.
  const size_t max_size_;

  // Max number of rows and columns.
  const size_t max_cols_, max_rowcol_nnz_;

  // Constructor taking max_rows, max_cols, and max_size as an argument and 
  // allocating space for that many SparseMatrixEntry structs.
  SparseMatrix(size_t max_cols, size_t max_rowcol_nnz);

  // Assignment by value up to current size, leaving
  // max size & other entries unchanged.
  inline SparseMatrix<IdxType, ValType>& operator=(const SparseMatrix<IdxType, ValType> &other);

  // Index reordering to Compressed Row Storage (CRS)
  inline void sort_crs();
  inline const bool check_crs_sorted() const {return is_crs_sorted_;}

  // Index reordering to Compressed Column Storage (CCS)
  inline void sort_ccs();
  inline const bool check_ccs_sorted() const {return is_ccs_sorted_;}

  inline const size_t size() const {return curr_size_;}

  inline const size_t ncols() const {return n_cols_;}

  inline const size_t nrows() const {return n_rows_;} 

  // Reset matrix without deallocating space.
  inline void clear();

  // Write column idx into a sparse vector.
  inline void get_col(const size_t col_num, SparseVector<IdxType,ValType>& other);

  // Write row idx into a sparse vector.
  inline void get_row(const size_t row_num, SparseVector<IdxType,ValType>& other);

  // Add a new column to the matrix.  If there's already a column with index idx
  // it is replaced.
  inline void set_col(const SparseVector<IdxType,ValType>& other, const IdxType idx);

  // Add a new row to the matrix.  If there's already a row with index idx
  // it is replaced.
  //inline void set_row(const SparseVector<IdxType,ValType>& other, const IdxType idx);

  // Remove a column from the matrix.
  inline void eject_col(const size_t col_num);

  // Compute column sums and output in sparse vector.
  inline void col_sums(SparseVector<IdxType,ValType>& other);

  // Compute column norms and output in sparse vector.
  inline void col_norms(SparseVector<IdxType,ValType>& other, size_t p);

  // Compute row sums and output in sparse vector.
  inline void row_sums(SparseVector<IdxType,ValType>& other);

  // Compute row norms and output in sparse vector.
  inline void row_norms(SparseVector<IdxType,ValType>& other, size_t p);

  inline size_t locate_col(const IdxType idx);

  inline size_t locate_row(const IdxType idx);

  // Get entry in CCS order
  // inline const SparseMatrixEntry<IdxType, ValType>& get_ccs_entry(const size_t idx) const {return entries_[idx];}

  // Get entry in CCR order
  //inline const SparseMatrixEntry<IdxType, ValType>& get_crs_entry(const size_t idx) const {return entries_[idx];}

  // Print a matrix to cout by printing the number 
  // of nonzero entries and then printing each entry
  // as a val idx pair. 
  inline void print_ccs();
  inline void print_crs();

  // Accessors for the underlying vector so one can do
  // some of the normal vector manipulation, but not all.
  // Specifically, subscripting and iterator begin and end
  // are provided. Incremental queue and stack function is 
  // intentionally not provided.
  //inline size_t ccs_lower_bound(const IdxType rowidx, const IdxType colidx);
  //inline void set_entry(const SparseMatrixEntry<IdxType, ValType>& a, const size_t idx);
  inline const SparseMatrixEntry<IdxType,ValType>& operator[](const size_t idx) const {return entries_[idx];}
  //inline const SparseMatrixEntry<IdxType, ValType>& get_entry(const size_t idx) const {return entries_[idx];}
  typedef typename std::vector<SparseMatrixEntry<IdxType, ValType>>::iterator spmat_iterator;
  inline spmat_iterator begin() {return entries_.begin();}
  inline spmat_iterator end() {return entries_.begin() + curr_size_;}

private:
  // Number of nonzero entries actually in the vector.
  size_t curr_size_;
  // Number of rows and columns.
  size_t n_rows_, n_cols_;
  // Do not allow manual pushing/popping of the entries 
  // or anything else that is not ordering safe.
  std::vector<SparseMatrixEntry<IdxType, ValType>> entries_;
  std::vector<size_t> inv_ccs_order_;   // lists the column-first rank of the indices
  std::vector<size_t> inv_crs_order_;  // lists the row-first rank of the indices
  std::vector<size_t> ccs_order_;  // lists the indices in column-first sorted order
  std::vector<size_t> crs_order_;  // lists the indices in row-first sorted order
  std::vector<size_t> row_ends_;   // the position (in row-first order) of the end of each row
  std::vector<size_t> col_ends_;    // the position (in column-first order) of the end of each column
  inline void inject_col(const size_t col_num);
  inline void clear_col(const size_t col_num);
  inline void fill_col(const size_t col_num, const IdxType idx, const SparseVector<IdxType,ValType>& other);
  inline size_t locate_entry_ccs( const IdxType rowidx, const IdxType colidx );
  inline size_t locate_entry_crs( const IdxType rowidx, const IdxType colidx );
  inline void swap_entries(const size_t ii, const size_t jj);
  bool is_crs_sorted_;
  bool is_ccs_sorted_;
};




// spcolwisemv multiplies each column of 
// the sparse matrix A by the corresponding
// entry of the sparse vector x.
// A is specified by a routine Acolumn that returns 
// a single column of A.  max_nz_col_entries is an
// upper bound on the number of non-zero entries in
// any column of A.  The resulting matrix B 
// is assumed to be of max_size
// at least max_nz_col_entries * x.curr_size_.
// B is overwritten upon output.
template <typename IdxType, typename ValType, typename ParType>
inline void sparse_colwisemv(int (*Acolumn)(SparseVector<IdxType, ValType>&, const IdxType, const ParType),
        const ParType params, const size_t max_nz_col_entries, const SparseVector<IdxType, ValType> &x, 
        SparseMatrix<IdxType,ValType>& B);



//---------------------------------------------------------
// Sparse matrix entry definition and helper routines
// Implementation
//---------------------------------------------------------

// Compare two sparse matrix entries by value.
// Relies on ValType having a less than comparator.
struct spmatcomparebyval {
  template <typename IdxType, typename ValType>
  inline bool operator () (const SparseMatrixEntry<IdxType, ValType> &a, const SparseMatrixEntry<IdxType, ValType> &b) {
    return a.val < b.val;
  }
};

// Compare two sparse matrix entries by lexicographic 
// ordering of indices with row index first.
// Relies on IdxType having a less than comparator.
// struct spmatcomparebyrowidxfirst {
//   template <typename IdxType, typename ValType>
//   inline bool operator () (const SparseMatrixEntry<IdxType, ValType> &a, const SparseMatrixEntry<IdxType, ValType> &b) {
//     if (a.rowidx<b.rowidx) return true;
//     else if (a.rowidx==b.rowidx & a.colidx<b.colidx) return true;
//     else return false;
//   }
// };


template <typename IdxType, typename ValType>
inline bool spmatcomparebyrowidxfirst(const SparseMatrixEntry<IdxType, ValType> &a, const SparseMatrixEntry<IdxType, ValType> &b) {
  if (a.rowidx<b.rowidx) return true;
  else if (a.rowidx==b.rowidx & a.colidx<b.colidx) return true;
  else return false;
}

// Compare two sparse matrix entries by lexicographic 
// ordering of indices with column index first.
// struct spmatcomparebycolidxfirst {
//   template <typename IdxType, typename ValType>
//   inline bool operator () (const SparseMatrixEntry<IdxType, ValType> &a, const SparseMatrixEntry<IdxType, ValType> &b) {
//     if (a.colidx<b.colidx) return true;
//     else if (a.colidx==b.colidx & a.rowidx<b.rowidx) return true;
//     else return false;
//   }
// };

template <typename IdxType, typename ValType>
inline bool spmatcomparebycolidxfirst(const SparseMatrixEntry<IdxType, ValType> &a, const SparseMatrixEntry<IdxType, ValType> &b) {
  if (a.colidx<b.colidx) return true;
  else if (a.colidx==b.colidx & a.rowidx<b.rowidx) return true;
  else return false;
}



//---------------------------------------------------------
// Sparse matrix class definition and routines
// Implementation
//---------------------------------------------------------

// Constructor taking max_rows, max_cols, and max_size as an argument and 
// allocating space for that many SparseMatrixEntry structs.
template <typename IdxType, typename ValType>
inline SparseMatrix<IdxType, ValType>::SparseMatrix(size_t max_cols, size_t max_rowcol_nnz)
  : max_size_(max_cols*max_rowcol_nnz), max_cols_(max_cols), max_rowcol_nnz_(max_rowcol_nnz) {
  curr_size_ = 0;
  n_rows_ = 0;
  n_cols_ = 0;
  entries_ = std::vector<SparseMatrixEntry<IdxType, ValType>>(max_cols*max_rowcol_nnz);
  inv_ccs_order_ = std::vector<size_t>(max_cols*max_rowcol_nnz);
  inv_crs_order_ = std::vector<size_t>(max_cols*max_rowcol_nnz);
  crs_order_ = std::vector<size_t>(max_cols*max_rowcol_nnz);
  ccs_order_ = std::vector<size_t>(max_cols*max_rowcol_nnz);
  row_ends_ = std::vector<size_t>(max_cols*max_rowcol_nnz);
  col_ends_ = std::vector<size_t>(max_cols);

  std::iota(row_ends_.begin(),row_ends_.end(),0);
  std::for_each(row_ends_.begin(), row_ends_.end(), [&](size_t& d) { d*=max_rowcol_nnz;});

  std::iota(col_ends_.begin(),col_ends_.end(),0);
  std::for_each(col_ends_.begin(), col_ends_.end(), [&](size_t& d) { d*=max_rowcol_nnz;});

  is_crs_sorted_ = true;
  is_ccs_sorted_ = true;
}


// Reset sparse matrix without deallocating space.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::clear(){
  curr_size_ = 0;
  n_rows_ = 0;
  n_cols_ = 0;
  std::iota(row_ends_.begin(),row_ends_.end(),0);
  std::for_each(row_ends_.begin(), row_ends_.end(), [&](size_t& d) { d*=max_rowcol_nnz_;});
  std::iota(col_ends_.begin(),col_ends_.end(),0);
  std::for_each(col_ends_.begin(), col_ends_.end(), [&](size_t& d) { d*=max_rowcol_nnz_;});
  is_crs_sorted_ = true;
  is_ccs_sorted_ = true;
}





// Add in a new empty column.  Does not check for whether the column is new or whether it's being
// placed in the correct position (that's why it's private).
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::inject_col(const size_t col_num){
  assert(col_num<=n_cols_);

  size_t ii, jj, col_head = col_num*max_rowcol_nnz_;

  // std::cout<<max_rowcol_nnz_<<std::endl;
  // std::cout<<n_cols_<<std::endl;
  // std::cout<<col_ends_[0]<<std::endl;

  if (n_cols_>0){
    for( jj=n_cols_; jj>col_num; jj--){      // shift columns with larger index backward to create space for new column
      //std::cout << jj-1 << std::endl;
      for( ii=(jj-1)*max_rowcol_nnz_; ii<col_ends_[jj]; ii++){
        //std::cout << ii <<std::endl;
        inv_ccs_order_[ccs_order_[ii]] += max_rowcol_nnz_;
      }
      std::move_backward(ccs_order_.begin()+(jj-1)*max_rowcol_nnz_,
        ccs_order_.begin()+col_ends_[jj-1],ccs_order_.begin()+col_ends_[jj-1]+max_rowcol_nnz_);
      col_ends_[jj] = col_ends_[jj-1]+max_rowcol_nnz_;
    }
  }
  col_ends_[col_num] = col_head;

  // for(ii=0;ii<n_cols_+1;ii++)
  //   std::cout << col_ends_[ii]<<std::endl;
  // std::cout<<std::endl;

  n_cols_++;

  is_crs_sorted_ = false;

  return;
}

// Removes a column from the matrix.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::eject_col(const size_t col_num){
  assert(col_num<n_cols_);

  clear_col(col_num);                        // make sure the column is cleared first;

  size_t ii, jj, col_head = col_num*max_rowcol_nnz_;

  // std::cout<<max_rowcol_nnz_<<std::endl;
  // std::cout<<n_cols_<<std::endl;
  // std::cout<<col_ends_[0]<<std::endl;

  for( jj=col_num+1; jj<n_cols_; jj++){      // shift columns with larger index forward to create space for new column
      //std::cout << jj-1 << std::endl;
    for( ii=(jj-1)*max_rowcol_nnz_; ii<col_ends_[jj]; ii++){
      //std::cout << ii <<std::endl;
      inv_ccs_order_[ccs_order_[ii]] -= max_rowcol_nnz_;
    }
    std::swap_ranges(ccs_order_.begin()+jj*max_rowcol_nnz_,
      ccs_order_.begin()+col_ends_[jj],ccs_order_.begin()+(jj-1)*max_rowcol_nnz_);
    col_ends_[jj-1] = col_ends_[jj]-max_rowcol_nnz_;
  }

  // for(ii=0;ii<n_cols_+1;ii++)
  //   std::cout << col_ends_[ii]<<std::endl;
  // std::cout<<std::endl;

  n_cols_--;

  is_crs_sorted_ = false;

  return;
}





// Assumes column col_num is empty and fills it with entries from the sparse vector
// other.  Does not check that the column index idx is unique or that it's being
// placed in the right position (that's why it's private).
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::fill_col(const size_t col_num, const IdxType idx, const SparseVector<IdxType, ValType>& other){
  assert(col_num<n_cols_);
  assert(col_num*max_rowcol_nnz_ == col_ends_[col_num]);

  size_t col_head = col_num*max_rowcol_nnz_;;

  for (size_t jj=0; jj<other.size(); jj++){
    entries_[curr_size_+jj].colidx = idx;
    entries_[curr_size_+jj].rowidx = other[jj].idx;
    entries_[curr_size_+jj].val = other[jj].val;
    // entries_[curr_size_+jj].rowidx = other[jj].idx;
    // entries_[curr_size_+jj].val = other[jj].val;
  }

  col_ends_[col_num] = col_head+other.size();

  std::iota(ccs_order_.begin()+col_head,ccs_order_.begin()+col_ends_[col_num],curr_size_);
  std::iota(inv_ccs_order_.begin()+curr_size_,inv_ccs_order_.begin()+curr_size_+other.size(),col_head);

  // std::cout<<"in fill"<<std::endl;
  // std::cout << col_num <<"\t" <<col_head << std::endl;
  // for(size_t jj=0;jj<n_cols_;jj++){
  //   std::cout<<jj<<"\t"<<jj*max_rowcol_nnz_<<"\t"<<col_ends_[jj] <<std::endl;
  // }
  // std::cout<<std::endl; 

  // for(size_t jj=0;jj<col_ends_[0];jj++){
  //   std::cout << jj <<"\t"<< ccs_order_[jj] << "\t" <<entries_[ccs_order_[jj]].colidx << std::endl;
  // }
  // std::cout<<"out fill"<< std::endl;

  curr_size_ += other.size();

  is_crs_sorted_ = false;

  return;
}



// Clear the contents of a column without removing the column.  Does not preserve crs ordering.  Private
// because I can't think of a reason to use this without also using fill_coll (which is private).
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::swap_entries(const size_t ii, const size_t jj){
  assert(ii<curr_size_);
  assert(jj<curr_size_);

  size_t tmp;

  std::iter_swap(entries_.begin()+ii,entries_.begin()+jj);

  if( is_ccs_sorted_ ){
    ccs_order_[inv_ccs_order_[jj]] = ii;
    ccs_order_[inv_ccs_order_[ii]] = jj;
    tmp = inv_ccs_order_[ii];
    inv_ccs_order_[ii] = inv_ccs_order_[jj];
    inv_ccs_order_[jj] = tmp;
  }
 
  if( is_crs_sorted_ ){
    crs_order_[inv_crs_order_[jj]] = ii;
    crs_order_[inv_crs_order_[ii]] = jj;
    tmp = inv_crs_order_[ii];
    inv_crs_order_[ii] = inv_crs_order_[jj];
    inv_crs_order_[jj] = tmp;
  }

  return;
}



// Clear the contents of a column without removing the column.  Does not preserve crs ordering.  Private
// because I can't think of a reason to use this without also using fill_coll (which is private).
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::clear_col(const size_t col_num){
  assert(col_num<n_cols_);

  size_t col_len, col_head, new_loc, ccs_loc;
  std::vector<size_t>::iterator ccs_pos;

  col_head = col_num*max_rowcol_nnz_;
  col_len = col_ends_[col_num] - col_head;
  ccs_pos = ccs_order_.begin() + col_head;

  if( col_len==0 )
    return;

  // std::cout<<"in clear"<<std::endl;
  // for(size_t jj=0;jj<curr_size_;jj++){
  //   std::cout << jj <<"\t"<< entries_[jj].rowidx << "\t"<<entries_[jj].colidx << "\t"<<inv_ccs_order_[jj] << std::endl;
  // }
  // std::cout<<std::endl;
  
  // for(size_t jj=0;jj<col_ends_[0];jj++){
  //   std::cout << jj <<"\t"<< ccs_order_[jj] << "\t" <<entries_[ccs_order_[jj]].rowidx
  //     << "\t" << entries_[ccs_order_[jj]].colidx << std::endl;
  // }
  // std::cout<<std::endl;
  // std::cout << col_head << "\t" << col_ends_[col_num] <<std::endl;
  // std::cout<<std::endl;

  assert(col_len>=0);

  for (size_t jj=0; jj < col_len; jj++ ){      // swap iterators from end to fill unfilled old entries
    assert(curr_size_>jj);
    if ( *(ccs_pos+jj)!=curr_size_-jj-1 ){
      // std::cout <<*(ccs_pos+jj)<<"\t"<<curr_size_-jj-1<<std::endl;
      swap_entries(*(ccs_pos+jj),curr_size_-jj-1);
    }
  }
  // std::cout<<std::endl;

  col_ends_[col_num] = col_head;

  
  // for(size_t jj=0;jj<n_cols_;jj++){
  //   std::cout<<jj<<"\t"<<jj*max_rowcol_nnz_<<"\t"<<col_ends_[jj] <<std::endl;
  // }
  // std::cout<<std::endl; 
  // for(size_t jj=0;jj<col_ends_[0];jj++){
  //   std::cout << jj <<"\t"<< ccs_order_[jj] << "\t" <<entries_[ccs_order_[jj]].rowidx
  //     << "\t" << entries_[ccs_order_[jj]].colidx << std::endl;
  // }
  // std::cout<<"out clear"<< std::endl;

  curr_size_ -= col_len;

  is_crs_sorted_ = false;

  return;
}


// locate_col returns the column number of the column with index 
// idx.  If no column has index idx the first column whose
// index compares greater than idx is returned.  If all columns
// have indices that compare less than idx then n_cols_ is returned.  
template <typename IdxType, typename ValType>
inline size_t SparseMatrix<IdxType, ValType>::locate_col(const IdxType idx){

  if (!is_ccs_sorted_)
    sort_ccs();

  std::vector<size_t>::iterator col_id;

  entries_[curr_size_].colidx = idx;
  ccs_order_[n_cols_*max_rowcol_nnz_]=curr_size_;
  col_id = std::lower_bound(col_ends_.begin(), col_ends_.begin()+n_cols_, n_cols_*max_rowcol_nnz_+1,
    [&](size_t ii, size_t jj) { return entries_[ccs_order_[ii-1]].colidx < entries_[ccs_order_[jj-1]].colidx; });

  if( col_id==col_ends_.begin()+n_cols_ ){
    return n_cols_;
  }
 
  return (*col_id-1) / max_rowcol_nnz_;
}


// locate_row returns the row number of the row with index 
// idx.  If no row has index idx the first row whose
// index compares greater than idx is returned.  If all rows
// have indices that compare less than idx then n_rows_ is returned.  
template <typename IdxType, typename ValType>
inline size_t SparseMatrix<IdxType, ValType>::locate_row(const IdxType idx){

  if (!is_crs_sorted_)
    sort_crs();

  std::vector<size_t>::iterator row_id;

  entries_[curr_size_].rowidx = idx;
  crs_order_[n_rows_*max_rowcol_nnz_]=curr_size_;
  row_id = std::lower_bound(row_ends_.begin(), row_ends_.begin()+n_rows_, n_rows_*max_rowcol_nnz_+1,
    [&](size_t ii, size_t jj) { return entries_[crs_order_[ii-1]].rowidx < entries_[crs_order_[jj-1]].rowidx; });

  if( row_id==row_ends_.begin()+n_rows_ ){
    return n_rows_;
  }
 
  return (*row_id-1) / max_rowcol_nnz_;
}


// locate the position in the ccs_order_ of entry with rowidx and colidx.  If no 
// such entry is found return first position with entry that compares greater
// than (rowidx,colidx) in CCS order.  Throws an error if the entry is new and
// would belong to a column that already has max_rowcol_nnz_.
template <typename IdxType, typename ValType>
inline size_t SparseMatrix<IdxType, ValType>::locate_entry_ccs(const IdxType rowidx, const IdxType colidx){

  if (!is_ccs_sorted_)
    sort_ccs();

  size_t col_head, col_num = locate_col(colidx);
  std::vector<size_t>::iterator ccs_pos;

  if( col_num == n_cols_ )
    return n_cols_*max_rowcol_nnz_;

  col_head = col_num*max_rowcol_nnz_;

  if ( colidx == entries_[ccs_order_[col_head]].colidx ){
    entries_[curr_size_].rowidx = rowidx;
    ccs_pos = std::lower_bound(ccs_order_.begin()+col_head, ccs_order_.begin()+col_ends_[col_num], curr_size_,
      [&](size_t ii, size_t jj) { return entries_[ii].rowidx < entries_[jj].rowidx; });
    if( ccs_pos == ccs_order_.begin()+col_ends_[col_num] ){
      assert( *ccs_pos-col_head < max_rowcol_nnz_);
    }
    return *ccs_pos;
  }
  else{
    return col_head;
  }
}


// locate the position in the crs_order_ of entry with rowidx and colidx.  If no 
// such entry is found return first position with entry that compares greater
// than (rowidx,colidx) in CRS order.  Throws an error if the entry is new and
// would belong to a row that already has max_rowcol_nnz_.
template <typename IdxType, typename ValType>
inline size_t SparseMatrix<IdxType, ValType>::locate_entry_crs(const IdxType rowidx, const IdxType colidx){

  if (!is_crs_sorted_)
    sort_crs();

  size_t row_head, row_num = locate_row(colidx);
  std::vector<size_t>::iterator crs_pos;

  if( row_num == n_rows_ )
    return n_rows_*max_rowcol_nnz_;

  row_head = row_num*max_rowcol_nnz_;

  if ( rowidx == entries_[crs_order_[row_head]].rowidx ){
    entries_[curr_size_].colidx = colidx;
    crs_pos = std::lower_bound(crs_order_.begin()+row_head, crs_order_.begin()+row_ends_[row_num], curr_size_,
      [&](size_t ii, size_t jj) { return entries_[ii].colidx < entries_[jj].colidx; });
    if( crs_pos == crs_order_.begin()+row_ends_[row_num] ){
      assert( *crs_pos-row_head < max_rowcol_nnz_);
    }
    return *crs_pos;
  }
  else{
    return row_head;
  }
}



// Add a sparse vector as a column to a sparse matrix.
// If the matrix already has that column it is replaced.
// This leaves the matrix CCS ordered, but destroys CRS ordering.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::set_col(const SparseVector<IdxType, ValType>& other, const IdxType idx){

  assert( other.size()<=max_rowcol_nnz_);
  assert( curr_size_+other.size()<=max_size_);
  assert( n_cols_<max_cols_);
  assert( n_rows_+other.size()<=max_size_);

  size_t col_num;

  if (!is_ccs_sorted_)
    sort_ccs();

  if (curr_size_==0){                    // is the matrix empty?

    col_num = 0;

    inject_col(col_num);
    fill_col(col_num,idx,other);

    for(size_t jj=0; jj<other.size(); jj++ ){          // because it's the first column we preserve row ordering too
      crs_order_[row_ends_[jj]] = jj;
      inv_crs_order_[jj] = row_ends_[jj];
      row_ends_[jj]++;
    }
    n_rows_ = other.size();
    is_crs_sorted_ = true;

    return;
  }

  is_crs_sorted_ = false;

  size_t col_head;

  if ( idx > entries_[ccs_order_[(n_cols_-1)*max_rowcol_nnz_]].colidx ){                // will this be the largest column index
    col_num = n_cols_;
    inject_col(col_num);  
  }
  else{                                                              // this won't be the largest column index

    col_num = locate_col(idx);
    assert(col_num<n_cols_);

    col_head = col_num*max_rowcol_nnz_;
    assert(col_head<col_ends_[col_num]);

    if (entries_[ccs_order_[col_head]].colidx == idx){                 // this replaces an existing column
      clear_col(col_num);
    }
    else{                                                                     // this column is new
      inject_col(col_num);
    }

    // std::cout<<"out"<<std::endl;
    // std::cout<<std::endl;
  }
  fill_col(col_num,idx,other);

  return;
}


// Add a sparse vector as a row to a sparse matrix.
// If the matrix already has that column it is replaced.
// This leaves the matrix CRS ordered, but destroys CCS ordering.


// Find the indexing that reorders the columns into CRS order.
// On output is_crs_ordered will be set to true and that
// value will be returned.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::sort_crs(){

  if (is_crs_sorted_==true) return;
  if (curr_size_==0){
    is_crs_sorted_=true;
    return;
  }

  std::iota(crs_order_.begin(),crs_order_.begin()+curr_size_,0);
  std::sort(crs_order_.begin(), crs_order_.begin()+curr_size_, 
    [&](size_t ii, size_t jj) { return spmatcomparebyrowidxfirst(entries_[ii],entries_[jj]); });

  n_rows_ = 1;
  for (size_t jj=1; jj<curr_size_;jj++){
    if ( entries_[crs_order_[jj]].rowidx > entries_[crs_order_[jj-1]].rowidx ){
      row_ends_[n_rows_-1] = jj;
      n_rows_++; 
    }
  }
  row_ends_[n_rows_-1] = curr_size_;

  size_t row_len;

  if( n_rows_>0 ){
    for ( size_t jj=n_rows_-1;jj>0;jj--){
      row_len = row_ends_[jj]-row_ends_[jj-1];
      std::move_backward(crs_order_.begin()+row_ends_[jj-1],crs_order_.begin()+row_ends_[jj],crs_order_.begin()+jj*max_rowcol_nnz_+row_len);
      row_ends_[jj] = jj*max_rowcol_nnz_ + row_len;
      //std::cout<<jj<<"\t"<<row_ends_[jj]<<std::endl;
    }
  }

  for (size_t jj=0;jj<n_rows_;jj++){
    for (size_t ii=jj*max_rowcol_nnz_;ii<row_ends_[jj];ii++){
      inv_crs_order_[crs_order_[ii]] = ii;
    }
  }

  is_crs_sorted_=true;
  is_ccs_sorted_=false;

  return;
}

// Find the indexing that reorders the columns into CCS order.
// On output is_ccs_ordered will be set to true and that
// value will be returned.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::sort_ccs(){

  if (is_ccs_sorted_==true) return;
  if (curr_size_==0){
    is_ccs_sorted_=true;
    return;
  }

  std::iota(ccs_order_.begin(),ccs_order_.begin()+curr_size_,0);
  std::sort(ccs_order_.begin(), ccs_order_.begin()+curr_size_, 
    [&](size_t ii, size_t jj) { return spmatcomparebycolidxfirst(entries_[ii],entries_[jj]); });

  n_cols_ = 1;
  for (size_t jj=1; jj<curr_size_;jj++){
    if ( entries_[ccs_order_[jj]].colidx > entries_[ccs_order_[jj-1]].colidx ){
      col_ends_[n_cols_-1] = jj;
      n_cols_++; 
    }
  }
  col_ends_[n_cols_-1] = curr_size_;

  size_t col_len;

  if( n_cols_>0 ){
    for ( size_t jj=n_cols_-1;jj>0;jj--){
      col_len = col_ends_[jj]-col_ends_[jj-1];
      std::move_backward(ccs_order_.begin()+col_ends_[jj-1],ccs_order_.begin()+col_ends_[jj],ccs_order_.begin()+jj*max_rowcol_nnz_+col_len);
      col_ends_[jj] = jj*max_rowcol_nnz_ + col_len;
    }
  }

  for (size_t jj=0;jj<n_cols_;jj++){
    for (size_t ii=jj*max_rowcol_nnz_;ii<col_ends_[jj];ii++){
      inv_ccs_order_[ccs_order_[ii]] = ii;
    }
  }

  is_ccs_sorted_=true;
  is_crs_sorted_=false;

  return;
}

// Copy entries of column idx of a sparse matrix
// into a sparse vector.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::get_col(const size_t col_num, SparseVector<IdxType, ValType>& y){
  assert(col_num<n_cols_);

  if(!is_ccs_sorted_)
    sort_ccs();

  size_t col_head = col_num*max_rowcol_nnz_;

  assert( y.max_size_ >= col_ends_[col_num] - col_head );
  y.curr_size_ = col_ends_[col_num] - col_head;

  for (size_t jj=col_head; jj<col_ends_[col_num]; jj++){
    y[jj-col_head].idx = entries_[ccs_order_[jj]].rowidx;
    y[jj-col_head].val = entries_[ccs_order_[jj]].val;
  }

  return;
}

// Copy entries of row idx of a sparse matrix
// into a sparse vector.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::get_row(const size_t row_num, SparseVector<IdxType, ValType>& y){
  assert(row_num<n_rows_);

  if(!is_crs_sorted_)
    sort_crs();

  size_t row_head = row_num*max_rowcol_nnz_;

  assert( y.max_size_ >= row_ends_[row_num] - row_head );
  y.curr_size_ = row_ends_[row_num] - row_head;

  for (size_t jj=row_head; jj<row_ends_[row_num]; jj++){
    y[jj-row_head].idx = entries_[crs_order_[jj]].colidx;
    y[jj-row_head].val = entries_[crs_order_[jj]].val;
  }

  return;
}


template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::row_sums(SparseVector<IdxType, ValType>& y){
  assert(y.max_size_>=n_rows_);

  if(!is_crs_sorted_)
    sort_crs();

  SparseVectorEntry<IdxType,ValType> e;

  y.clear();

  for( size_t ii = 0; ii<n_rows_; ii++){
    e.val = entries_[crs_order_[ii*max_rowcol_nnz_]].val;
    e.idx = entries_[crs_order_[ii*max_rowcol_nnz_]].rowidx;
    y.set_entry(e);
    for( size_t jj = ii*max_rowcol_nnz_+1; jj<row_ends_[ii]; jj++){
      y.set_value(ii,y[ii].val+entries_[crs_order_[jj]].val);
    }
  }

  return;
}


template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::col_sums(SparseVector<IdxType, ValType>& y){
  assert(y.max_size_>=n_cols_);

  if(!is_ccs_sorted_)
    sort_ccs();

  SparseVectorEntry<IdxType,ValType> e;

  y.clear();

  for( size_t jj = 0; jj<n_cols_; jj++){
    e.val = entries_[ccs_order_[jj*max_rowcol_nnz_]].val;
    e.idx = entries_[ccs_order_[jj*max_rowcol_nnz_]].colidx;
    y.set_entry(e);
    for( size_t ii = jj*max_rowcol_nnz_+1; ii<col_ends_[jj]; ii++){
      y[jj] += entries_[ccs_order_[ii]].val;
    }
  }

  return;
}



// Quary the entries in CRS order.  The CRS order
// must have already been set by a 
// call to set_crs_order()
// template <typename IdxType, typename ValType>
// inline const SparseMatrixEntry<IdxType, ValType>& SparseMatrix<IdxType,ValType>::get_crs_entry(const size_t ii) {
//   sort_crs();
//   return entries_[crs_order_[ii]];
// }

// Quary the entries in CCS order.  The CCS order
// must have already been set by a 
// call to set_ccs_order()
// template <typename IdxType, typename ValType>
// inline const SparseMatrixEntry<IdxType, ValType>& SparseMatrix<IdxType,ValType>::ccs(const size_t idx) {
//   assert(is_ccs_ordered_);
//   return *ccs_order_[idx];
// }



// Assignment by value up to current size, leaving
// max size & other entries unchanged.
template <typename IdxType, typename ValType>
inline SparseMatrix<IdxType, ValType>& SparseMatrix<IdxType, ValType>::operator=(const SparseMatrix<IdxType, ValType> &other) {
  assert(other.curr_size_ <= max_size_);
  curr_size_ = other.curr_size_;
  n_cols_ = other.n_cols_;
  n_rows_ = other.n_rows_;
  for (size_t i = 0; i < other.curr_size_; i++) {
    entries_[i] = other.entries_[i];
    crs_order_[i] = other.crs_order_[i];
    ccs_order_[i] = other.ccs_order_[i];
    inv_crs_order_[i] = other.inv_crs_order_[i];
    inv_ccs_order_[i] = other.inv_ccs_order_[i];
  }
  for (size_t jj = 0; jj<other.n_rows_; jj++){
    row_ends_[jj] = other.row_ends_[jj];
  }
  for (size_t jj = 0; jj<other.n_cols_; jj++){
    col_ends_[jj] = other.col_ends_[jj];
  }

  is_crs_sorted_ = other.is_crs_sorted_;
  is_ccs_sorted_ = other.is_ccs_sorted_;

  return *this;
}

// Print a matrix to cout by printing the number 
// of nonzero entries and then printing each entry
// as a val idx pair. 
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::print_ccs() {

  if(!is_ccs_sorted_)
    sort_ccs();

  size_t ccs_loc;
  std::cout << "entries: "<<curr_size_ << "  rows: " << n_cols_ << std::endl;
  for( size_t jj = 0; jj<n_cols_; jj++){
    for (size_t ii = jj*max_rowcol_nnz_; ii < col_ends_[jj]; ii++) {
      ccs_loc = ccs_order_[ii];
      std::cout << entries_[ccs_loc].rowidx << "\t " << entries_[ccs_loc].colidx << "\t" << entries_[ccs_loc].val << std::endl;
    }
  }
  std::cout << std::endl;
}


// Print a matrix to cout by printing the number 
// of nonzero entries and then printing each entry
// as a val idx pair. 
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::print_crs() {

  if(!is_crs_sorted_)
    sort_crs();

  size_t crs_loc;
  std::cout << "entries: "<<curr_size_ << "  rows: "<< n_rows_ << std::endl;
  for( size_t jj = 0; jj<n_rows_; jj++){
    for (size_t ii = jj*max_rowcol_nnz_; ii < row_ends_[jj]; ii++) {
      crs_loc = crs_order_[ii];
      std::cout << entries_[crs_loc].rowidx << "\t " << entries_[crs_loc].colidx << "\t" << entries_[crs_loc].val << std::endl;
    }
  }
  std::cout << std::endl;
}

// sparse_colwisemv multiplies each column of 
// the sparse matrix A by the corresponding
// entry of the sparse vector x.
// A is specified by a routine Acolumn that returns 
// a single column of A.  max_nz_col_entries is an
// upper bound on the number of non-zero entries in
// any column of A.  The resulting matrix B 
// is assumed to be of max_size
// at least max_nz_col_entries * x.curr_size_.
// B is overwritten upon output.
template <typename IdxType, typename ValType, typename ParType>
inline void sparse_colwisemv(int (*Acolumn)(SparseVector<IdxType, ValType>&, const IdxType, const ParType),
        const ParType params, const size_t max_nz_col_entries, const SparseVector<IdxType, ValType> &x, SparseMatrix<IdxType,ValType>& B)
{
  // Check for correct size; multiplication must not overflow.
  assert(max_nz_col_entries<=B.max_rowcol_nnz_);
  assert(x.size() <= B.max_cols_);

  B.clear();

  // Make a list of all the entries in A scaled
  // by the entry of x corresponding to the column
  // containing the particular entry of A; these
  // will all be added to the result.
  SparseVector<IdxType, ValType> single_column_add(max_nz_col_entries);
  for (size_t jj = 0; jj < x.size(); jj++) {
    single_column_add.clear();
    Acolumn(single_column_add, x[jj].idx, params);
    single_column_add *= x[jj].val;
    B.set_col(single_column_add,jj);
  }

  return;
}















//---------------------------------------------------------
// Sparse vector compression helper class and routines
// Declaration
//---------------------------------------------------------

// Sparse vector compression class.
template <typename IdxType, typename ValType, class RNG>
class Compressor {
private:
  
  // Temporary storage for an intermediate.
  std::valarray<size_t> ind_vec_;
  std::valarray<double> xabs_;
  
  // Random number generator.
  RNG* gen_;
  std::uniform_real_distribution<> uu_;
  
  // Do not allow copying or assignment for compressors.
  Compressor<IdxType, ValType, RNG>(Compressor<IdxType, ValType, RNG> &);
  Compressor<IdxType, ValType, RNG>& operator= (Compressor<IdxType, ValType, RNG> &);

  size_t nnz_;
  
  // Helper function: compress the internal 
  // vector of moduli.
  inline size_t preserve_xabs(size_t target_nnz);
  // inline void compress_xabs_sys(size_t target_nnz);
  // inline void compress_xabs_dmc(size_t target_nnz);

public:

  // Constructor based on maximum size of the
  // modulus vector and a random seed.
  inline Compressor<IdxType, ValType, RNG>(size_t max_size, RNG& generator) : 
    xabs_(max_size) {
    // Set up the pseudorandom number generator.
    gen_ = &generator;
    uu_ = std::uniform_real_distribution<>(0,1);
  }
  
  // Compress a SparseVector using the stored
  // temp vector and pseudorandom number generator.
  void compress(SparseVector<IdxType, ValType> &x, size_t target_nnz);

  // Return a mask of the sparse vector indicating which entries should
  // be preserved exactly in a compression.
  std::vector<size_t> preserve(SparseVector<IdxType, ValType> &x, size_t target_nnz);
};



template <class RNG>
inline size_t resample_sys(std::valarray<double>& xabs_, size_t target_nnz, RNG* gen_);

template <class RNG>
inline size_t resample_piv(std::valarray<double>& xabs_, size_t target_nnz, RNG* gen_);



//---------------------------------------------------------
// Sparse vector compression helper class and routines
// Implementation
//---------------------------------------------------------

using std::abs;

// Compress a SparseVector using the stored
// temp vector and pseudorandom number generator.
template <typename IdxType, typename ValType, class RNG>
inline void Compressor<IdxType, ValType, RNG>::compress(SparseVector<IdxType, ValType> &x, size_t target_nnz) {
  
  // if x already has less than target_nnz non-zero entries this 
  // is a do nothing routine.
  if(x.size()<=target_nnz){
    return;
  }


  xabs_.resize(x.size());
  ind_vec_.resize(x.size());

  // Copy the modulus of each entry into xabs_.
  for (size_t jj = 0; jj < ind_vec_.size(); jj++){
    ind_vec_[jj] = jj;
    xabs_[jj] = abs(x[jj].val);
    if( x[jj].val==0){
      std::cout << jj << std::endl;
    }
  }

  // Set tiny entries to zero.
  double xabs_sum = xabs_.sum();
  double Tol = 1e-12;
  for (size_t jj = 0; jj < xabs_.size(); jj++) {
    if (xabs_[jj] < Tol * xabs_sum) {
      xabs_[jj] = 0.0;
    }
  }
  xabs_sum = xabs_.sum();

  size_t nnz_large = preserve_xabs(target_nnz);
  //size_t nnz_large = 0;
  assert(nnz_large<=target_nnz);

  if (nnz_large < target_nnz){
    size_t nnz_small = xabs_.size()-nnz_large;
    size_t ii;

    // write in the preserved entries.
    for( size_t jj = nnz_small; jj < ind_vec_.size(); jj++ ){
      ii = ind_vec_[jj];
      assert(x[ii].val != 0);
      x.set_value(ii, (x[ii].val / abs(x[ii].val))*xabs_[ii]);
    }

    //std::cout<<nnz_small<<std::endl;

    // Cut the preserved entries out of xabs_
    std::valarray<size_t> ind_resample = ind_vec_[std::slice(0,nnz_small,1)];
    std::valarray<double> xabs_resample = xabs_[ind_resample];
    xabs_sum = xabs_resample.sum();
    xabs_resample /= xabs_sum;

    // Resample the remaining entries.
    resample_piv(xabs_resample,target_nnz-nnz_large, gen_);
    //resample_sys(xabs_resample,target_nnz-nnz_large, gen_);

    // compress_xabs_sys(target_nnz-nnz_large);
    // Translate the compression of the moduli vector to
    // a compression of the input vector. For each entry
    // of the compressed xabs,
    for( size_t jj = 0; jj < nnz_small; jj++){
      // Find the corresponding member of x and
      // set its modulus according to the modulus
      // of xabs.
      //std::cout<< ind_resample[jj] <<"\t"<< x[ind_resample[jj]].val << std::endl;
      assert(x[ind_resample[jj]].val != 0);
      x.set_value(ind_resample[jj],(x[ind_resample[jj]].val / abs(x[ind_resample[jj]].val)) 
        * xabs_resample[jj] * xabs_sum/(double)(target_nnz-nnz_large));
    }
  }
  // Remove the entries set to zero.
  x.remove_zeros();
  // The vector is now compressed.
  return;
}

// Return the indices of the sparse vector that should
// be preserved in a compression.
template <typename IdxType, typename ValType, class RNG>
inline std::vector<size_t> Compressor<IdxType, ValType, RNG>::preserve(SparseVector<IdxType, ValType> &x, size_t target_nnz) {

  // if x already has less than target_nnz non-zero entries this 
  // is a do nothing routine.
  if(x.curr_size_<=target_nnz){
    size_t nnz_large = x.curr_size_;
    std::vector<size_t> result(nnz_large);
    for (size_t jj = 0; jj < ind_vec_.size(); jj++){
      result[jj] = jj;
    }
    return result;
  }

  xabs_.resize(x.curr_size_);
  ind_vec_.resize(x.curr_size_);

  for (size_t jj = 0; jj < ind_vec_.size(); jj++){
    ind_vec_[jj] = jj;
    xabs_[jj] = abs(x[jj]);
    if( x[jj]==0){
      std::cout << jj << std::endl;
    }
  }

  // Find entries to be preserved.
  size_t nnz_large = preserve_xabs(target_nnz);
  size_t nnz_small = xabs_.size()-nnz_large;

  std::vector<size_t> result(nnz_large);

  // Translate the compression of the moduli vector to
  // a compression of the input vector. For each entry
  // of the compressed xabs,
  for(size_t jj = nnz_small; jj < xabs_.size(); jj++){
    // Find the corresponding member of x and
    // set its modulus according to the modulus
    // of xabs.
    result[jj-nnz_small] = ind_vec_[jj];
  }
  return result;
}

// Move the largest entries to the back of xabs so they will be preserved exactly
// by the compression.
template <typename IdxType, typename ValType, class RNG>
inline size_t Compressor<IdxType, ValType, RNG>::preserve_xabs(size_t target_nnz) {

  if (xabs_.size()<=target_nnz){
    return xabs_.size();
  }
  else{
    size_t imax;
    double sum;

    // Find the maximum and storage position of the maximum.
    double dmax = 0;
    for (size_t jj = 0; jj < ind_vec_.size(); jj++) {
      if (xabs_[ind_vec_[jj]] > dmax) {
        imax = jj;
        dmax = xabs_[imax];
      }
    }
    // Place it at the end of the stored vector;
    // we are building a new vector in place by
    // transferring entries within the old one.
    std::swap( ind_vec_[imax], ind_vec_[ind_vec_.size()-1] );

    // Check if there are any elements large
    // enough to be preserved exactly.  If so
    // heapify and pull large entries until remaining
    // entries are not large enough to be preserved
    // exactly.

  // sum = 0;
  //   for( size_t jj=0; jj<ind_vec_.size(); jj++) sum+= xabs_[jj];  

    //std::cout<< sum << "\t"<< xabs_.sum() << std::endl;
    //assert(1<0); 

    size_t nnz_large = 0;
    double sum_unprocessed = xabs_.sum();
    if (target_nnz * dmax > sum_unprocessed) {
      nnz_large = 1;
      sum_unprocessed -= dmax;
      std::make_heap(begin(ind_vec_),end(ind_vec_) - nnz_large,
        [&](size_t ii, size_t jj){return xabs_[ii]<xabs_[jj];} );

      while (( (target_nnz - nnz_large) * xabs_[ind_vec_[0]] > sum_unprocessed) 
               and (nnz_large < target_nnz)) {
        sum_unprocessed -= xabs_[ind_vec_[0]];
        //std::cout << sum_unprocessed << std::endl; 
        assert(sum_unprocessed >0);
        std::pop_heap(begin(ind_vec_),end(ind_vec_) - nnz_large, 
          [&](size_t ii, size_t jj){return xabs_[ii]<xabs_[jj];} );
        nnz_large++;
      }
    }
    return nnz_large;
  }
}



template <class RNG>
inline size_t resample_sys(std::valarray<double>& xabs_, size_t target_nnz, RNG* gen_) {

  std::uniform_real_distribution<> uu_=std::uniform_real_distribution<>(0,1);

  xabs_ *= (double)target_nnz;

  size_t jj = 0;
  double w = xabs_[0];
  xabs_[0] = 0;
  double U = uu_(*gen_);
  for(size_t ii=0; ii<target_nnz; ii++ ){
    // double U = uu_(gen_);
    while( w < ii + U ){
      jj++;
      w += xabs_[jj];
      xabs_[jj] = 0;
    }
    xabs_[jj]+=1.0;
    //assert(xabs_[jj] < 2);
  }
  jj++;
  while( jj < xabs_.size()){
    xabs_[jj] = 0;
    jj++; 
  }

  size_t nnz = 0;
  for (size_t jj = 0; jj < xabs_.size(); jj++) {
    if( xabs_[jj] > 0) nnz++;
  }
  return nnz;
}





template <class RNG>
inline size_t resample_piv(std::valarray<double>& xabs_, size_t target_nnz, RNG* gen_) {

  std::uniform_real_distribution<> uu_=std::uniform_real_distribution<>(0,1);

  xabs_ *= (double)target_nnz;

  //std::cout<<target_nnz<<std::endl;

  size_t ii = 0;
  size_t jj = 1;
  size_t kk = 2;
  double a=xabs_[0]-floor(xabs_[0]), b=xabs_[1]-floor(xabs_[1]);
  double EPS = 1e-6;

  // for(size_t ll=0; ll<xabs_.size(); ll++)
  //   std::cout<<ll<<" "<<xabs_[ll]<<std::endl;
  //std::cout <<xabs_.sum()<<std::endl;

  while( kk< xabs_.size() ){
    //std::cout<<kk<<" "<<ii<<" "<<jj<<" "<<a<<" "<<b<<" "<<xabs_[kk]<<std::endl;
    if( a>=EPS and b<=1.0-EPS and a+b>1.0 ){
      if( uu_(*gen_)<(1.0-b)/(2.0-a-b) ){
        b+=a-1.0;
        a=1.0;
      }
      else{
        a+=b-1.0;
        b=1.0;
      }
    }
    if ( a>=EPS and b<=1.0-EPS and a+b<=1.0 ){
      if ( uu_(*gen_)< b/(a+b) ){
        b+=a;
        a=0;
      }
      else{
        a+=b;
        b=0;
      }
    }
    if ( (a<EPS or a>1.0-EPS) and kk<xabs_.size() ){
      //std::cout<<a<<" "<<b<<std::endl;
      xabs_[ii] = a + floor(xabs_[ii]);
      a = xabs_[kk] - floor(xabs_[kk]);
      ii = kk;
      kk++;
    }
    if ( (b<EPS or b>1.0-EPS) and kk<xabs_.size() ){
      //std::cout<<a<<" "<<b<<std::endl;
      xabs_[jj]=b + floor(xabs_[jj]);
      b = xabs_[kk] - floor(xabs_[kk]);
      jj = kk;
      kk++;
    }
  }

  //std::cout<<a<<" "<<b<<std::endl;
  if( a>=EPS and b<=1.0-EPS and a+b>1.0 ){
    if( uu_(*gen_)<(1.0-b)/(2.0-a-b) ){
      b+=a-1.0;
      a=1.0;
    }
    else{
      a+=b-1.0;
      b=1.0;
    }
  }
  if ( a>=EPS and b<=1.0-EPS and a+b<=1.0 ){
    if( uu_(*gen_)<b/(a+b) ){
      b+=a;
      a=0;
    }
    else{
      a+=b;
      b=0;
    }
  }
  xabs_[ii] = a + floor(xabs_[ii]);
  xabs_[jj] = b + floor(xabs_[jj]);

  return 0;
}








#endif