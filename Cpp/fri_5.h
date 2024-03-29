// declaration of generic Fast Randomized Iteration
// subroutines
// (c) Jonathan Weare 2015

#ifndef _fri_5_h_
#define _fri_5_h_

#include <iostream> 
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <numeric>
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

  // norm() returns the sum of the current entry magnitudes.
  inline const ValType norm() const;

  // sum() returns the sum of the current entry values.
  inline const ValType sum() const;

  // Constructor taking max_size as an argument and 
  // allocating space for that many SparseEntry structs.
  SparseVector(size_t max_size);

  SparseVector();

  void remove_zeros();

  inline void clear();

  inline const size_t size() const {return curr_size_;}

  inline const size_t capacity() const {return max_size_;}

  //void normalize();

  inline void print();

  // Assignment by value up to current size, leaving
  // max size & other entries unchanged.
  inline SparseVector<IdxType, ValType>& operator=(const SparseVector<IdxType, ValType> &other);

  inline SparseVector<IdxType, ValType>& operator*=(const ValType alpha);
  inline SparseVector<IdxType, ValType>& operator/=(const ValType alpha);
  inline SparseVector<IdxType, ValType>& operator+=(const ValType alpha);
  inline SparseVector<IdxType, ValType>& operator-=(const ValType alpha);

  inline SparseVector<IdxType, ValType>& operator+=(const SparseVector<IdxType, ValType> &other);
  inline SparseVector<IdxType, ValType>& operator-=(const SparseVector<IdxType, ValType> &other);
  inline SparseVector<IdxType, ValType>& operator*=(const SparseVector<IdxType, ValType> &other);

  inline ValType dot(const SparseVector<IdxType, ValType> &other);

  inline void allocate(size_t max_size);

  // Accessors for the underlying vector so one can do
  // some of the normal vector manipulation, but not all.
  // Specifically, subscripting and iterator begin and end
  // are provided. Incremental queue and stack function is 
  // intentionally not provided.
  //inline ValType& operator[](const size_t idx) {return entries_[idx].val;}
  inline const SparseVectorEntry<IdxType,ValType>& operator[](const size_t idx) const {assert(idx<curr_size_); return entries_[idx];}
  inline void set_entry(const SparseVectorEntry<IdxType,ValType>& other);
  inline void set_entry(const IdxType idx, const ValType val);
  inline void set_value(const size_t ent_num, const ValType val);
  //inline const SparseVectorEntry<IdxType, ValType>& get_entry(const size_t idx) const {return entries_[idx];}
  typedef typename std::vector<SparseVectorEntry<IdxType, ValType>>::iterator spvec_iterator;
  inline spvec_iterator begin() {return entries_.begin();}
  inline spvec_iterator end() {return entries_.begin() + curr_size_;}

private:
  // Number of nonzero entries that could be in the vector.
  // Can't be changed by user.
  size_t max_size_;

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
inline void SparseVector<IdxType, ValType>::set_value(const size_t ent_num, const ValType val){
  assert(ent_num<curr_size_);

  entries_[ent_num].val = val;
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

  // Constructor taking max_size as an argument and 
  // allocating space for that many SparseVectorEntry structs.
template <typename IdxType, typename ValType>
inline SparseVector<IdxType, ValType>::SparseVector(){
  max_size_ = 0;
}




template <typename IdxType, typename ValType>
inline void SparseVector<IdxType, ValType>::allocate(size_t max_size){
  max_size_ = max_size;

  curr_size_ = 0;
  entries_ = std::vector<SparseVectorEntry<IdxType, ValType>>(max_size);

  return;
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

  if(max_size_==0){
    max_size_ = other.max_size_;
    entries_ = std::vector<SparseVectorEntry<IdxType, ValType>>(max_size_);
  }

  assert(other.curr_size_ <= max_size_);
  curr_size_ = 0;
  for (size_t ii = 0; ii < other.curr_size_; ii++) {
    if ( other[ii].val != 0 ){
      entries_[curr_size_] = other.entries_[ii];
      curr_size_++;
    }
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
// template <typename IdxType, typename ValType>
// inline void SparseVector<IdxType, ValType>::remove_zeros() {

//   size_t jj = 0;
//   while (jj<curr_size_){
//     if (entries_[jj].val == 0){
//       std::rotate(entries_.begin()+jj,entries_.begin()+jj+1,entries_.begin()+curr_size_);
//       curr_size_--;
//     }
//     else{
//       jj++;
//     }
//   }
//   return;
// }


// Remove all zero elements from a sparse vector.
template <typename IdxType, typename ValType>
inline void SparseVector<IdxType, ValType>::remove_zeros() {
  size_t nnz = 0;
  for(size_t jj = 0; jj < curr_size_; jj++){
    if (entries_[jj].val != 0){
      entries_[nnz] = entries_[jj];
      nnz++;
    }
  }
  curr_size_ = nnz;
  return;
}

// Print a vector to cout by printing the number 
// of nonzero entries and then printing each entry
// as a val idx pair. 
template <typename IdxType, typename ValType>
inline void SparseVector<IdxType, ValType>::print() {
  std::cout << "entries: "<< curr_size_ << std::endl;
  for (size_t jj = 0; jj < curr_size_; jj++) {
    std::cout << entries_[jj].idx <<"\t"<<entries_[jj].val << std::endl;
  }
  std::cout << std::endl;
  return;
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

  // std::cout<<x.size()<<" "<<curr_size_<<" "<<max_size_<<std::endl;

  assert(x.size() + curr_size_<= max_size_);

  if( curr_size_==0 ){
    *this = x;
    return *this;
  }

  if( x.size()==0 ){
    return *this;
  }

  std::vector<SparseVectorEntry<IdxType,ValType>> work(x.size()+curr_size_);

  size_t y_entries_begin = 0;
  size_t y_entries_end = curr_size_;
  size_t n_result_entries = 0;
  // Move over the entries in each vector from least
  // index to highest, iterating on the x vector as
  // the driver loop and y as a following iterator.
  size_t jj = 0;
  while (jj < x.size()){
    // If there are y entries left and the y entry has lower
    // index, move it in.
    while (y_entries_begin < y_entries_end and entries_[y_entries_begin].idx < x[jj].idx){
      work[n_result_entries] = entries_[y_entries_begin];
      y_entries_begin++;
      if( work[n_result_entries].val!=0 ){
        n_result_entries++;
      }
    }
    // If the x and y entries had equal index, add the x 
    // entry times alpha to the y entry.
    while ((jj<x.size()) and (y_entries_begin < y_entries_end) and (x[jj].idx == entries_[y_entries_begin].idx)){
      work[n_result_entries] = entries_[y_entries_begin];
      work[n_result_entries].val += x[jj].val;
      jj++;
      y_entries_begin++;
      if( work[n_result_entries].val != 0 ){
        n_result_entries++;
      }
    }
    // Otherwise, just move the x entry in. 
    while (jj<x.size() and (y_entries_begin < y_entries_end) and (entries_[y_entries_begin].idx > x[jj].idx)){
      work[n_result_entries].val = x[jj].val;
      work[n_result_entries].idx = x[jj].idx;
      jj++;
      if( work[n_result_entries].val != 0 ){
        n_result_entries++;
      }
    }

    while (jj<x.size() and (y_entries_begin==y_entries_end) ){
      work[n_result_entries].val = x[jj].val;
      work[n_result_entries].idx = x[jj].idx;
      jj++;
      if( work[n_result_entries].val != 0 ){
        n_result_entries++;
      }
    }

    // std::cout<<jj<<" "<<y_entries_begin<<" "<<curr_size_<<" "<<x.size()<<" "<<n_result_entries<<std::endl;
  }
  // If all x entries are handled and y entries remain, move
  // them all in.
  while (y_entries_begin < y_entries_end){
    work[n_result_entries].val = entries_[y_entries_begin].val;
    work[n_result_entries].idx = entries_[y_entries_begin].idx;
    y_entries_begin++;
    if( work[n_result_entries].val != 0 ){
      n_result_entries++;
    }
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

  assert(x.size() + curr_size_<= max_size_);

  if( curr_size_==0 ){
    *this = x;
    return *this;
  }

  if( x.size()==0 ){
    return *this;
  }

  std::vector<SparseVectorEntry<IdxType,ValType>> work(x.size()+curr_size_);

  size_t y_entries_begin = 0;
  size_t y_entries_end = curr_size_;
  size_t n_result_entries = 0;
  // Move over the entries in each vector from least
  // index to highest, iterating on the x vector as
  // the driver loop and y as a following iterator.
  size_t jj = 0;
  while (jj < x.size()){
    // If there are y entries left and the y entry has lower
    // index, move it in.
    while (y_entries_begin < y_entries_end and entries_[y_entries_begin].idx < x[jj].idx){
      work[n_result_entries] = entries_[y_entries_begin];
      y_entries_begin++;
      if( work[n_result_entries].val!=0 ){
        n_result_entries++;
      }
    }
    // If the x and y entries had equal index, add the x 
    // entry times alpha to the y entry.
    while ((jj<x.size()) and (y_entries_begin < y_entries_end) and (x[jj].idx == entries_[y_entries_begin].idx)){
      work[n_result_entries] = entries_[y_entries_begin];
      work[n_result_entries].val -= x[jj].val;
      jj++;
      y_entries_begin++;
      if( work[n_result_entries].val != 0 ){
        n_result_entries++;
      }
    }
    // Otherwise, just move the x entry in. 
    while (jj<x.size() and (y_entries_begin < y_entries_end) and (entries_[y_entries_begin].idx > x[jj].idx)){
      work[n_result_entries].val = -x[jj].val;
      work[n_result_entries].idx = x[jj].idx;
      jj++;
      if( work[n_result_entries].val != 0 ){
        n_result_entries++;
      }
    }

    while (jj<x.size() and (y_entries_begin==y_entries_end) ){
      work[n_result_entries].val = -x[jj].val;
      work[n_result_entries].idx = x[jj].idx;
      jj++;
      if( work[n_result_entries].val != 0 ){
        n_result_entries++;
      }
    }

    // std::cout<<jj<<" "<<y_entries_begin<<" "<<curr_size_<<" "<<x.size()<<" "<<n_result_entries<<std::endl;
  }
  // If all x entries are handled and y entries remain, move
  // them all in.
  while (y_entries_begin < y_entries_end){
    work[n_result_entries].val = entries_[y_entries_begin].val;
    work[n_result_entries].idx = entries_[y_entries_begin].idx;
    y_entries_begin++;
    if( work[n_result_entries].val != 0 ){
      n_result_entries++;
    }
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
inline SparseVector<IdxType, ValType>& SparseVector<IdxType, ValType>::operator*=(const SparseVector<IdxType, ValType>& x){

  if( curr_size_==0 ){
    *this = x;
    return *this;
  }

  if( x.size()==0 ){
    curr_size_=0;
    return *this;
  }

  std::vector<SparseVectorEntry<IdxType,ValType>> work(std::min(x.size(),curr_size_));

  size_t y_entries_begin = 0;
  size_t y_entries_end = curr_size_;
  size_t n_result_entries = 0;
  // Move over the entries in each vector from least
  // index to highest, iterating on the x vector as
  // the driver loop and y as a following iterator.
  size_t jj = 0;
  while (jj < x.size() and y_entries_begin<y_entries_end ){
    // If there are y entries left and the y entry has lower
    // index, move it in.
    while (y_entries_begin < y_entries_end and entries_[y_entries_begin].idx < x[jj].idx){
      y_entries_begin++;
    }
    // If the x and y entries had equal index, add the x 
    // entry times alpha to the y entry.
    while ((jj<x.size()) and (y_entries_begin < y_entries_end) and (x[jj].idx == entries_[y_entries_begin].idx)){
      work[n_result_entries] = entries_[y_entries_begin];
      work[n_result_entries].val *= x[jj].val;
      jj++;
      y_entries_begin++;
      if( work[n_result_entries].val != 0 ){
        n_result_entries++;
      }
    }
    // Otherwise, just move the x entry in. 
    while (jj<x.size() and (y_entries_begin < y_entries_end) and (entries_[y_entries_begin].idx > x[jj].idx)){
      jj++;
    }
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
inline ValType SparseVector<IdxType, ValType>::dot(const SparseVector<IdxType, ValType>& x){

  if( curr_size_==0 ){
    return 0;
  }

  if( x.size()==0 ){
    return 0;
  }

  ValType product = 0;

  size_t y_entries_begin = 0;
  size_t y_entries_end = curr_size_;
  size_t n_result_entries = 0;
  // Move over the entries in each vector from least
  // index to highest, iterating on the x vector as
  // the driver loop and y as a following iterator.
  size_t jj = 0;
  while (jj < x.size() and y_entries_begin<y_entries_end ){
    // If there are y entries left and the y entry has lower
    // index, move it in.
    while (y_entries_begin < y_entries_end and entries_[y_entries_begin].idx < x[jj].idx){
      y_entries_begin++;
    }
    // If the x and y entries had equal index, add the x 
    // entry times alpha to the y entry.
    while ((jj<x.size()) and (y_entries_begin < y_entries_end) and (x[jj].idx == entries_[y_entries_begin].idx)){
      product += entries_[y_entries_begin].val*x[jj].val;
      jj++;
      y_entries_begin++;
    }
    // Otherwise, just move the x entry in. 
    while (jj<x.size() and (y_entries_begin < y_entries_end) and (entries_[y_entries_begin].idx > x[jj].idx)){
      jj++;
    }
  }

  return product;
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

  // Constructor taking max_rows, max_cols, and max_size as an argument and 
  // allocating space for that many SparseMatrixEntry structs.
  SparseMatrix(size_t max_rows, size_t max_cols, size_t max_rowcol_nnz);
  SparseMatrix();

  inline void allocate(size_t max_rows, size_t max_cols, size_t max_rowcol_nnz);

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

  inline const size_t rowcol_capacity() const {return max_rowcol_nnz_;}

  inline const size_t capacity() const {return max_size_;}

  inline const size_t max_rows() const {return max_rows_;}

  inline const size_t max_cols() const {return max_cols_;}

  inline const size_t ncols() const {return n_cols_;}

  inline const size_t nrows() const {return n_rows_;} 

  // Reset matrix without deallocating space.
  inline void clear();

  // Write column idx into a sparse vector.
  inline void get_col(const size_t col_num, SparseVector<IdxType,ValType>& other);

  inline void get_all_col_values(const size_t col_num, std::vector<ValType>& other);

  // Write row idx into a sparse vector.
  inline void get_row(const size_t row_num, SparseVector<IdxType,ValType>& other);

  inline void get_all_row_values(const size_t row_num, std::vector<ValType>& other);

  // Add a new column to the matrix.  If there's already a column with index idx
  // it is replaced.
  inline void set_col(const SparseVector<IdxType,ValType>& other, const IdxType idx);

  inline const size_t col_size(const size_t col_num);

  inline const size_t row_size(const size_t row_num);

  // Add a new row to the matrix.  If there's already a row with index idx
  // it is replaced.
  inline void set_row(const SparseVector<IdxType,ValType>& other, const IdxType idx);

  inline void scale_col(const size_t col_num, const ValType a);

  inline void clear_col(const size_t col_num);

  inline void clear_row(const size_t row_num);

  // Remove a column from the matrix.
  inline void eject_col(const size_t col_num);

  inline void eject_row(const size_t row_num);

  // Compute column sums and output in sparse vector.
  inline void col_sums(std::vector<ValType>& other);

  // Compute column norms and output in sparse vector.
  inline void col_norms(std::vector<ValType>& other);

  // Compute row sums and output in sparse vector.
  inline void row_sums(std::vector<ValType>& other);

  inline void row_sums(SparseVector<IdxType,ValType>& other);

  // Compute row norms and output in sparse vector.
  inline void row_norms(std::vector<ValType>& other);

  inline size_t locate_col(const IdxType idx);

  inline size_t locate_row(const IdxType idx);

  inline const IdxType col_idx(const size_t col_num);

  inline const IdxType row_idx(const size_t row_num);

  // Get entry in CCS order
  // inline const SparseMatrixEntry<IdxType, ValType>& get_ccs_entry(const size_t idx) const {return entries_[idx];}

  // Get entry in CCR order
  //inline const SparseMatrixEntry<IdxType, ValType>& get_crs_entry(const size_t idx) const {return entries_[idx];}

  // Print a matrix to cout by printing the number 
  // of nonzero entries and then printing each entry
  // as a val idx pair. 
  inline void print_ccs();
  inline void print_crs();

  inline SparseMatrixEntry<IdxType,ValType>& get_col_entry(const size_t col_num, const size_t ent_num);
  inline SparseMatrixEntry<IdxType,ValType>& get_row_entry(const size_t row_num, const size_t ent_num);
  inline size_t get_col_entry_number(const size_t col_num, const size_t ent_num);
  inline size_t get_row_entry_number(const size_t row_num, const size_t ent_num);
  inline size_t col_entry2crs_pos(const size_t col_num, const size_t col_ent_num);
  inline size_t row_entry2ccs_pos(const size_t row_num, const size_t row_ent_num);

  inline void set_col_value(const size_t col_num, const size_t ent_num, const ValType value);
  inline void set_row_value(const size_t row_num, const size_t ent_num, const ValType value);

  inline void set_all_col_values(const size_t col_num, const ValType value);
  inline void set_all_col_values(const size_t col_num, const std::vector<ValType>& other);

  inline void set_all_row_values(const size_t row_num, const ValType value);
  inline void set_all_row_values(const size_t row_num, const std::vector<ValType>& other);

  inline size_t locate_entry_ccs( const IdxType rowidx, const IdxType colidx );
  inline size_t locate_entry_crs( const IdxType rowidx, const IdxType colidx );

  // Accessors for the underlying vector so one can do
  // some of the normal vector manipulation, but not all.
  // Specifically, subscripting and iterator begin and end
  // are provided. Incremental queue and stack function is 
  // intentionally not provided.
  //inline size_t ccs_lower_bound(const IdxType rowidx, const IdxType colidx);
  //inline void set_entry(const SparseMatrixEntry<IdxType, ValType>& a, const size_t idx);
  inline const SparseMatrixEntry<IdxType,ValType>& operator[](const size_t idx) const {assert(idx<curr_size_); return entries_[idx];}
  //inline const SparseMatrixEntry<IdxType, ValType>& get_entry(const size_t idx) const {return entries_[idx];}
  typedef typename std::vector<SparseMatrixEntry<IdxType, ValType>>::iterator spmat_iterator;
  inline spmat_iterator begin() {return entries_.begin();}
  inline spmat_iterator end() {return entries_.begin() + curr_size_;}

private:
  // Number of nonzero entries that could be in the vector.
  // Must not change.
  size_t max_size_;

  // Max number of rows and columns.
  size_t max_cols_, max_rows_, max_rowcol_nnz_;


  // Number of nonzero entries actually in the vector.
  size_t curr_size_;
  // Number of rows and columns.
  size_t n_rows_, n_cols_;
  // Do not allow manual pushing/popping of the entries 
  // or anything else that is not ordering safe.
  std::vector<SparseMatrixEntry<IdxType, ValType>> entries_;
  std::vector<size_t> inv_ccs_order_;   // lists the column-first position of the indices
  std::vector<size_t> inv_crs_order_;  // lists the row-first position of the indices
  std::vector<size_t> ccs_order_;  // lists the indices in column-first sorted order
  std::vector<size_t> crs_order_;  // lists the indices in row-first sorted order
  std::vector<size_t> row_lens_;   // the position (in row-first order) of the end of each row
  std::vector<size_t> col_lens_;    // the position (in column-first order) of the end of each column
  std::vector<size_t> counter_;
  inline void inject_col(const size_t col_num);
  inline void inject_row(const size_t row_num);
  inline void fill_col(const size_t col_num, const IdxType idx, const SparseVector<IdxType,ValType>& other);
  inline void fill_row(const size_t row_num, const IdxType idx, const SparseVector<IdxType,ValType>& other);
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
inline SparseMatrix<IdxType, ValType>::SparseMatrix(size_t max_rows, size_t max_cols, size_t max_rowcol_nnz)
  : max_size_(std::min(max_cols,max_rows)*max_rowcol_nnz), max_rows_(max_rows), max_cols_(max_cols), max_rowcol_nnz_(max_rowcol_nnz) {
  curr_size_ = 0;
  n_rows_ = 0;
  n_cols_ = 0;
  entries_ = std::vector<SparseMatrixEntry<IdxType, ValType>>(std::min(max_rows,max_cols)*max_rowcol_nnz);
  inv_ccs_order_ = std::vector<size_t>(std::min(max_rows,max_cols)*max_rowcol_nnz);
  inv_crs_order_ = std::vector<size_t>(std::min(max_rows,max_cols)*max_rowcol_nnz);
  crs_order_ = std::vector<size_t>(max_rows*max_rowcol_nnz);
  ccs_order_ = std::vector<size_t>(max_cols*max_rowcol_nnz);
  row_lens_ = std::vector<size_t>(max_rows,0);
  col_lens_ = std::vector<size_t>(max_cols,0);
  counter_ = std::vector<size_t>( std::max(max_rows,max_cols) );

  std::iota(counter_.begin(),counter_.end(),0);


  is_crs_sorted_ = true;
  is_ccs_sorted_ = true;
}



// Constructor taking max_size as an argument and 
// allocating space for that many SparseVectorEntry structs.
template <typename IdxType, typename ValType>
inline SparseMatrix<IdxType, ValType>::SparseMatrix(){
  max_size_ = 0;
}




template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::allocate(size_t max_rows, size_t max_cols, size_t max_rowcol_nnz){

  max_size_ = std::min(max_cols,max_rows)*max_rowcol_nnz;
  max_rows_ = max_rows;
  max_cols_ = max_cols;
  max_rowcol_nnz_ = max_rowcol_nnz;
  
  curr_size_ = 0;
  n_rows_ = 0;
  n_cols_ = 0;
  entries_ = std::vector<SparseMatrixEntry<IdxType, ValType>>(std::min(max_rows,max_cols)*max_rowcol_nnz);
  inv_ccs_order_ = std::vector<size_t>(std::min(max_rows,max_cols)*max_rowcol_nnz);
  inv_crs_order_ = std::vector<size_t>(std::min(max_rows,max_cols)*max_rowcol_nnz);
  crs_order_ = std::vector<size_t>(max_rows*max_rowcol_nnz);
  ccs_order_ = std::vector<size_t>(max_cols*max_rowcol_nnz);
  row_lens_ = std::vector<size_t>(max_rows,0);
  col_lens_ = std::vector<size_t>(max_cols,0);
  counter_ = std::vector<size_t>( std::max(max_rows,max_cols) );

  std::iota(counter_.begin(),counter_.end(),0);


  is_crs_sorted_ = true;
  is_ccs_sorted_ = true;

  return;
}





// Reset sparse matrix without deallocating space.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::clear(){
  curr_size_ = 0;
  n_rows_ = 0;
  n_cols_ = 0;
  std::fill(row_lens_.begin(), row_lens_.end(),0);
  std::fill(col_lens_.begin(), col_lens_.end(),0);
  is_crs_sorted_ = true;
  is_ccs_sorted_ = true;
}





template <typename IdxType, typename ValType>
inline const size_t SparseMatrix<IdxType, ValType>::col_size(const size_t col_num){
  if (!is_ccs_sorted_)
    sort_ccs();

  assert(col_num<n_cols_);

  return col_lens_[col_num];
}





template <typename IdxType, typename ValType>
inline const size_t SparseMatrix<IdxType, ValType>::row_size(const size_t row_num){
  if (!is_crs_sorted_)
    sort_crs();

  assert(row_num<n_rows_);

  return row_lens_[row_num];
}


// returns the index of column number col_num.  If that column is empty returns
// the index of the next non-empty column.  The last 
// column is not allowed to be empty.
template <typename IdxType, typename ValType>
inline const IdxType SparseMatrix<IdxType, ValType>::col_idx(const size_t col_num){
  if (!is_ccs_sorted_)
    sort_ccs();

  assert(col_num<=n_cols_);

  if( col_num==n_cols_ ){
    return entries_[curr_size_].colidx;
  }

  size_t jj=col_num, col_head;

  while( jj<n_cols_ and col_lens_[jj]==0 ){
    jj++;
  }
  assert(jj<n_cols_);
  col_head = jj*max_rowcol_nnz_;
  return entries_[ccs_order_[col_head]].colidx;
}



template <typename IdxType, typename ValType>
inline const IdxType SparseMatrix<IdxType, ValType>::row_idx(const size_t row_num){
  if (!is_crs_sorted_)
    sort_crs();

  assert(row_num<=n_rows_);

  if( row_num==n_rows_ ){
    return entries_[curr_size_].rowidx;
  }

  size_t jj=row_num, row_head;

  while( jj<n_rows_ and row_lens_[jj]==0 ){
    jj++;
  }
  assert(jj<n_rows_);
  row_head = jj*max_rowcol_nnz_;
  return entries_[crs_order_[row_head]].rowidx;
}



template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::set_col_value(
  const size_t col_num, const size_t ent_num, const ValType value){

  if (!is_ccs_sorted_)
    sort_ccs();

  assert( col_num<n_cols_ );

  assert( ent_num< col_lens_[col_num] );
  entries_[ccs_order_[col_num*max_rowcol_nnz_+ent_num]].val = value;

  return;
}



template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::set_all_col_values(const size_t col_num, const ValType value){

  if (!is_ccs_sorted_)
    sort_ccs();

  assert( col_num<n_cols_ );

  for(size_t jj=0; jj<col_lens_[col_num]; jj++ ){
    entries_[ccs_order_[col_num*max_rowcol_nnz_+jj]].val = value;
  }

  return;
}


template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::set_all_col_values(const size_t col_num, const std::vector<ValType>& other){

  if (!is_ccs_sorted_)
    sort_ccs();

  assert( col_num<n_cols_ );

  assert( other.size()>=col_lens_[col_num]);

  for(size_t jj=0; jj<col_lens_[col_num]; jj++ ){
    entries_[ccs_order_[col_num*max_rowcol_nnz_+jj]].val = other[jj];
  }

  return;
}




template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::set_row_value(
  const size_t row_num, const size_t ent_num, const ValType value){

  if (!is_crs_sorted_)
    sort_crs();

  assert( row_num<n_rows_ );

  assert( ent_num< row_lens_[row_num] );
  entries_[crs_order_[row_num*max_rowcol_nnz_+ent_num]].val = value;
  
  return;
}


template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::set_all_row_values( const size_t row_num, const std::vector<ValType>& other){

  if (!is_crs_sorted_)
    sort_crs();

  assert( row_num<n_rows_ );

  assert( other.size()>=row_lens_[row_num]);

  for(size_t jj=0; jj<row_lens_[row_num]; jj++ ){
    entries_[crs_order_[row_num*max_rowcol_nnz_+jj]].val = other[jj];
  }
  
  return;
}


template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::set_all_row_values(const size_t row_num, const ValType value){

  if (!is_crs_sorted_)
    sort_crs();

  assert( row_num<n_rows_ );

  for(size_t jj=0; jj<row_lens_[row_num]; jj++){
    entries_[crs_order_[row_num*max_rowcol_nnz_+jj]].val = value;
  }
  
  return;
}





template <typename IdxType, typename ValType>
inline SparseMatrixEntry<IdxType,ValType>& SparseMatrix<IdxType, ValType>::get_col_entry(
  const size_t col_num, const size_t ent_num){

  if (!is_ccs_sorted_)
    sort_ccs();

  assert( col_num<n_cols_ );

  assert( ent_num< col_lens_[col_num] );
  return entries_[ccs_order_[col_num*max_rowcol_nnz_+ent_num]];
  
}


template <typename IdxType, typename ValType>
inline size_t SparseMatrix<IdxType, ValType>::get_col_entry_number(
  const size_t col_num, const size_t ent_num){

  if (!is_ccs_sorted_)
    sort_ccs();

  assert( col_num<n_cols_ );

  assert( ent_num< col_lens_[col_num] );
  return ccs_order_[col_num*max_rowcol_nnz_+ent_num];
}


template <typename IdxType, typename ValType>
inline size_t SparseMatrix<IdxType, ValType>::col_entry2crs_pos(const size_t col_num, const size_t col_ent_num){

  if (!is_ccs_sorted_)
    sort_ccs();

  if (!is_crs_sorted_)
    sort_crs();

  assert( col_num<n_cols_ );

  assert( col_ent_num< col_lens_[col_num] );

  return inv_crs_order_[ccs_order_[col_num*max_rowcol_nnz_+col_ent_num]];
}



// template <typename IdxType, typename ValType>
// inline SparseMatrixEntry<IdxType,ValType>& SparseMatrix<IdxType, ValType>::get_entry_ccs(const size_t ent_num){

//   if (!is_ccs_sorted_)
//     sort_ccs();

//   assert( ent_num<n_cols_*max_rowcol_nnz_ );



//   assert( col_num<n_cols_ );

//   assert( ent_num< col_lens_[col_num] );
//   return entries_[ccs_order_[col_num*max_rowcol_nnz_+ent_num]];
  
// }


template <typename IdxType, typename ValType>
inline SparseMatrixEntry<IdxType,ValType>& SparseMatrix<IdxType, ValType>::get_row_entry(
  const size_t row_num, const size_t ent_num){

  if (!is_crs_sorted_)
    sort_crs();

  assert( row_num<n_rows_ );

  //std::cout<<ent_num<<" "<<row_num<<" "<<row_lens_[row_num]<<std::endl;

  assert( ent_num<row_lens_[row_num] );
  return entries_[crs_order_[row_num*max_rowcol_nnz_+ent_num]];
  
}


template <typename IdxType, typename ValType>
inline size_t SparseMatrix<IdxType, ValType>::get_row_entry_number(
  const size_t row_num, const size_t ent_num){

  if (!is_crs_sorted_)
    sort_crs();

  assert( row_num<n_rows_ );

  //std::cout<<ent_num<<" "<<row_num<<" "<<row_lens_[row_num]<<std::endl;

  assert( ent_num<row_lens_[row_num] );
  return crs_order_[row_num*max_rowcol_nnz_+ent_num];
}




template <typename IdxType, typename ValType>
inline size_t SparseMatrix<IdxType, ValType>::row_entry2ccs_pos(
  const size_t row_num, const size_t row_ent_num){

  if (!is_crs_sorted_)
    sort_crs();

  if (!is_ccs_sorted_)
    sort_ccs();

  assert( row_num<n_rows_ );

  //std::cout<<ent_num<<" "<<row_num<<" "<<row_lens_[row_num]<<std::endl;

  assert( row_ent_num<row_lens_[row_num] );
  return inv_ccs_order_[crs_order_[row_num*max_rowcol_nnz_+row_ent_num]];
}


// Add in a new empty column.  Does not check for whether the column is new or whether it's being
// placed in the correct position (that's why it's private).
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::inject_col(const size_t col_num){
  assert(col_num<=n_cols_);
  assert(n_cols_<max_cols_);

  size_t ii, jj, col_head, col_end;

  //print_ccs();

  if (n_cols_>0){
    for( jj=n_cols_; jj>col_num; jj--){      // shift columns with larger index backward to create space for new column
      col_head = (jj-1)*max_rowcol_nnz_;
      col_end = col_head+col_lens_[jj-1];
      for( ii=col_head; ii<col_end; ii++){
        inv_ccs_order_[ccs_order_[ii]] += max_rowcol_nnz_;
      }
      if( col_end>col_head ){
        std::move_backward(ccs_order_.begin()+col_head,
          ccs_order_.begin()+col_end,ccs_order_.begin()+col_end+max_rowcol_nnz_);
        col_lens_[jj] = col_lens_[jj-1];
      }
    }
  }

  col_lens_[col_num] = 0;

  n_cols_++;

  //print_ccs();

  is_crs_sorted_ = false;

  return;
}






// Add in a new empty column.  Does not check for whether the column is new or whether it's being
// placed in the correct position (that's why it's private).
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::inject_row(const size_t row_num){
  assert(row_num<=n_rows_);
  assert(n_rows_<max_rows_);

  size_t ii, jj, row_head, row_end;

  //print_ccs();

  if (n_rows_>0){
    for( jj=n_rows_; jj>row_num; jj--){      // shift columns with larger index backward to create space for new column
      row_head = (jj-1)*max_rowcol_nnz_;
      row_end = row_head+row_lens_[jj-1];
      for( ii=row_head; ii<row_end; ii++){
        inv_crs_order_[crs_order_[ii]] += max_rowcol_nnz_;
      }
      if( row_end>row_head ){
        std::move_backward(crs_order_.begin()+row_head,
          crs_order_.begin()+row_end,crs_order_.begin()+row_end+max_rowcol_nnz_);
        row_lens_[jj] = row_lens_[jj-1];
      }
    }
  }

  row_lens_[row_num] = 0;

  n_rows_++;

  //print_ccs();

  is_ccs_sorted_ = false;

  return;
}







// Removes a column from the matrix.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::eject_col(const size_t col_num){
  assert(col_num<n_cols_);

  if (!is_ccs_sorted_)
    sort_ccs();

  if(col_num==n_cols_-1){
    clear_col(col_num);   // make sure the column is cleared first;
  }
  else{                 
    clear_col(col_num);   // make sure the column is cleared first;

    size_t ii, jj, col_head, col_end;

    for( jj=col_num+1; jj<n_cols_; jj++){      // shift columns with larger index forward to create space for new column
      col_head = (jj-1)*max_rowcol_nnz_;
      col_end = col_head+col_lens_[jj];
      for( ii=col_head; ii<col_end; ii++){
        inv_ccs_order_[ccs_order_[ii]] -= max_rowcol_nnz_;
      }
      std::swap_ranges(ccs_order_.begin()+col_head,
        ccs_order_.begin()+col_end,ccs_order_.begin()+jj*max_rowcol_nnz_);
      col_lens_[jj-1] = col_lens_[jj];
    }

    n_cols_--;
  }

  is_crs_sorted_ = false;

  return;
}






// Removes a row from the matrix.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::eject_row(const size_t row_num){
  assert(row_num<n_rows_);

  if (!is_crs_sorted_)
    sort_crs();

  if( row_num<n_rows_-1){
    clear_row(row_num);                        // make sure the row is cleared first;

    size_t ii, jj, row_head, row_end;

    for( jj=row_num+1; jj<n_rows_; jj++){      // shift row with larger index forward to create space for new row
      row_head = (jj-1)*max_rowcol_nnz_;
      row_end = row_head+row_lens_[jj];
      for( ii=row_head; ii<row_end; ii++){
        inv_crs_order_[crs_order_[ii]] -= max_rowcol_nnz_;
      }
      std::swap_ranges(crs_order_.begin()+row_head,
        crs_order_.begin()+row_end,crs_order_.begin()+jj*max_rowcol_nnz_);
      row_lens_[jj-1] = row_lens_[jj];
    }

    n_rows_--;
  }
  else{
    clear_row(row_num);
  }

  is_ccs_sorted_ = false;

  return;
}






// Assumes column col_num is empty and fills it with entries from the sparse vector
// other.  Does not check that the column index idx is unique or that it's being
// placed in the right position (that's why it's private).
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::fill_col(const size_t col_num, const IdxType idx, const SparseVector<IdxType, ValType>& other){
  assert(col_num<n_cols_);

  clear_col(col_num);

  assert( curr_size_+other.size()<=max_size_);
  assert( other.size()<=max_rowcol_nnz_);

  size_t col_head = col_num*max_rowcol_nnz_;
  size_t col_end = col_head + other.size();

  //print_ccs();

  for (size_t jj=0; jj<other.size(); jj++){
    entries_[curr_size_+jj].colidx = idx;
    entries_[curr_size_+jj].rowidx = other[jj].idx;
    entries_[curr_size_+jj].val = other[jj].val;
    //std::cout<<idx<<" "<<other[jj].idx<<" "<<other[jj].val<<std::endl;
  }

  col_lens_[col_num] = other.size();

  //std::cout<<curr_size_<<" "<< col_head<<" "<<col_end<<" "<<other.size()<<std::endl;

  std::iota(ccs_order_.begin()+col_head,ccs_order_.begin()+col_end,curr_size_);
  std::iota(inv_ccs_order_.begin()+curr_size_,inv_ccs_order_.begin()+curr_size_+other.size(),col_head);

  curr_size_ += other.size();

  is_crs_sorted_ = false;

  return;
}









template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::fill_row(const size_t row_num, const IdxType idx, const SparseVector<IdxType, ValType>& other){
  assert(row_num<n_rows_);

  clear_row(row_num);

  assert( curr_size_+other.size()<=max_size_);
  assert( other.size()<=max_rowcol_nnz_);

  size_t row_head = row_num*max_rowcol_nnz_;
  size_t row_end = row_head + other.size();

  //print_ccs();

  for (size_t jj=0; jj<other.size(); jj++){
    entries_[curr_size_+jj].rowidx = idx;
    entries_[curr_size_+jj].colidx = other[jj].idx;
    entries_[curr_size_+jj].val = other[jj].val;
    //std::cout<<idx<<" "<<other[jj].idx<<" "<<other[jj].val<<std::endl;
  }

  row_lens_[row_num] = other.size();

  //std::cout<<curr_size_<<" "<< col_head<<" "<<col_end<<" "<<other.size()<<std::endl;

  std::iota(crs_order_.begin()+row_head,crs_order_.begin()+row_end,curr_size_);
  std::iota(inv_crs_order_.begin()+curr_size_,inv_crs_order_.begin()+curr_size_+other.size(),row_head);

  curr_size_ += other.size();

  is_ccs_sorted_ = false;

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



// Clear the contents of a column without removing the column.  Does not preserve crs ordering.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::clear_col(const size_t col_num){
  assert(col_num<n_cols_);

  if( col_lens_[col_num]==0 )
    return;

  size_t col_head, col_end, new_loc, ccs_loc;
  std::vector<size_t>::iterator ccs_pos;

  col_head = col_num*max_rowcol_nnz_;
  col_end = col_head+col_lens_[col_num];
  ccs_pos = ccs_order_.begin() + col_head;

  for (size_t jj=0; jj < col_lens_[col_num]; jj++ ){      // swap iterators from end to fill unfilled old entries
    assert(curr_size_>jj);
    if ( *(ccs_pos+jj)!=curr_size_-jj-1 ){
      swap_entries(*(ccs_pos+jj),curr_size_-jj-1);
    }
  }

  curr_size_ -= col_lens_[col_num];

  col_lens_[col_num] = 0;

  while( n_cols_>0 and col_lens_[n_cols_-1]==0 ){                 // last column aren't allowed to be empty unless matrix is empty.
    n_cols_--;
  }

  is_crs_sorted_ = false;

  return;
}





// Clear the contents of a column without removing the column.  Does not preserve crs ordering.  Private
// because I can't think of a reason to use this without also using fill_coll (which is private).
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::clear_row(const size_t row_num){
  assert(row_num<n_rows_);

  if( row_lens_[row_num]==0 )
    return;

  size_t row_head, row_end, new_loc, crs_loc;
  std::vector<size_t>::iterator crs_pos;

  row_head = row_num*max_rowcol_nnz_;
  row_end = row_head+row_lens_[row_num];
  crs_pos = crs_order_.begin() + row_head;

  for (size_t jj=0; jj < row_lens_[row_num]; jj++ ){      // swap iterators from end to fill unfilled old entries
    assert(curr_size_>jj);
    if ( *(crs_pos+jj)!=curr_size_-jj-1 ){
      swap_entries(*(crs_pos+jj),curr_size_-jj-1);
    }
  }

  curr_size_ -= row_lens_[row_num];

  row_lens_[row_num] = 0;

  while( n_rows_>0 and row_lens_[n_rows_-1]==0 ){                 // last row isn't allowed to be empty unless matrix is empty
    n_rows_--;
  }

  is_ccs_sorted_ = false;

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

  col_id = std::lower_bound(counter_.begin(), counter_.begin()+n_cols_, n_cols_,
    [&](size_t ii, size_t jj) { return col_idx(ii) < col_idx(jj); });
 
  return *col_id;
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

  row_id = std::lower_bound(counter_.begin(), counter_.begin()+n_rows_, n_rows_,
    [&](size_t ii, size_t jj) { return row_idx(ii) < row_idx(jj); });
 
  return *row_id;
}






// locate the position in the ccs_order_ of entry with rowidx and colidx.  If no 
// such entry is found return first position with entry that compares greater
// than (rowidx,colidx) in CCS order.  Throws an error if the entry is new and
// would belong to a column that already has max_rowcol_nnz_.
template <typename IdxType, typename ValType>
inline size_t SparseMatrix<IdxType, ValType>::locate_entry_ccs(const IdxType rowidx, const IdxType colidx){

  if (!is_ccs_sorted_)
    sort_ccs();

  size_t col_head, col_end, col_num = locate_col(colidx);
  std::vector<size_t>::iterator ccs_pos;

  if( col_num == n_cols_ )
    return n_cols_*max_rowcol_nnz_;

  col_head = col_num*max_rowcol_nnz_;
  col_end = col_head + col_lens_[col_num];

  if ( colidx == entries_[ccs_order_[col_head]].colidx ){
    entries_[curr_size_].rowidx = rowidx;
    ccs_pos = std::lower_bound(ccs_order_.begin()+col_head, ccs_order_.begin()+col_end, curr_size_,
      [&](size_t ii, size_t jj) { return entries_[ii].rowidx < entries_[jj].rowidx; });
    if( ccs_pos == ccs_order_.begin()+col_end ){
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

  size_t row_head, row_end, row_num = locate_row(rowidx);
  std::vector<size_t>::iterator crs_pos;

  if( row_num == n_rows_ )
    return n_rows_*max_rowcol_nnz_;

  row_head = row_num*max_rowcol_nnz_;
  row_end = row_head+row_lens_[row_num];

  if ( rowidx == entries_[crs_order_[row_head]].rowidx ){
    entries_[curr_size_].colidx = colidx;
    crs_pos = std::lower_bound(crs_order_.begin()+row_head, crs_order_.begin()+row_end, curr_size_,
      [&](size_t ii, size_t jj) { return entries_[ii].colidx < entries_[jj].colidx; });
    if( crs_pos == crs_order_.begin()+row_end ){
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

  size_t col_num;

  if (!is_ccs_sorted_)
    sort_ccs();

  if (curr_size_==0){                    // is the matrix empty?

    col_num = 0;

    assert(other.size()>0);  // last column can't be empty

    inject_col(col_num);
    fill_col(col_num,idx,other);

    size_t row_head;

    for(size_t jj=0; jj<other.size(); jj++ ){          // because it's the first column we preserve row ordering too
      row_head = jj*max_rowcol_nnz_;
      row_lens_[jj] = 1;
      crs_order_[row_head] = jj;
      inv_crs_order_[jj] = row_head;
    }
    n_rows_ = other.size();
    is_crs_sorted_ = true;

    return;
  }

  is_crs_sorted_ = false;

  size_t col_head, col_end;

  if ( idx > entries_[ccs_order_[(n_cols_-1)*max_rowcol_nnz_]].colidx ){                // will this be the largest column index
    assert(other.size()>0); 
    col_num = n_cols_;                           // last column can't be empty
    inject_col(col_num); 
  }
  else{                                                              // this won't be the largest column index

    col_num = locate_col(idx);
    while( col_lens_[col_num]==0 ){
      col_num++;
    }
    assert(col_num<n_cols_);

    // std::cout<<col_num<<std::endl;

    col_head = col_num*max_rowcol_nnz_;
    col_end = col_head + col_lens_[col_num];

    assert(col_head<col_end);
    assert(col_lens_[col_num]>0);

    if (entries_[ccs_order_[col_head]].colidx == idx){                 // this replaces an existing column
      // std::cout<<col_num<<" "<<n_cols_<<std::endl;
      // print_ccs();
      clear_col(col_num);
      if(col_num>=n_cols_){
        col_num=n_cols_;
        inject_col(col_num);
      }
      // print_ccs();
      // std::cout<<curr_size_<<" "<<n_cols_<<std::endl;
    }
    else if ( col_num>0 and col_lens_[col_num-1]==0 ){                          // there's already an empty column in the right position
      //std::cout<<col_num<<col_lens_[col_num-1]<<std::endl;
      col_num--;
    }           
    else{                                                         // we need to insert a new column in the right position
      //std::cout<<col_num<<" "<<n_cols_<<std::endl;
      inject_col(col_num);
    }

    // std::cout<<"out"<<std::endl;
    // std::cout<<std::endl;
  }
  fill_col(col_num,idx,other);

  return;
}





// Add a sparse vector as a column to a sparse matrix.
// If the matrix already has that column it is replaced.
// This leaves the matrix CCS ordered, but destroys CRS ordering.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::set_row(const SparseVector<IdxType, ValType>& other, const IdxType idx){

  size_t row_num;

  if (!is_crs_sorted_)
    sort_crs();

  if (curr_size_==0){                    // is the matrix empty?

    row_num = 0;

    assert(other.size()>0);  // last column can't be empty

    inject_col(row_num);
    fill_col(row_num,idx,other);

    size_t col_head;

    for(size_t jj=0; jj<other.size(); jj++ ){          // because it's the first column we preserve row ordering too
      col_head = jj*max_rowcol_nnz_;
      col_lens_[jj] = 1;
      ccs_order_[col_head] = jj;
      inv_ccs_order_[jj] = col_head;
    }
    n_cols_ = other.size();
    is_ccs_sorted_ = true;

    return;
  }

  is_ccs_sorted_ = false;

  size_t row_head, row_end;

  if ( idx > entries_[crs_order_[(n_rows_-1)*max_rowcol_nnz_]].rowidx ){                // will this be the largest column index
    assert(other.size()>0); 
    row_num = n_rows_;                           // last column can't be empty
    inject_row(row_num); 
  }
  else{                                                              // this won't be the largest column index

    row_num = locate_row(idx);
    while( row_lens_[row_num]==0 ){
      row_num++;
    }
    assert(row_num<n_rows_);

    // std::cout<<col_num<<std::endl;

    row_head = row_num*max_rowcol_nnz_;
    row_end = row_head + row_lens_[row_num];

    assert(row_head<row_end);
    assert(row_lens_[row_num]>0);

    if (entries_[crs_order_[row_head]].rowidx == idx){                 // this replaces an existing column
      // std::cout<<col_num<<" "<<n_cols_<<std::endl;
      // print_ccs();
      clear_row(row_num);
      if(row_num>=n_rows_){
        row_num=n_rows_;
        inject_row(row_num);
      }
      // print_ccs();
      // std::cout<<curr_size_<<" "<<n_cols_<<std::endl;
    }
    else if ( row_num>0 and row_lens_[row_num-1]==0 ){                          // there's already an empty column in the right position
      //std::cout<<col_num<<col_lens_[col_num-1]<<std::endl;
      row_num--;
    }           
    else{                                                         // we need to insert a new column in the right position
      //std::cout<<col_num<<" "<<n_cols_<<std::endl;
      inject_row(row_num);
    }

    // std::cout<<"out"<<std::endl;
    // std::cout<<std::endl;
  }
  fill_row(row_num,idx,other);

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
  row_lens_[0] = 1;
  for (size_t jj=1; jj<curr_size_;jj++){
    if ( entries_[crs_order_[jj]].rowidx > entries_[crs_order_[jj-1]].rowidx ){
      n_rows_++; 
      row_lens_[n_rows_-1]=1;
    }
    else{
      row_lens_[n_rows_-1]++;
    }
  }
  
  size_t row_end, row_head;

  if( n_rows_>0 ){
    row_end = curr_size_;
    for ( size_t jj=n_rows_-1;jj>0;jj--){
      row_head = jj*max_rowcol_nnz_;
      std::move_backward(crs_order_.begin()+row_end-row_lens_[jj],crs_order_.begin()+row_end,crs_order_.begin()+row_head+row_lens_[jj]);
      row_end -= row_lens_[jj];
      //std::cout<<jj<<"\t"<<row_ends_[jj]<<std::endl;
    }
  }

  for (size_t jj=0;jj<n_rows_;jj++){
    row_head = jj*max_rowcol_nnz_;
    row_end = row_head + row_lens_[jj];
    for (size_t ii=row_head;ii<row_end;ii++){
      inv_crs_order_[crs_order_[ii]] = ii;
    }
  }

  is_crs_sorted_=true;

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
  col_lens_[0] = 1;
  for (size_t jj=1; jj<curr_size_;jj++){
    if ( entries_[ccs_order_[jj]].colidx > entries_[ccs_order_[jj-1]].colidx ){
      n_cols_++; 
      col_lens_[n_cols_-1]=1;
    }
    else{
      col_lens_[n_cols_-1]++;
    }
  }

  size_t col_end, col_head;

  if( n_cols_>0 ){
    col_end = curr_size_;
    for ( size_t jj=n_cols_-1;jj>0;jj--){
      col_head = jj*max_rowcol_nnz_;
      std::move_backward(ccs_order_.begin()+col_end-col_lens_[jj],ccs_order_.begin()+col_end,ccs_order_.begin()+col_head+col_lens_[jj]);
      col_end -= col_lens_[jj];
    }
  }

  for (size_t jj=0;jj<n_cols_;jj++){
    col_head = jj*max_rowcol_nnz_;
    col_end = col_head + col_lens_[jj];
    for (size_t ii=col_head;ii<col_end;ii++){
      inv_ccs_order_[ccs_order_[ii]] = ii;
    }
  }

  is_ccs_sorted_=true;

  return;
}




template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::scale_col(const size_t col_num, const ValType a){
  assert(col_num<n_cols_);

  if(!is_ccs_sorted_)
    sort_ccs();

  size_t col_head = col_num*max_rowcol_nnz_;
  size_t col_end = col_head+col_lens_[col_num];

  for (size_t jj=col_head; jj<col_end; jj++){
    entries_[ccs_order_[jj]].val *= a;
  }

  return;
}



// Copy entries of column idx of a sparse matrix
// into a sparse vector.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::get_col(const size_t col_num, SparseVector<IdxType, ValType>& y){
  assert(col_num<n_cols_);

  if(!is_ccs_sorted_)
    sort_ccs();

  assert( y.max_size_ >= col_lens_[col_num] );

  size_t col_head = col_num*max_rowcol_nnz_;
  size_t col_end = col_head+col_lens_[col_num];

  y.clear();

  for (size_t jj=col_head; jj<col_end; jj++){
    y.set_entry(entries_[ccs_order_[jj]].rowidx,entries_[ccs_order_[jj]].val);
  }

  assert(y.size()==col_lens_[col_num]);

  return;
}


template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::get_all_col_values(const size_t col_num, std::vector<ValType>& other){
  assert(col_num<n_cols_);

  if(!is_ccs_sorted_)
    sort_ccs();

  assert( other.size() >= col_lens_[col_num] );

  size_t col_head = col_num*max_rowcol_nnz_;
  size_t col_end = col_head+col_lens_[col_num];

  for (size_t jj=col_head; jj<col_end; jj++){
    other[jj-col_head] = entries_[ccs_order_[jj]].val;
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

  assert( y.max_size_ >= row_lens_[row_num] );

  size_t row_head = row_num*max_rowcol_nnz_;
  size_t row_end = row_head + row_lens_[row_num];

  y.clear();

  for (size_t jj=row_head; jj<row_end; jj++){
    y.set_entry(entries_[crs_order_[jj]].colidx,entries_[crs_order_[jj]].val);
  }

  assert(y.size()==row_lens_[row_num]);

  return;
}


template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::get_all_row_values(const size_t row_num, std::vector<ValType>& other){
  assert(row_num<n_rows_);

  if(!is_crs_sorted_)
    sort_crs();

  assert( other.size() >= row_lens_[row_num] );

  size_t row_head = row_num*max_rowcol_nnz_;
  size_t row_end = row_head + row_lens_[row_num];

  for (size_t jj=row_head; jj<row_end; jj++){
    other[jj-row_head] = entries_[crs_order_[jj]].val;
  }

  return;
}


template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::row_sums(std::vector<ValType>& sums){
  assert(sums.size()>=n_rows_);

  if(!is_crs_sorted_)
    sort_crs();

  size_t row_head, row_end;

  for( size_t ii = 0; ii<n_rows_; ii++){
    if( row_lens_[ii]>0 ){
      row_head = ii*max_rowcol_nnz_;
      row_end = row_head + row_lens_[ii];
      sums[ii] = entries_[crs_order_[row_head]].val;
      for( size_t jj = row_head+1; jj<row_end; jj++){
        sums[ii] += entries_[crs_order_[jj]].val;
      }
    }
    else{
      sums[ii] = 0;
    }
  }

  return;
}


template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::row_sums(SparseVector<IdxType,ValType>& sums){
  assert(sums.capacity()>=n_rows_);

  if(!is_crs_sorted_)
    sort_crs();

  size_t row_head, row_end;
  IdxType idx;
  ValType val;

  sums.clear();

  for( size_t ii = 0; ii<n_rows_; ii++){
    if( row_lens_[ii]>0 ){
      row_head = ii*max_rowcol_nnz_;
      row_end = row_head + row_lens_[ii];
      idx = row_idx(ii);
      val = entries_[crs_order_[row_head]].val;
      for( size_t jj = row_head+1; jj<row_end; jj++){
        val+=entries_[crs_order_[jj]].val;
      }
      sums.set_entry(idx, val);
    }
  }

  return;
}


template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::col_sums(std::vector<ValType>& sums){
  assert(sums.capacity()>=n_cols_);

  sums.resize(n_cols_);

  if(!is_ccs_sorted_)
    sort_ccs();

  size_t  col_head, col_end;

  for( size_t jj = 0; jj<n_cols_; jj++){
    if( col_lens_[jj]>0 ){
      col_head = jj*max_rowcol_nnz_;
      col_end = col_head + col_lens_[jj];
      sums[jj] = entries_[ccs_order_[col_head]].val;
      for( size_t ii = col_head+1; ii<col_end; ii++){
        sums[jj]+=entries_[ccs_order_[ii]].val;
      }
    }
    else{
      sums[jj] = 0;
    }
  }

  return;
}

template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::col_norms(std::vector<ValType>& norms){

  // std::cout<<norms.capacity()<<" "<<n_cols_<<std::endl;

  assert(norms.capacity()>=n_cols_);

  if(!is_ccs_sorted_)
    sort_ccs();

  SparseVectorEntry<IdxType,ValType> e;

  size_t  col_head, col_end;

  for( size_t jj = 0; jj<n_cols_; jj++){
    if( col_lens_[jj]>0 ){
      col_head = jj*max_rowcol_nnz_;
      col_end = col_head + col_lens_[jj];
      norms[jj] = abs(entries_[ccs_order_[col_head]].val);
      for( size_t ii = col_head+1; ii<col_end; ii++){
        norms[jj] += abs(entries_[ccs_order_[ii]].val);
      }
    }
    else{
      norms[jj] = 0;
    }
  }

  return;
}







template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::row_norms(std::vector<ValType>& norms){

  // std::cout<<norms.capacity()<<" "<<n_cols_<<std::endl;

  assert(norms.capacity()>=n_rows_);

  if(!is_crs_sorted_)
    sort_crs();

  SparseVectorEntry<IdxType,ValType> e;

  size_t  row_head, row_end;

  for( size_t jj = 0; jj<n_rows_; jj++){
    if( row_lens_[jj]>0 ){
      row_head = jj*max_rowcol_nnz_;
      row_end = row_head + row_lens_[jj];
      norms[jj] = abs(entries_[crs_order_[row_head]].val);
      for( size_t ii = row_head+1; ii<row_end; ii++){
        norms[jj] += abs(entries_[crs_order_[ii]].val);
      }
    }
    else{
      norms[jj] = 0;
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
    row_lens_[jj] = other.row_lens_[jj];
  }
  for (size_t jj = 0; jj<other.n_cols_; jj++){
    col_lens_[jj] = other.col_lens_[jj];
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

  size_t ccs_loc, col_head, col_end;
  std::cout << "entries: "<<curr_size_ << "  columns: " << n_cols_ << std::endl;
  for( size_t jj = 0; jj<n_cols_; jj++){
    col_head = jj*max_rowcol_nnz_;
    col_end = col_head+col_lens_[jj];
    if(col_head==col_end){
      std::cout<<"column "<<jj<<" is empty"<<std::endl;
    }
    for (size_t ii = col_head; ii < col_end; ii++) {
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

  size_t crs_loc, row_head, row_end;
  std::cout << "entries: "<<curr_size_ << "  rows: "<< n_rows_ << std::endl;
  for( size_t jj = 0; jj<n_rows_; jj++){
    row_head = jj*max_rowcol_nnz_;
    row_end = row_head+row_lens_[jj];
    if(row_head==row_end){
      std::cout<<"row "<<jj<<" is empty"<<std::endl;
    }
    for (size_t ii = row_head; ii < row_end; ii++) {
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

  assert(max_nz_col_entries<=B.rowcol_capacity());
  assert(x.size() <= B.max_cols());

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
    // single_column_add.print();
    // std::cout<< jj<<" "<<x.size()<<" "<<single_column_add.size()<<std::endl;
    B.set_col(single_column_add,x[jj].idx);
    // B.print_ccs();
  }

  // std::cout << "pass" << std::endl;

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
  // std::valarray<size_t> inds_;
  std::vector<double> xabs_;
  std::vector<bool> pres_;
  
  // Random number generator.
  RNG* gen_;
  std::uniform_real_distribution<> uu_;
  
  // Do not allow copying or assignment for compressors.
  Compressor<IdxType, ValType, RNG>(Compressor<IdxType, ValType, RNG> &);
  Compressor<IdxType, ValType, RNG>& operator= (Compressor<IdxType, ValType, RNG> &);
  
  // Helper function: compress the internal 
  // vector of moduli.
  // inline size_t preserve_xabs(const size_t target_nnz);
  // inline void compress_xabs_sys(size_t target_nnz);
  // inline void compress_xabs_dmc(size_t target_nnz);

  inline const size_t preserve_xabs_(SparseVector<IdxType, ValType> &x, const size_t target_nnz);
  inline const size_t preserve_xabs_(SparseMatrix<IdxType, ValType> &A, const size_t target_nnz);

  inline void sample_xabs_(SparseVector<IdxType, ValType> &x, const size_t target_nnz);
  inline void sample_xabs_(SparseMatrix<IdxType, ValType> &A, const size_t target_nnz);

  inline size_t setup(SparseVector<IdxType, ValType> &x);
  inline size_t setup(SparseMatrix<IdxType, ValType> &A);


public:


  // Constructor based on maximum size of the
  // modulus vector and a random seed.
  inline Compressor<IdxType, ValType, RNG>(size_t max_size, RNG& generator) : 
    xabs_(max_size), pres_(max_size) {
    // Set up the pseudorandom number generator.
    gen_ = &generator;
    uu_ = std::uniform_real_distribution<>(0,1);
  }

  Compressor();

  inline void allocate(size_t max_size, RNG& generator){

    xabs_ = std::vector<double>(max_size);
    pres_ = std::vector<bool>(max_size);

    gen_ = &generator;
    uu_ = std::uniform_real_distribution<>(0,1);

    return;
  }
  
  // Compress a SparseVector using the stored
  // temp vector and pseudorandom number generator.
  inline void compress(SparseVector<IdxType, ValType> &x, const size_t target_nnz);
  inline void compress(SparseMatrix<IdxType, ValType> &A, const size_t target_nnz);

  inline void sample(SparseVector<IdxType, ValType> &x, const size_t ncopies);
  inline void sample(SparseMatrix<IdxType, ValType> &A, const size_t ncopies);

  // Return a mask of the sparse vector indicating which entries should
  // be preserved exactly in a compression.
  inline const size_t preserve(SparseVector<IdxType, ValType> &x, const size_t target_nnz);
  inline const size_t preserve(SparseMatrix<IdxType, ValType> &A, const size_t target_nnz);

 
};



template <class RNG>
inline size_t sample_sys(std::vector<double>& xabs_,const size_t target_nnz, RNG* gen_);

template <class RNG>
inline size_t sample_piv(std::vector<double>& xabs_, RNG* gen_);

inline size_t adjust(std::vector<double>& xabs_,const size_t);



//---------------------------------------------------------
// Sparse vector compression helper class and routines
// Implementation
//---------------------------------------------------------

using std::abs;


template <typename IdxType, typename ValType, class RNG>
inline Compressor<IdxType, ValType, RNG>::Compressor(){}



template <typename IdxType, typename ValType, class RNG>
inline size_t Compressor<IdxType, ValType, RNG>::setup(SparseVector<IdxType, ValType> &x) {

  xabs_.resize(x.size());
  pres_.resize(x.size());

  double Tol = 1e-9;
  size_t nnz=0;

  for(size_t jj=0;jj<x.size();jj++){
    xabs_[jj] = abs(x[jj].val);
    if( xabs_[jj]>Tol ){
      nnz++;
    }
    else{
      x.set_value(jj,0);
      xabs_[jj]=0;
    }
  }

  return nnz;
}




template <typename IdxType, typename ValType, class RNG>
inline size_t Compressor<IdxType, ValType, RNG>::setup(SparseMatrix<IdxType, ValType> &A) {

  xabs_.resize(A.size());
  pres_.resize(A.size());

  double Tol = 1e-9;
  size_t nnz=0;

  for(size_t jj=0;jj<A.size();jj++){
    xabs_[jj] = abs(A[jj].val);
    if( xabs_[jj]>Tol ){
      nnz++;
    }
    else{
      A.set_value(jj,0);
      xabs_[jj]=0;
    }
  }

  return nnz;
}




// Compress a SparseVector using the stored
// temp vector and pseudorandom number generator.
template <typename IdxType, typename ValType, class RNG>
inline void Compressor<IdxType, ValType, RNG>::compress(SparseVector<IdxType, ValType> &x, const size_t target_nnz) {

  size_t nnz = setup(x);

  if( nnz <= target_nnz ){
    std::fill(pres_.begin(), pres_.end(), true);
    std::fill(xabs_.begin(), xabs_.end(), 0);
    // x.remove_zeros();
    return;
  }

  // std::cout << nnz << " " << target_nnz << std::endl;

  // std::cout << xabs_.sum() << std::endl;

  size_t npres = preserve_xabs_(x,target_nnz);

  // std::cout<< nnz<< " " << npres << std::endl;

  if( npres == target_nnz){
    // x.remove_zeros();
    return;
  }

  size_t nnz_sample = target_nnz-npres;

  sample_xabs_(x,nnz_sample);

  for( size_t jj = 0; jj < x.size(); jj++){
    // Find the corresponding member of x and
    // set its modulus according to the modulus
    // of xabs.
    if( xabs_[jj]>0 ){
      x.set_value(jj,(x[jj].val / abs(x[jj].val))*xabs_[jj]);
    }
    if( pres_[jj]==false and xabs_[jj]==0 ){
      x.set_value(jj,0);
    }
  }
  // Remove the entries set to zero.
  // x.remove_zeros();
  // The vector is now compressed.
  return;
}






// Compress a SparseVector using the stored
// temp vector and pseudorandom number generator.
template <typename IdxType, typename ValType, class RNG>
inline void Compressor<IdxType, ValType, RNG>::compress(SparseMatrix<IdxType, ValType> &A, const size_t target_nnz) {

  size_t nnz = setup(A);

  if( nnz <= target_nnz ){
    std::fill(pres_.begin(), pres_.end(), true);
    std::fill(xabs_.begin(), xabs_.end(), 0);
    // A.remove_zeros();
    return;
  }

  // std::cout << nnz << " " << target_nnz << std::endl;

  // std::cout << xabs_.sum() << std::endl;

  size_t npres = preserve_xabs_(A,target_nnz);

  // std::cout<< nnz<< " " << npres << std::endl;

  if( npres == target_nnz){
    // A.remove_zeros();
    return;
  }

  size_t nnz_sample = target_nnz-npres;

  sample_xabs_(A,nnz_sample);

  for( size_t jj = 0; jj < A.size(); jj++){
    // Find the corresponding member of x and
    // set its modulus according to the modulus
    // of xabs.
    if( xabs_[jj]>0 ){
      A.set_value(jj,(A[jj].val / abs(A[jj].val))*xabs_[jj]);
    }
    if( pres_[jj]==false and xabs_[jj]==0 ){
      A.set_value(jj,0);
    }
  }
  // Remove the entries set to zero.
  // A.remove_zeros();
  // The vector is now compressed.
  return;
}










template <typename IdxType, typename ValType, class RNG>
inline void Compressor<IdxType, ValType, RNG>::sample(SparseVector<IdxType, ValType> &x, const size_t ncopies) {

  setup(x);

  sample_xabs_(x,ncopies);

  for( size_t jj = 0; jj < x.size(); jj++ ){
    x.set_value(jj,(x[jj].val / abs(x[jj].val))*xabs_[jj]);
  }
  
  return;
}








template <typename IdxType, typename ValType, class RNG>
inline void Compressor<IdxType, ValType, RNG>::sample(SparseMatrix<IdxType, ValType> &A, const size_t ncopies) {

  setup(A);

  sample_xabs_(A,ncopies);

  for( size_t jj = 0; jj < A.size(); jj++ ){
    A.set_value(jj,(A[jj].val / abs(A[jj].val))*xabs_[jj]);
  }
  
  return;
}







template <typename IdxType, typename ValType, class RNG>
inline void Compressor<IdxType, ValType, RNG>::sample_xabs_(SparseVector<IdxType, ValType> &x, const size_t ncopies) {

  double xabs_sum = std::accumulate(xabs_.begin(), xabs_.end(), 0.0);

  assert(xabs_sum>0);

  // std::cout << xabs_.sum() << " " << ncopies << std::endl;

  // xabs_ *= (double)ncopies/xabs_sum;
  std::transform(xabs_.begin(), xabs_.end(), xabs_.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, (double)ncopies/xabs_sum));

  // std::cout << xabs_.sum() << " " << xabs_.size() << " " << x.size() << std::endl;

  std::vector<double> floor_vec(x.size());
  std::vector<double> select_vec(x.size());

  for( size_t jj = 0; jj< x.size(); jj++ ){
    floor_vec[jj] = floor(xabs_[jj]);
  }

  std::transform(xabs_.begin(),xabs_.end(),floor_vec.begin(),[](double &c){ return floor(c); });

  // std::cout << xabs_.sum() << " " << floor_vec.sum() << std::endl;

  std::transform(xabs_.begin(),xabs_.end(),floor_vec.begin(), select_vec.begin(),[](double &a, double &b){ return a-b; });

  // select_vec = xabs_-floor_vec;

  size_t nfloor = (size_t)std::accumulate(floor_vec.begin(), floor_vec.end(),decltype(floor_vec)::value_type(0));

  // std::cout << xabs_.sum() << " " << select_vec.sum() << std::endl;

  // Resample the remaining entries.
  sample_piv(select_vec, gen_);
  //resample_sys(xabs_resample,target_nnz-nnz_large, gen_);
  //xabs_resample *= xabs_sum/(double)(target_nnz-nnz_large);

  // compress_xabs_sys(target_nnz-nnz_large);
  // Translate the compression of the moduli vector to
  // a compression of the input vector. For each entry
  // of the compressed xabs,
  for( size_t jj = 0; jj < x.size(); jj++){
    xabs_[jj] = (xabs_sum/(double)ncopies)*(floor_vec[jj]+select_vec[jj]);
  }
  
  return;
}







template <typename IdxType, typename ValType, class RNG>
inline void Compressor<IdxType, ValType, RNG>::sample_xabs_(SparseMatrix<IdxType, ValType> &A, const size_t ncopies) {

  double xabs_sum = std::accumulate(xabs_.begin(), xabs_.end(), 0.0);

  assert(xabs_sum>0);

  // std::cout << xabs_.sum() << " " << ncopies << std::endl;

  // xabs_ *= (double)ncopies/xabs_sum;
  std::transform(xabs_.begin(), xabs_.end(), xabs_.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, (double)ncopies/xabs_sum));

  // std::cout << xabs_.sum() << " " << xabs_.size() << " " << x.size() << std::endl;

  std::vector<double> floor_vec(A.size());
  std::vector<double> select_vec(A.size());

  for( size_t jj = 0; jj< A.size(); jj++ ){
    floor_vec[jj] = floor(xabs_[jj]);
  }

  std::transform(xabs_.begin(),xabs_.end(),floor_vec.begin(),[](double &c){ return floor(c); });

  // std::cout << xabs_.sum() << " " << floor_vec.sum() << std::endl;

  std::transform(xabs_.begin(),xabs_.end(),floor_vec.begin(), select_vec.begin(),[](double &a, double &b){ return a-b; });

  // select_vec = xabs_-floor_vec;

  size_t nfloor = (size_t)std::accumulate(floor_vec.begin(), floor_vec.end(),decltype(floor_vec)::value_type(0));

  // std::cout << xabs_.sum() << " " << select_vec.sum() << std::endl;

  // Resample the remaining entries.
  sample_piv(select_vec, gen_);
  //resample_sys(xabs_resample,target_nnz-nnz_large, gen_);
  //xabs_resample *= xabs_sum/(double)(target_nnz-nnz_large);

  // compress_xabs_sys(target_nnz-nnz_large);
  // Translate the compression of the moduli vector to
  // a compression of the input vector. For each entry
  // of the compressed xabs,
  for( size_t jj = 0; jj < A.size(); jj++){
    xabs_[jj] = (xabs_sum/(double)ncopies)*(floor_vec[jj]+select_vec[jj]);
  }
  
  return;
}








template <typename IdxType, typename ValType, class RNG>
inline const size_t Compressor<IdxType, ValType, RNG>::preserve(SparseVector<IdxType, ValType> &x, const size_t target_nnz) {

  size_t nnz = setup(x);

  if( nnz <= target_nnz ){
    std::fill(pres_.begin(), pres_.end(), true);
    std::fill(xabs_.begin(), xabs_.end(), 0);
    return nnz;
  }

  return preserve_xabs_(x);
}






template <typename IdxType, typename ValType, class RNG>
inline const size_t Compressor<IdxType, ValType, RNG>::preserve(SparseMatrix<IdxType, ValType> &A, const size_t target_nnz) {

  size_t nnz = setup(A);

  if( nnz <= target_nnz ){
    std::fill(pres_.begin(), pres_.end(), true);
    std::fill(xabs_.begin(), xabs_.end(), 0);
    return nnz;
  }

  return preserve_xabs_(A);
}







// Return the indices of the sparse vector that should
// be preserved in a compression.
template <typename IdxType, typename ValType, class RNG>
inline const size_t Compressor<IdxType, ValType, RNG>::preserve_xabs_(SparseVector<IdxType, ValType> &x, const size_t target_nnz) {

  // if x already has less than target_nnz non-zero entries this 
  // is a do nothing routine.


  double xabs_sum, scale;
  size_t npos, npres;

  // std::cout << xabs_.sum() <<std::endl;

  std::fill(pres_.begin(), pres_.end(), false);
  bool done = false;
  npos = 0;
  xabs_sum = std::accumulate(xabs_.begin(), xabs_.end(), 0.0);
  while( !done and target_nnz>npos and xabs_sum>0){
    scale = xabs_sum/(target_nnz-npos);
    done = true;
    xabs_sum = 0;
    npos = 0;
    npres = 0;
    for( size_t kk=0; kk<x.size(); kk++ ){
      if( xabs_[kk]>=scale ){
        done = false;
        pres_[kk] = true;
        xabs_[kk] = 0;
      }
      if( pres_[kk]==true ){
        npos++;
      }
      if( xabs_[kk]==0 ){
        npres++;
      }
      xabs_sum += xabs_[kk];
    }
  }

  assert( npos<=target_nnz );

  // std::cout<<"here: "<<npos<<std::endl;

  return npres;
}










// Return the indices of the sparse vector that should
// be preserved in a compression.
template <typename IdxType, typename ValType, class RNG>
inline const size_t Compressor<IdxType, ValType, RNG>::preserve_xabs_(SparseMatrix<IdxType, ValType> &A, const size_t target_nnz) {

  // if x already has less than target_nnz non-zero entries this 
  // is a do nothing routine.


  double xabs_sum, scale;
  size_t npos, npres;

  // std::cout << xabs_.sum() <<std::endl;

  std::fill(pres_.begin(), pres_.end(), false);
  bool done = false;
  npos = 0;
  xabs_sum = std::accumulate(xabs_.begin(), xabs_.end(), 0.0);
  while( !done and target_nnz>npos and xabs_sum>0){
    scale = xabs_sum/(target_nnz-npos);
    done = true;
    xabs_sum = 0;
    npos = 0;
    npres = 0;
    for( size_t kk=0; kk<A.size(); kk++ ){
      if( xabs_[kk]>=scale ){
        done = false;
        pres_[kk] = true;
        xabs_[kk] = 0;
      }
      if( pres_[kk]==true ){
        npos++;
      }
      if( xabs_[kk]==0 ){
        npres++;
      }
      xabs_sum += xabs_[kk];
    }
  }

  assert( npos<=target_nnz );

  // std::cout<<"here: "<<npos<<std::endl;

  return npres;
}












template <class RNG>
inline size_t sample_sys(std::vector<double>& xabs_,const  size_t target_nnz, RNG* gen_) {

  double xabs_sum;

  xabs_sum = std::accumulate(xabs_.begin(), xabs_.end(), 0.0);

  assert(xabs_sum==(double)target_nnz);

  assert(*std::max_element(xabs_.begin(),xabs_.end())<=1.0);

  std::uniform_real_distribution<> uu_=std::uniform_real_distribution<>(0,1);

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





// template <class RNG>
// inline void budget_piv(std::valarray<double>& xabs_, const size_t m, RNG* gen_) {

//   double xabs_sum, scale;

//   std::valarray<double> floor_vec(xabs_.size());

//   xabs_sum = xabs_.sum();

//   scale = xabs_sum/m;
//   xabs_ /= scale;
//   floor_vec = budgets.apply([](double c)->double {return floor(c);});
//   xabs_ -= floor_vec;
//   sample_piv(xabs_, gen_);
//   xabs_ += floor_vec;

//   return;

// }




template <class RNG>
inline size_t sample_piv(std::vector<double>& xabs_, RNG* gen_) {

  double xabs_sum = std::accumulate(xabs_.begin(), xabs_.end(), 0.0);

  

 

  if (xabs_sum==0){
    return 0;
  }


  if(abs(xabs_sum-round(xabs_sum))>1e-9){
    std::cout<<xabs_sum<<" "<<abs(xabs_sum-round(xabs_sum))<<std::endl;
  }
  

  assert(abs(xabs_sum-round(xabs_sum))<1e-9);

  // xabs_ *= (double)target_nnz/xabs_sum;



  std::uniform_real_distribution<> uu_=std::uniform_real_distribution<>(0,1);

  //std::cout<<target_nnz<<std::endl;

  size_t ii = 0;
  size_t jj = 1;
  size_t kk = 2;
  double a=xabs_[0]-floor(xabs_[0]), b=xabs_[1]-floor(xabs_[1]);
  double EPS = 1e-9;

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
      xabs_[ii] = round(a + floor(xabs_[ii]));
      a = xabs_[kk] - floor(xabs_[kk]);
      ii = kk;
      kk++;
    }
    if ( (b<EPS or b>1.0-EPS) and kk<xabs_.size() ){
      //std::cout<<a<<" "<<b<<std::endl;
      xabs_[jj]= round(b + floor(xabs_[jj]));
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
  xabs_[ii] = round(a + floor(xabs_[ii]));
  xabs_[jj] = round(b + floor(xabs_[jj]));

  return 0;
}





inline size_t adjust(std::vector<double>& xabs_, const size_t up) {

  double xabs_sum = std::accumulate(xabs_.begin(), xabs_.end(), 0.0);
  double xabs_max = *std::max_element(xabs_.begin(),xabs_.end());
  double xabs_min = *std::min_element(xabs_.begin(),xabs_.end());

  if (xabs_sum==0){
    return 0;
  }
  
  assert(xabs_min>=0);

  // std::cout<<xabs_max<<std::endl;

  assert(xabs_max<=1.0+1e-9);

  size_t cl = (size_t)ceil(xabs_sum), fl = (size_t)floor(xabs_sum);

  double re = xabs_sum-(double)fl;

  // std::cout<<re<<std::endl;

  if(re<=0){
    assert(up==0);
    return 0;
  }

  if(re>=1.0-1e-9){
    assert(up==1);
    return 0;
  }

  // std::cout<<"up: "<<up<<std::endl;
  // std::cout<<xabs_max<<" "<<cl<<" "<<xabs_sum<<std::endl;

  if(xabs_max*cl/xabs_sum<=1.0){
    std::transform(xabs_.begin(), xabs_.end(), xabs_.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, (double)(fl+up)/xabs_sum));
    return 0;
  }

  size_t kk;

  // std::cout<<"pass1"<<std::endl;
  // std::cout<<up<<"  "<<xabs_.size()<<"  "<<xabs_.sum()<<std::endl;
  // for(size_t jj=0; jj<xabs_.size(); jj++){
  //   std::cout<<jj<<"  "<<xabs_[jj]<<std::endl;
  // }
  if (up==1){
    kk=0;
    while( kk<xabs_.size() and xabs_sum<(double)cl ){
      if( xabs_[kk]<re ){
        xabs_sum += (xabs_[kk]/re)*(1.0-re);
        xabs_[kk] /= re;
      }
      else{
        xabs_sum += 1.0 - xabs_[kk];
        xabs_[kk] = 1.0;
      }
      kk++;
    }
    if( kk<xabs_.size() ){
      xabs_[kk] += cl-xabs_sum;
    }
  }
  else{
    kk=0;
    while( kk<xabs_.size() and xabs_sum>(double)fl){
      if( xabs_[kk]>re ){
        xabs_sum -= ((1.0-xabs_[kk])/(1.0-re))*re;
        xabs_[kk] = (xabs_[kk]-re)/(1.0-re);
      }
      else{
        xabs_sum -= xabs_[kk];
        xabs_[kk] = 0;
      }
      kk++;
    }
    if( kk<xabs_.size() ){
      xabs_[kk] += fl-xabs_sum;
    }
  }
  // std::cout<<"pass2"<<std::endl;

return 0;
}











template <typename IdxType, typename ValType, class RNG>
class FRIiter {
public:

  // Constructor taking max_size as an argument and 
  // allocating space for that many SparseEntry structs.
  FRIiter(size_t max_size, RNG& generator);

  inline void clear();

  inline const size_t size() const {return curr_size_;}

  //void normalize();

  inline void print();

  inline void set_vector(const SparseVector<IdxType,ValType>& other);

  inline void get_vector(SparseVector<IdxType,ValType>& other);

  // inline const IdxType get_parents(const size_t ent_num);

  inline void compress(size_t m);

  inline void remove_zeros();

  inline void assign_parents(SparseMatrix<IdxType,ValType>& A);

  inline const SparseVectorEntry<IdxType,ValType>& operator[](const size_t ent_num) const {assert(ent_num<curr_size_); return state_[ent_num];}

  // Assignment by value up to current size, leaving
  // max size & other entries unchanged.
  // inline FRIiter<IdxType, ValType>& operator=(const SparseVector<IdxType, ValType> &other);


private:
  // Number of nonzero entries that could be in the vector.
  // Must not be changed by user.
  size_t max_size_;

  // Number of nonzero entries actually in the vector.
  size_t curr_size_;
  // Do not allow manual resizing and pushing/popping of the entries vector.
  std::vector<size_t> split_times_;

  SparseVector<IdxType, ValType> state_;

  // SparseVector<IdxType, ValType> vector_;

  Compressor<IdxType, ValType, RNG> compressor_;

  RNG* gen_;

};


template <typename IdxType, typename ValType, class RNG>
inline FRIiter<IdxType, ValType, RNG>::FRIiter(size_t max_size, RNG& generator) : max_size_(max_size) {

  gen_ = &generator;
  curr_size_ = 0;
  split_times_ = std::vector<size_t>(max_size*max_size);
  state_.allocate(max_size);
  compressor_.allocate(max_size, generator);

  std::fill(split_times_.begin(),split_times_.end(),0);
}



template <typename IdxType, typename ValType, class RNG>
inline void FRIiter<IdxType, ValType, RNG>::clear() {
  curr_size_ = 0;
  state_.clear();
  return;
}




template <typename IdxType, typename ValType, class RNG>
inline void FRIiter<IdxType, ValType, RNG>::print() {
  state_.print();
  
  return;
}



template <typename IdxType, typename ValType, class RNG>
inline void FRIiter<IdxType, ValType, RNG>::compress(size_t m) {
  
  compressor_.compress(state_,m);
  
  return;
}

template <typename IdxType, typename ValType, class RNG>
inline void FRIiter<IdxType, ValType, RNG>::remove_zeros() {

  state_.remove_zeros();

  curr_size_ = state_.size();
  
  return;
}


template <typename IdxType, typename ValType, class RNG>
inline void FRIiter<IdxType, ValType, RNG>::set_vector(const SparseVector<IdxType,ValType>& other){

  assert(max_size_>=other.size());

  state_ = other;

  // std::fill(split_times_.begin(), split_times_.end(),0);

  return;
}


template <typename IdxType, typename ValType, class RNG>
inline void FRIiter<IdxType, ValType, RNG>::get_vector(SparseVector<IdxType,ValType>& other){

  assert(curr_size_<=other.capacity());

  other = state_;

  return;
}



// template <typename IdxType, typename ValType, class RNG>
// inline const IdxType FRIiter<IdxType, ValType, RNG>::get_parents(const size_t ent_num){

//   assert(ent_num< curr_size_);

//   e = get_row_entry(ent_num, 0);

//   return e.col_idx;
// }











template <typename IdxType, typename ValType, class RNG>
inline void FRIiter<IdxType, ValType, RNG>::assign_parents(SparseMatrix<IdxType, ValType> &A) {

  if(!A.check_crs_sorted())
    A.sort_crs();

  std::vector<ValType> p(A.rowcol_capacity());
  std::valarray<double> p_abs(A.rowcol_capacity());

  std::uniform_real_distribution<> uu_=std::uniform_real_distribution<>(0,1);

  ValType rowsum;
  double rowabssum;
  double U;
  double S;
  size_t kk;

  for( size_t ii = 0; ii<A.nrows(); ii++){
    if (A.row_size(ii)>0){
      A.get_all_row_values(ii,p);
      A.set_all_row_values(ii,0);
      rowsum = 0;
      rowabssum = 0;
      for(size_t jj=0; jj<A.row_size(ii); jj++){
        p_abs[jj] = abs(p[jj]);
        if( p_abs[jj]<1e-9 ){
          p_abs[jj]=0;
          p[jj]=0;
        }
        rowabssum += p_abs[jj];
        rowsum += p[jj];
      }
      // assert(rowabssum>0);
      if(rowabssum>0){
        U = rowabssum*uu_(*gen_);
        S = p_abs[0];
        kk = 0;
        // std::cout<<A.row_size(ii)<<" "<<U<<" "<<rowsum<<std::endl;
        while( S<U ){
          kk++;
          assert(kk<A.row_size(ii));
          S+=p_abs[kk];
        }
        // std::cout<<kk<<" "<<A.row_size(ii)<<" "<<S<<" "<<U<<std::endl;
        A.set_row_value(ii,kk,rowsum);
      }
    }
  }

  return;
}







// template <typename IdxType, typename ValType, class RNG>
// inline void FRIiter<IdxType, ValType, RNG>::update_split_times(){


//   for(size_t jj=0; jj<curr_size_; jj++){
//     for(size_t ii=jj+1; ii<curr_size_; ii++){

//     }
//   }

// }




#endif