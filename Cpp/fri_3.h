// declaration of generic Fast Randomized Iteration
// subroutines
// (c) Jonathan Weare 2015

#ifndef _fri_3_h_
#define _fri_3_h_

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
  inline SparseVectorEntry<IdxType, ValType>& operator[](const size_t idx) {return entries_[idx];}
  inline const SparseVectorEntry<IdxType, ValType>& operator[](const size_t idx) const {return entries_[idx];}
  typedef typename std::vector<SparseVectorEntry<IdxType, ValType>>::iterator spvec_iterator;
  inline spvec_iterator begin() {return entries_.begin();}
  inline spvec_iterator end() {return entries_.begin() + curr_size_;}

private:
  // Do not allow manual resizing and pushing/popping of the entries vector.
  std::vector<SparseVectorEntry<IdxType, ValType>> entries_;
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

// Perform  y <- α x + y.
// y must be large enough to contain 
// all entries of x and y; no allocation
// will be performed by this routine.
template <typename IdxType, typename ValType>
inline int sparse_axpy_v2(ValType alpha,
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
  // Number of nonzero entries actually in the vector.
  size_t curr_size_;
  // Number of nonzero entries that could be in the vector.
  // Must not change.
  const size_t max_size_;

  // Number of rows and columns.
  size_t n_rows_, n_cols_;

  // Max number of rows and columns.
  const size_t max_rows_, max_cols_;

  // Constructor taking max_rows, max_cols, and max_size as an argument and 
  // allocating space for that many SparseMatrixEntry structs.
  SparseMatrix(size_t max_rows, size_t max_cols, size_t max_size);

  // Assignment by value up to current size, leaving
  // max size & other entries unchanged.
  inline SparseMatrix<IdxType, ValType>& operator=(const SparseMatrix<IdxType, ValType> &other);

  // Index reordering to Compressed Row Storage (CRS)
  inline void sort_crs();
  inline void heap_crs();
  inline const bool check_crs_sorted() const {return is_crs_sorted_;}
  inline const bool check_crs_heaped() const {return is_crs_heaped_;}
  inline const SparseMatrixEntry<IdxType, ValType>& crs_get(const size_t idx);

  // Index reordering to Compressed Column Storage (CCS)
  inline void sort_ccs();
  inline void heap_ccs();
  inline const bool check_ccs_sorted() const {return is_ccs_sorted_;}
  inline const bool check_ccs_heaped() const {return is_ccs_heaped_;}
  inline const SparseMatrixEntry<IdxType, ValType>& ccs_get(const size_t idx);

  // Reset matrix without deallocating space.
  inline void clear();

  // Write column idx into a sparse vector.
  inline void get_col(const size_t idx, SparseVector<IdxType,ValType>& other);

  // Write row idx into a sparse vector.
  inline void get_row(const size_t idx, SparseVector<IdxType,ValType>& other);

  // Add a new column to the matrix.  If there's already a column with index idx
  // it is replaced.
  inline void set_col(const SparseVector<IdxType,ValType>& other, const IdxType idx);

  // Add a new row to the matrix.  If there's already a row with index idx
  // it is replaced.
  inline void set_row(const SparseVector<IdxType,ValType>& other, const IdxType idx);

  // Compute column sums and output in sparse vector.
  inline void column_sums(SparseVector<IdxType,ValType>& other);

  // Compute column norms and output in sparse vector.
  inline void column_norms(SparseVector<IdxType,ValType>& other, size_t p);

  // Compute row sums and output in sparse vector.
  inline void row_sums(SparseVector<IdxType,ValType>& other);

  // Compute row norms and output in sparse vector.
  inline void row_norms(SparseVector<IdxType,ValType>& other, size_t p);

  // Accessors for the underlying vector so one can do
  // some of the normal vector manipulation, but not all.
  // Specifically, subscripting and iterator begin and end
  // are provided. Incremental queue and stack function is 
  // intentionally not provided.
  inline void set_entry(const SparseMatrixEntry<IdxType, ValType>& a, const size_t idx);
  inline const SparseMatrixEntry<IdxType, ValType>& get_entry(const size_t idx) const {return entries_[idx];}
  typedef typename std::vector<SparseMatrixEntry<IdxType, ValType>>::iterator spmat_iterator;
  inline spmat_iterator begin() {return entries_.begin();}
  inline spmat_iterator end() {return entries_.begin() + curr_size_;}

private:
  // Do not allow manual resizing and pushing/popping of the entries 
  // or CCS and CRS ordering vectors.
  std::vector<SparseMatrixEntry<IdxType, ValType>> entries_;
  std::vector<size_t> ccs_order_;
  std::vector<size_t> crs_order_;
  std::vector<size_t> row_heads_;
  std::vector<size_t> col_heads_;

  // std::vector<size_t> ccs_order_;
  // std::vector<size_t> crs_order_;
  // std::vector<size_t> rows_;
  // std::vector<size_t> cols_;
  bool is_crs_sorted_;
  bool is_ccs_sorted_;
};



// Print a matrix to cout by printing the number 
// of nonzero entries and then printing each entry
// as a val idx pair. 
template <typename IdxType, typename ValType>
inline void print_matrix(SparseMatrix<IdxType, ValType> &mat);


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
template <typename IdxType, typename ValType>
inline int spcolwisemv(int (*Acolumn)(SparseVector<IdxType, ValType> &col, const IdxType jj),
        size_t max_nz_col_entries, const SparseVector<IdxType, ValType> &x,
        SparseMatrix<IdxType, ValType> &B);







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
struct spmatcomparebyrowidxfirst {
  template <typename IdxType, typename ValType>
  inline bool operator () (const SparseMatrixEntry<IdxType, ValType> &a, const SparseMatrixEntry<IdxType, ValType> &b) {
    if (a.rowidx<b.rowidx) return true;
    else if (a.rowidx==b.rowidx & a.colidx<b.colidx) return true;
    else return false;
  }
};

// Compare two sparse matrix entries by lexicographic 
// ordering of indices with column index first.
struct spmatcomparebycolidxfirst {
  template <typename IdxType, typename ValType>
  inline bool operator () (const SparseMatrixEntry<IdxType, ValType> &a, const SparseMatrixEntry<IdxType, ValType> &b) {
    if (a.colidx<b.colidx) return true;
    else if (a.colidx==b.colidx & a.rowidx<b.rowidx) return true;
    else return false;
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
  // allocating space for that many SparseVectorEntry structs.
template <typename IdxType, typename ValType>
inline SparseVector<IdxType, ValType>::SparseVector(size_t max_size) : max_size_(max_size) {
  curr_size_ = 0;
  entries_ = std::vector<SparseVectorEntry<IdxType, ValType>>(max_size);
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

// Perform  y <- α x + y.
// y must be large enough to contain 
// all entries of x and y; no allocation
// will be performed by this routine.
// *** UNTESTED ****
template <typename IdxType, typename ValType>
inline int sparse_axpy_v2(ValType alpha,
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
  size_t jj = 0;
  while (jj < x.curr_size_){
    // If there are y entries left and the y entry has lower
    // index, move it in.
    while (y_entries_begin < y_entries_end && y[y_entries_begin].idx < x[jj].idx){
      y[n_result_entries] = y[y_entries_begin];
      y_entries_begin++;
      n_result_entries++;
    }
    // If the x and y entries had equal index, add the x 
    // entry times alpha to the y entry.
    while (jj<x.curr_size_ && y_entries_begin < y_entries_end && x[jj].idx == y[y_entries_begin].idx){
      y[n_result_entries] = y[y_entries_begin];
      y[n_result_entries].val += alpha * x[jj].val;
      jj++;
      y_entries_begin++;
      n_result_entries++;
    }
    // Otherwise, just move the x entry times alpha in. 
    while (jj<x.curr_size_ && y[y_entries_begin].idx > x[jj].idx){
      y[n_result_entries].idx = x[jj].idx;
      y[n_result_entries].val = alpha * x[jj].val;
      jj++;
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
  std::make_heap(y.begin(), y.begin() + y.curr_size_, spveccomparebyidx());
  std::sort_heap(y.begin(), y.begin() + y.curr_size_, spveccomparebyidx());

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






//---------------------------------------------------------
// Sparse matrix class definition and routines
// Implementation
//---------------------------------------------------------

// Constructor taking max_rows, max_cols, and max_size as an argument and 
// allocating space for that many SparseMatrixEntry structs.
template <typename IdxType, typename ValType>
inline SparseMatrix<IdxType, ValType>::SparseMatrix(size_t max_rows, size_t max_cols, size_t max_size)
  : max_size_(max_size), max_rows_(max_rows), max_cols_(max_cols) {
  curr_size_ = 0;
  n_rows_ = 0;
  n_cols_ = 0;
  entries_ = std::vector<SparseMatrixEntry<IdxType, ValType>>(max_size);
  crs_order_ = std::vector<size_t>(max_size);
  ccs_order_ = std::vector<size_t>(max_size);
  row_heads_ = std::vector<size_t>(max_rows);
  col_heads_ = std::vector<size_t>(max_cols);
  is_crs_sorted_ = true;
  is_ccs_sorted_ = true;
}


// Reset sparse matrix without deallocating space.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::clear(){
  curr_size_ = 0;
  n_rows_ = 0;
  n_cols_ = 0;
  is_crs_sorted_ = true;
  is_ccs_sorted_ = true;
}



// Add a single SparseMatrixEntry to a sparse matrix.
// This destroys CCS or CRS sorting but preserves heaped ordering.
// template <typename IdxType, typename ValType>
// inline bool SparseMatrix<IdxType, ValType>::set_entry(const SparseMatrixEntry<IdxType, ValType>& a){

//   assert( curr_size_ < max_size_ );

//   entries_[curr_size_] = a;
//   curr_size_++;

//   if (!is_crs_heaped_){
//     size_t jj;
//     for (jj=0; jj<curr_size_; jj++)
//       crs_order_[jj] = &entries_[jj];

//     std::make_heap(crs_order_.begin(), crs_order_.begin()+curr_size_, 
//       [&](SparseMatrixEntry<IdxType, ValType>* a, SparseMatrixEntry<IdxType, ValType>* b) { return spmatcomparebyrowidxfirst(*a,*b); });
//   }
//   else {
//     crs_order_[curr_size_-1] = &entries_[curr_size_-1];
//     std::push_heap(crs_order_.begin(), crs_order_.begin()+curr_size_, 
//       [&](SparseMatrixEntry<IdxType, ValType>* a, SparseMatrixEntry<IdxType, ValType>* b) { return spmatcomparebyrowidxfirst(*a,*b); });
//   }

//   if (!is_ccs_heaped_){
//     size_t jj;
//     for (jj=0; jj<curr_size_; jj++)
//       ccs_order_[jj] = &entries_[jj];

//     std::make_heap(ccs_order_.begin(), ccs_order_.begin()+curr_size_, 
//       [&](SparseMatrixEntry<IdxType, ValType>* a, SparseMatrixEntry<IdxType, ValType>* b) { return spmatcomparebycolidxfirst(*a,*b); });
//   }
//   else {
//     ccs_order_[curr_size_-1] = &entries_[curr_size_-1];
//     std::push_heap(ccs_order_.begin(), ccs_order_.begin()+curr_size_, 
//       [&](SparseMatrixEntry<IdxType, ValType>* a, SparseMatrixEntry<IdxType, ValType>* b) { return spmatcomparebycolidxfirst(*a,*b); });
//   }

//   is_crs_sorted_ = false;
//   is_crs_heaped_ = true;
//   is_ccs_sorted_ = false;
//   is_ccs_heaped_ = true;
//   return true;
// }


// Add a single SparseMatrixEntry to a sparse matrix.
// If there is already an entry in the matrix with
// the same row and column indices, its value is updated.
// This leaves the entries both CCS and CRS sorted.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::set_entry(const SparseMatrixEntry<IdxType, ValType>& other){

  assert( curr_size_ < max_size_ );

  if (!is_crs_sorted_)
    sort_crs();

  if (!is_ccs_sorted_)
    sort_ccs();

  // Was the sparse matrix empty?
  if (curr_size_ == 0){
    assert( n_cols_ < max_cols_);
    assert( n_rows_ < max_rows_);

    ccs_order_[0] = 0;
    crs_order_[0] = 0;

    col_heads_[0] = 0;
    row_heads_[0] = 0;

    entries_[0] = other;

    curr_size_++;
    n_cols_++;
    n_rows_++;

    return;
  }

  // Decide if it's a new entry and fix column ordering.

  bool is_new_entry;

  if ( entries_[ccs_order_[curr_size_-1]].colidx < other.colidx ){  // is this a new column with greater index than the rest?
    assert( n_cols_ < max_cols_);

    is_new_entry = true;

    ccs_order_[curr_size_] = curr_size_;

    col_heads_[n_cols_] = curr_size_;

    n_cols_++;
  }
  else{                                                            // entry doesn't have greater column index than the rest.
    std::vector<size_t>::iterator col_num, ccs_pos, it;

    col_num = std::lower_bound(col_heads_.begin(), col_heads_.begin()+n_cols_, other.colidx,
      [&](size_t ii, size_t jj) { return entries_[ccs_order_[ii]].colidx < entries_[ccs_order_[jj]].colidx; });

    if( entries_[ccs_order_[*col_num]].colidx > other.colidx ){  // is this a new column?
      assert( n_cols_ < max_cols_);
      is_new_entry = true;

      ccs_pos = ccs_order_.begin() + *col_num;

      std::move_backwards(col_pos, ccs_order_.begin() + curr_size_, ccs_order_.begin()+curr_size_+1);
      *ccs_pos = curr_size_;

      it = col_heads_.begin()+n_cols_;
      while ( it != col_num){
        *it = *(it-1) + 1;
        it--;
      }
      n_cols_++;
    }
    else{                                                            // this isn't a new column.
      assert( entries_[ccs_order_[*col_num]].colidx == other.colidx );

      size_t col_end;

      if (col_num == col_heads_.begin()+n_cols_-1)
        col_end = curr_size_;
      else
        col_end = *(col_num+1);

      if ( entries_[ccs_order_[col_end-1]].rowidx < other.rowidx ){   // does this entry go at the end of a column
        is_new_entry = true;

        ccs_pos = ccs_order_.begin()+col_end;

        if (col_end < curr_size_){
          std::move_backwards(ccs_pos, ccs_order_.begin()+curr_size_,ccs_order_.begin()+curr_size_+1);
          for ( it = col_num+1; it != col_heads_.begin()+n_cols_; it++)
            (*it)++;
        }
        
        *ccs_pos = curr_size_;
      }
      else{                                                             // this doesn't go at the end of a column
        ccs_pos = std::lower_bound(ccs_order_.begin()+*col_num, ccs_order_.begin()+col_end, other.rowidx,
          [&](size_t ii, size_t jj) { return entries_[ii].rowidx < entries_[jj].rowidx; });

        if( entries_[*ccs_pos].rowidx == other.rowidx ){                  // does it replace an existing entry?
          is_new_entry=false;
          entries_[*ccs_pos].val = other.val
        }
        else{                                                               // it doesn't replace existing entry.
          is_new_entry=true;
          std::move_backwards(ccs_pos, ccs_order_.begin()+curr_size_,ccs_order_.begin()+curr_size_+1);
          *ccs_pos = curr_size_;
          

          for ( it = col_num+1; it != col_heads_.begin()+n_cols_; it++)
            (*it)++;
        }
      }
    }
  }


  // Fix row ordering.


  if(is_new_entry){
    entries_[curr_size_] = other;
    curr_size_++;
  }

  return;
}



// Add a sparse vector as a column to a sparse matrix.
// If the matrix already has that column it is replaced.
// This leaves the matrix CCS ordered, but destroys CRS ordering.
template <typename IdxType, typename ValType>
inline void SparseMatrix<IdxType, ValType>::set_col(const IdxType idx, const SparseVector<IdxType, ValType>& other){

  assert( curr_size_+other.curr_size_<=max_size_);
  assert( n_cols_<max_cols_);
  assert( n_rows_+other.curr_size_<=max_rows_);

  if (!is_ccs_sorted_)
    sort_crs();

  if (curr_size_==0){                    // is the matrix empty?

    std::iota(ccs_order_.begin(),ccs_order_.begin()+other.curr_size_,0);
    std::iota(crs_order_.begin(),crs_order_.begin()+other.curr_size_,0);
    std::iota(row_heads_.begin(),row_heads_.begin()+other.curr_size_,0);
    
    col_heads_[0] = 0;

    for( jj=0; jj<other.curr_size_;jj++){            // copy the new column at the end of entries_
      entries_[curr_size_+jj].colidx = idx;
      entries_[curr_size_+jj].rowidx = other[jj].idx;
      entries_[curr_size_+jj].val = other[jj].val;
    }
  
    curr_size_ += other.curr_size_;
    n_cols_++;
    n_rows_ = other.curr_size_;

    return;
  }

  is_crs_sorted_ = false;

  std::vector<size_t>::iterator ccs_pos, col_id, it;
  size_t col_end, old_len;

  if ( idx > entries_[ccs_order_[curr_size_-1]] ){                // will this be the largest column index

    ccs_pos = ccs_order_.begin()+curr_size_;

    std::iota(ccs_pos, ccs_pos+other.curr_size_,curr_size_);

    col_heads_[n_cols_] = curr_size_;

    for( jj=0; jj<other.curr_size_;jj++){            // copy the new column at the end of entries_
      entries_[curr_size_+jj].colidx = idx;
      entries_[curr_size_+jj].rowidx = other[jj].idx;
      entries_[curr_size_+jj].val = other[jj].val;
    }

    curr_size_+=other.curr_size_;
    n_cols_++;
  }
  else{                                                              // this won't be the largest column index
    col_id = std::lower_bound(col_heads_.begin(), col_heads_.begin()+n_cols_, idx,
      [&](size_t ii, size_t jj) { return entries_[ccs_order_[ii]].colidx < entries_[ccs_order_[jj]].colidx; });

    ccs_pos = ccs_order_.begin() + *col_id;

    if (entries_[ccs_order_[*col_id]].colidx > idx){                 // is this a new column index?

      std::move_backwards(ccs_pos, ccs_order_.begin()+curr_size_,ccs_order_.begin()+curr_size_+other.curr_size_);
      std::iota(ccs_pos, ccs_pos+other.curr_size_,curr_size_);

      it = col_heads_.begin()+n_cols_;
      while ( it != col_id){
        *it = *(it-1) + 1;
        it--;
      }
      n_cols_++

      for( jj=0; jj<other.curr_size_;jj++){            // copy the new column at the end of entries_
        entries_[curr_size_+jj].colidx = idx;
        entries_[curr_size_+jj].rowidx = other[jj].idx;
        entries_[curr_size_+jj].val = other[jj].val;
      }

      curr_size_ += other.curr_size_;
    }
    else{                                                        // this replaces an existing column

      assert( entries_[ccs_order_[*col_id]].colidx == idx );

      if (col_id == col_heads_.begin()+n_cols_-1)
        col_end = curr_size_;
      else
        col_end = *(col_id+1);

      old_len = col_end - *col_id;

      if(other.curr_size_>=old_len){                             // is the new column longer than the old one?

        for( jj=0; jj<old_len;jj++){                             // copy new entries over old entries
          entries_[*(ccs_pos+jj)].colidx = idx;
          entries_[*(ccs_pos+jj)].rowidx = other[jj].idx;
          entries_[*(ccs_pos+jj)].val = other[jj].val;
        }
        for( jj=old_len; jj<other.curr_size_;jj++){              // copy the rest of the new column at the end of entries_
          entries_[curr_size_-old_len+jj].colidx = idx;
          entries_[curr_size_-old_len+jj].rowidx = other[jj].idx;
          entries_[curr_size_-old_len+jj].val = other[jj].val;
        }

        std::move_backwards(ccs_order_.begin()+col_end, ccs_order_.begin()+curr_size_,ccs_order_.begin()+curr_size_+other.curr_size_-old_len);
        std::iota(ccs_pos+other.curr_size_-old_len, ccs_pos+other.curr_size_, curr_size_-old_len);

      }
      else{                                                       // the new column is shorter than the old one.
        
        for( jj=0; jj<other.curr_size_;jj++){                             // copy new entries over old entries
          entries_[*(ccs_pos+jj)].colidx = idx;
          entries_[*(ccs_pos+jj)].rowidx = other[jj].idx;
          entries_[*(ccs_pos+jj)].val = other[jj].val;
        }
        for( jj=0;jj<curr_size_-col_end;jj++ )                       // move iterators forward to cover unfilled old entries
          std::iter_swap(entries_.begin()+*(ccs_pos+other.curr_size_+jj),entries_.begin()+*(ccs_pos+col_end+jj));
        for( jj=0;jj<curr_size_-col_end;jj++ ) 
          *(ccs_pos+other.curr_size_+jj) -= old_len-other.curr_size_;

      }

      if( col_end < curr_size_ )                  // fix column head positions
        for ( it = col_id+1; it != col_heads_.begin()+n_cols_; it++)
          (*it) += other.curr_size_-old_len;

      curr_size_ += other.curr_size_-old_len;
    }
  }

  return;
}


// Add a sparse vector as a row to a sparse matrix.
// If the matrix already has that column it is replaced.
// This leaves the matrix CRS ordered, but destroys CCS ordering.


// Find the indexing that reorders the columns into CRS order.
// On output is_crs_ordered will be set to true and that
// value will be returned.
template <typename IdxType, typename ValType>
inline bool SparseMatrix<IdxType, ValType>::sort_crs(){

  if (is_crs_sorted_==true) return true;

  size_t jj;
  for (jj=0; jj<curr_size_; jj++)
    crs_order_[jj] = &entries_[jj];

  std::sort(crs_order_.begin(), crs_order_.begin()+curr_size_, 
    [&](SparseMatrixEntry<IdxType, ValType>* a, SparseMatrixEntry<IdxType, ValType>* b) { return spmatcomparebyrowidxfirst(*a,*b); });

  size_t row_count = 1;
  for (jj=1; jj<curr_size_;jj++){
    if ( *(crs_order_[jj]).rowidx > *(crs_order_[jj-1]).rowidx )
      row_ends_[row_count-1] = jj;
      row_count++; 
  }
  n_rows_ = row_count;

  is_crs_sorted_=true;

  return is_crs_sorted_;
}

// Find the indexing that reorders the columns into CCS order.
// On output is_ccs_ordered will be set to true and that
// value will be returned.
template <typename IdxType, typename ValType>
inline bool SparseMatrix<IdxType, ValType>::sort_ccs(){

  if (is_crs_sorted_==true) return true;

  
  size_t jj;
  for (jj=0; jj<curr_size_; jj++)
    ccs_order_[jj] = &entries_[jj];

  std::sort(ccs_order_.begin(), ccs_order_.begin()+curr_size_, 
    [&](SparseMatrixEntry<IdxType, ValType>* a, SparseMatrixEntry<IdxType, ValType>* b) { return spmatcomparebycolidxfirst(*a,*b); });

  size_t col_count = 1;
  for (jj=1; jj<curr_size_;jj++){
    if ( *(ccs_order_[jj]).colidx > *(ccs_order_[jj-1]).colidx )
      col_ends_[col_count-1] = jj;
      col_count++; 
  }
  n_cols_ = col_count;

  is_ccs_sorted_=true;

  return is_ccs_sorted_;
}

// Copy entries of column idx of a sparse matrix
// into a sparse vector.
template <typename IdxType, typename ValType>
inline int SparseMatrix<IdxType, ValType>::copy_col(const size_t idx, SparseVector<IdxType, ValType>& y){
  assert(idx<n_cols_);

  if(!is_ccs_sorted_)
    sort_ccs();

  size_t col_start;
  if (idx==0)
    col_start = 0;
  if (idx>0)
    col_start = col_ends_[idx-1];

  assert( y.max_size_ >= col_ends_[idx] - col_start );
  y.curr_size_ = col_ends_[idx] - col_start;

  for (size_t jj=col_start; jj<col_ends_[idx]; j++){
    y.entries_[jj].idx = *ccs_order_[jj].rowidx;
    y.entries_[jj].val = *ccs_order_[jj].val;
  }

  return 0;
}

// Copy entries of row idx of a sparse matrix
// into a sparse vector.
template <typename IdxType, typename ValType>
inline int SparseMatrix<IdxType, ValType>::copy_row(const size_t idx, SparseVector<IdxType, ValType>& y){
  assert(idx<n_rows_);

  if(!is_crs_sorted_)
    sort_crs();

  size_t row_start;
  if (idx==0)
    row_start = 0;
  if (idx>0)
    row_start = row_ends_[idx-1];

  assert( y.max_size_ >= row_ends_[idx] - row_start );
  y.curr_size_ = row_ends_[idx] - row_start;

  for (size_t jj=row_start; jj<row_ends_[idx]; j++){
    y.entries_[jj].idx = *crs_order_[jj].colidx;
    y.entries_[jj].val = *crs_order_[jj].val;
  }

  return 0;
}



// Quary the entries in CRS order.  The CRS order
// must have already been set by a 
// call to set_crs_order()
template <typename IdxType, typename ValType>
inline const SparseMatrixEntry<IdxType, ValType>& SparseMatrix<IdxType,ValType>::crs(const size_t idx) {
  assert(is_crs_ordered_);
  return *crs_order_[idx];
}

// Quary the entries in CCS order.  The CCS order
// must have already been set by a 
// call to set_ccs_order()
template <typename IdxType, typename ValType>
inline const SparseMatrixEntry<IdxType, ValType>& SparseMatrix<IdxType,ValType>::ccs(const size_t idx) {
  assert(is_ccs_ordered_);
  return *ccs_order_[idx];
}

  // Assignment by value up to current size, leaving
  // max size & other entries unchanged.
template <typename IdxType, typename ValType>
inline SparseMatrix<IdxType, ValType>& SparseMatrix<IdxType, ValType>::operator=(const SparseMatrix<IdxType, ValType> &other) {
  assert(other.curr_size_ <= max_size_);
  curr_size_ = other.curr_size_;
  for (size_t i = 0; i < other.curr_size_; i++) {
    entries_[i] = other.entries_[i];
    crs_order_[i] = other.crs_order_[i];
    ccs_order_[i] = other.ccs_order_[i];
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
inline void print_vector(SparseMatrix<IdxType, ValType> &mat) {
  std::cout << mat.curr_size_ << std::endl;
  for (size_t jj = 0; jj < mat.curr_size_; jj++) {
    std::cout << mat[jj].val << "\t " << mat[jj].rowidx << "\t" << mat[jj].colidx << std::endl;
  }
  std::cout << std::endl;
}


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
template <typename IdxType, typename ValType>
inline int spcolwisemv(int (*Acolumn)(SparseVector<IdxType, ValType> &col, const IdxType jj),
        size_t max_nz_col_entries, const SparseVector<IdxType, ValType> &x,
        SparseMatrix<IdxType, ValType> &B)
{
  // Check for correct size; multiplication must not overflow.
  assert(max_nz_col_entries * x.curr_size_ <= B.max_size_);

  // Make a list of all the entries in A scaled
  // by the entry of x corresponding to the column
  // containing the particular entry of A; these
  // will all be added to the result.
  size_t n_entry_adds=0;
  SparseVector<IdxType, ValType> single_row_by_column_adds(max_nz_col_entries);
  for (size_t jj = 0; jj < x.curr_size_; jj++) {
    Acolumn(single_row_by_column_adds, x[jj].idx);
    for(size_t ii = 0; ii < single_row_by_column_adds.curr_size_; ii++){
      B[n_entry_adds].val = x[jj].val * single_row_by_column_adds[ii].val;
      B[n_entry_adds].rowidx = single_row_by_column_adds[ii].idx;
      B[n_entry_adds].colidx = x[jj].idx;
      n_entry_adds++;
    }
  }
  B.curr_size_ = n_entry_adds;

  std::iota(begin(B.ccs_order_), begin(B.ccs_order_)+n_entry_adds, static_cast<size_t>(0));

  B.is_ccs_ordered_ = true;

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
                     spveccomparebyval());
    
      while (( (target_nnz - nnz_large) * xabs_[0].val > sum_unprocessed) 
               && (nnz_large < target_nnz)) {
        sum_unprocessed -= xabs_[0].val;
        assert(sum_unprocessed >0);
        std::pop_heap(xabs_.begin(),
                     xabs_.begin() + xabs_.curr_size_ - nnz_large, 
                     spveccomparebyval());
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

    nnz = 0;
    for (size_t ii = 0; ii < xabs_.curr_size_; ii++) {
      if( xabs_[ii].val > 0) nnz++;
    }
  }
}

#endif