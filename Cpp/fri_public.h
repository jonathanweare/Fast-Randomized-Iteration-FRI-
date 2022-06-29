// declaration of generic Fast Randomized Iteration
// subroutines
// (c) Jonathan Weare 2015

#ifndef _fri_public_h_
#define _fri_public_h_


// The struct spentry is declared in fri_public.h
// and contains a value and an index.  
// A sparse vector v is stored as an integer (say nv)
// along with an array with elements of type spentry

template <typename Type>
struct spentry{
   Type    val;
   long    loc;
};

int sparse_daxpy(double alpha,
		 spentry<double> *x, long *nx,
		 spentry<double> *y, long *ny);

int dspcompmatmult(long m, double alpha,
		  int (*Acolumn)(spentry<double> *col, long *nrows, long jj),
		  long bw, spentry<double> *x, long *nx,double beta,
		  spentry<double> *y, long *ny);

int sparse_dgemv(double alpha,
		int (*Acolumn)(spentry<double> *col, long *nrows, long jj),
		long bw,
		spentry<double> *x, long *nx, double beta,
		spentry<double> *y, long *ny);

template <typename Type>
int spcomparebyloc( Type& a, Type& b);

template <typename Type>
int spcomparebyval( Type& a, Type& b);

int compress(spentry<double> *y, long ny, long n);

template <typename Type>
int heapsort(Type *base, long L, int (*compr)(Type& a, Type& b));

template <typename Type>
int pullroot(Type *base, long L, int (*compr)(Type& a, Type& b));

template <typename Type>
int heapify(Type *base, long L, int (*compr)(Type& a, Type& b));

template <typename Type>
int siftdown(Type *base, long start, long end, int (*compr)(Type& a, Type& b));

#endif
