#ifndef _fri_public_h_
#define _fri_public_h_



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

int compress(double *y, long L, long n);

template <typename Type>
int heapsort(Type *base, long L, int (*compr)(Type& a, Type& b));

template <typename Type>
int pullroot(Type *base, long L, int (*compr)(Type& a, Type& b));

template <typename Type>
int heapify(Type *base, long L, int (*compr)(Type& a, Type& b));

template <typename Type>
int siftdown(Type *base, long start, long end, int (*compr)(Type& a, Type& b));

#endif
