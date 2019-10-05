// A driver to run Fast Randomized Iteration
// applied to solving Ax=b
// (c) Jonathan Weare 2019

#include <iostream>
#include <cstdlib>
#include <math.h>
#include "fri_4.h"

using namespace std;

// A subroutine that returns the ii-th column of
// the matrix of interest
int Gcolumn(SparseVector<long, double> &col, const long jj){

  const size_t d = 10;

  assert(d<=col.max_size_);

  // for(size_t ii=0;ii<d;ii++){
  //   col[ii].val = (double)jj*d+ii;
  //   col[ii].idx = (long)ii;
  // }
  // col.curr_size_ = d;

  for(size_t ii=0;ii<d;ii++){
    col[ii].val = ( 1.12-0.72*((double)ii)/(double)d )*(1.12-0.72*((double)jj)/(double)d)/(double)d;
    col[ii].idx = (long)ii;
  }
  col.curr_size_ = d;

  return 0;
}


int main() {
  size_t d = 10;         // full dimension 
  size_t Nspls = 1;      // number of independent samples of the estimator to generate
  size_t Nit = 10;      // number of iterations after burn in
  size_t m = d;      // compression parameter (after compression vectors have
                         // no more than m non-zero entries)
  size_t bw = d;         // upper bound on the number of entries in each
                         // column of matrix

  // Initialize iterate vectors and submatrix.
  SparseMatrix<long, double> A(m,d);
  SparseVector<long, double> xtrue(d);
  SparseVector<long, double> b(d);
  SparseVector<long, double> y(d);
  SparseVector<long, double> x(2*d);
  SparseVector<long, double> xave(2*d);

  for(size_t jj=0;jj<d;jj++){
  	xave[jj].idx = jj;
  	xave[jj].val = 0;
  }
  

  // Initialize a seeded random compressor.
  std::random_device rd;
  Compressor<long, double> compressor(bw * m, rd());

  // Initialize timings.
  clock_t start, end;
  double cpuTime;

  // we'll measure the l2 error.
  double l2err, bias, var=0;


  // The true solution vector xtrue
  for(size_t jj=0; jj<d; jj++){
  	xtrue[jj].val = 1.0/(2.25-1.45*((double)jj)/(double)d);
  	//x[jj].val = 1.0;
  	xtrue[jj].idx = (long)jj;
  }
  xtrue.curr_size_ = d;

  // b = (I-G)*xtrue
  A.sparse_colwisemv(Gcolumn, d, xtrue);
  A.row_sums(b);
  x = xtrue;
  sparse_axpy(-1.0,b,x);
  b = x;

	start = clock();
  for (size_t spl = 0; spl<Nspls; spl++){
  	// Compute the Neumann sum up to Nit powers of G starting from b
  	x = b;
  	y = b;
  	
  	for (size_t jj=0; jj<Nit; jj++){
  		compressor.compress(y, m);
    	A.sparse_colwisemv(Gcolumn, d, y);
    	A.row_sums(y);
    	sparse_axpy(1.0,y,x);
  	}

  	// Compute the l2 error of the approximate solution.
  	sparse_axpy(-1.0,xtrue,x);
  	l2err = 0;
  	for(size_t jj=0; jj<d; jj++){
  		l2err += x[jj].val*x[jj].val;
  	}
  	var += l2err/(double)Nspls;

  	sparse_axpy(1.0/Nspls,x,xave);
	}
	end = clock();

	sparse_axpy(-1.0,xtrue,xave);
	bias = 0;
  for(size_t jj=0; jj<d; jj++){
  	bias += fabs(xave[jj].val);
  }

  // Print result.
  printf("Bias and variance after %lu samples of %lu iteration:  %le\t %le\n", Nspls, Nit, bias, var);
  printf("Time elapsed:  %lf\n",((double)end - start)/CLOCKS_PER_SEC);


  return 0;  
}
