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

  const size_t d = 200;

  assert(d<=col.max_size_);

  for(size_t ii=0;ii<d;ii++){
    col[ii].val = ( 1.12-0.72*((double)ii)/(double)d )*(1.12-0.72*((double)jj)/(double)d)/(double)d;
    col[ii].idx = (long)ii;
  }
  col.curr_size_ = d;

  return 0;
}


int main() {
  size_t d = 200;         // full dimension 
  size_t Nspls = 1<<12;      // number of independent samples of the estimator to generate
  size_t Nit = 20;      // number of iterations after burn in
  size_t m = 50;      // compression parameter (after compression vectors have
                         // no more than m non-zero entries)
  size_t bw = d;         // upper bound on the number of entries in each
                         // column of matrix
  size_t seed = 0;        // seed for RNG.  Can choose random seet below.

  // Initialize iterate vectors and submatrix.
  SparseMatrix<long, double> A(d,d);
  SparseVector<long, double> xtrue(d);
  SparseVector<long, double> b(d);
  SparseVector<long, double> y(d);
  SparseVector<long, double> x(2*d);
  SparseVector<long, double> bias(2*d);

  std::vector<size_t> preserve;

  for(size_t jj=0;jj<2*d;jj++){
  	bias[jj].idx = jj;
  	bias[jj].val = 0;
  }
  bias.curr_size_ = d;
  

  // Initialize a seeded random compressor.
  std::random_device rd;
  std::mt19937_64 generator;
  generator = std::mt19937_64(seed);
  //seed = rd();
  Compressor<long, double, std::mt19937_64> compressor(bw * m, generator);

  // Initialize timings.
  clock_t start, end;
  double cpuTime;

  // we'll measure the l2 error.
  double l2err, l1bias=0, var=0;


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

  // compute the true Neumann sum up to Nit powers of G starting from b
  x = b;
  y = b;

	for (size_t jj=0; jj<Nit; jj++){
    A.sparse_colwisemv(Gcolumn, d, y);
    A.row_sums(y);
    sparse_axpy(1.0,y,x);
  }
  xtrue = x;

  // Generate Nspls independent samples of the estimatator of the Neumann sum
	start = clock();
  for (size_t spl = 0; spl<Nspls; spl++){
  	// Compute the Neumann sum up to Nit powers of G starting from b
  	x = b;
  	y = b;
  	
  	for (size_t jj=0; jj<Nit; jj++){
  		// preserve = compressor.preserve(y, m);
  		// std::cout << preserve.size() << std::endl;
  		compressor.compress(y, m);
    	A.sparse_colwisemv(Gcolumn, d, y);
    	A.row_sums(y);
    	sparse_axpy(1.0,y,x);
  	}

  	// Update the bias vector and compute the l2 error of the approximate solution.
  	sparse_axpy(-1.0,xtrue,x);
  	sparse_axpy(1.0,x,bias);

  	l2err = 0;
  	for(size_t jj=0; jj<d; jj++){
  		l2err += x[jj].val*x[jj].val;
  	}
  	var += l2err/(double)Nspls;
	}
	end = clock();

	// Compute the 1 norm of the bias vector.
  for(size_t jj=0; jj<d; jj++){
  	l1bias += fabs(bias[jj].val);
  }
  l1bias /= (double)Nspls;

  // Print result.
  printf("Bias and standard deviation after %lu samples of %lu iterations:  %le\t %le\n", Nspls, Nit, l1bias, sqrt(var));
  printf("Time elapsed:  %lf\n",((double)end - start)/CLOCKS_PER_SEC);


  return 0;  
}
