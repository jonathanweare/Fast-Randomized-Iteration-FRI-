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

  const size_t d = 3;

  assert(d<=col.max_size_);

  // for(size_t ii=0;ii<d;ii++){
  //   col[ii].val = (double)jj*d+ii;
  //   col[ii].idx = (long)ii;
  // }
  // col.curr_size_ = d;

  for(size_t ii=0;ii<d;ii++){
    col[ii].val = ( 1.12-0.72*((double)ii-1.0)/(double)d )*(1.12-0.72*((double)jj-1.0)/(double)d)/(double)d;
    col[ii].idx = (long)ii;
  }
  col.curr_size_ = d;

  return 0;
}


int main() {
  size_t Nit = 2;      // number of iterations after burn in
  size_t Brn = 100;      // number of burn in iterations (these
                         // are not included in trajectory averages)
  size_t m = 2;      // compression parameter (after compression vectors have
                         // no more than m non-zero entries)
  size_t bw = 2;         // upper bound on the number of entries in each
                         // column of matrix
  size_t d = 3;

  // Initialize iterate vectors.
  SparseVector<long, double> v(bw * m);
  SparseVector<long, double> vnew(bw * m);
  vnew.curr_size_ = 1;
  v.curr_size_ = 1;                                                    // initial vector
  v[0].val = 1.0;
  v[0].idx = 0;
  normalize(v);
  // Initialize a seeded random compressor.
  std::random_device rd;
  Compressor<long, double> compressor(bw * m, rd());

  SparseMatrix<long, double> A(d,d);

  SparseVector<long, double> x(d);
  SparseVector<long, double> b(2*d);

  for(size_t jj=0; jj<d; jj++){
  	x[jj].val = 1.0/(2.25-1.45*((double)jj-1.0)/(double)d);
  	x[jj].idx = (long)jj;
  }
  x.curr_size_ = d;

  print_vector(x);

  A.sparse_colwisemv(Gcolumn, d, x);

  A.row_sums(b);
  print_vector(b);

  sparse_axpy(-1.0,x,b);

  
  for( size_t jj=0; jj<d; jj++ ){
    b[jj].val = -b[jj].val;
  }

  print_vector(b);

  for (size_t jj=0; jj<Nit; jj++){
    A.sparse_colwisemv(Gcolumn, d, b);
    A.row_sums(b);
  }

  print_vector(b);

  assert(0>1);



  // // Initialize timings.
  // clock_t start, end;
  // double cpuTime;
  
  // // Run Brn iterations without accumulating trajectory
  // // averages to discard initial transient.
  // double lambda;
  // for (size_t t = 0; t < Brn; t++) {

  //   // Perform and time a compression and multiplication.
  //   start = clock();
  //   compressor.compress(v, m);
  //   sparse_gemv(1.0, transfer, bw, v, 0.0, vnew);
  //   end = clock();

  //   // Finish the iteration.
  //   lambda = vnew.norm();
  //   normalize(vnew);
  //   v = vnew;

  //   // Print an iterate summary.
  //   printf("burn: %ld\t lambda: %lf\t nonzeros: %ld\t time / iteration: %lf\n",
  //      t+1, lambda, v.curr_size_, ((double)end - start)/CLOCKS_PER_SEC);
  // }

  // // Generate a trajectory of Nit iterations and 
  // // accumulate trajectory averages.
  // double lambda_ave = 0;
  // for (size_t t = 0; t < Nit; t++){

  //   // Perform and time a compression and multiplication.
  //   start = clock();
  //   compressor.compress(v, m);
  //   sparse_gemv(1.0, transfer, bw, v, 0.0, vnew);
  //   end = clock();

  //   // Finish the iteration.
  //   lambda = vnew.norm();
  //   double eps = 1.0 / (1.0 + t);
  //   lambda_ave = (1.0 - eps) * lambda_ave + eps * lambda;
  //   normalize(vnew);
  //   v = vnew;

  //   // Print an iterate summary.
  //   printf("iteration: %ld\t lambda: %lf\t average: %lf\t nonzeros: %ld\t time / iteration: %lf\n",
  //      t+1, lambda, lambda_ave, v.curr_size_, ((double)end - start)/CLOCKS_PER_SEC);
  // }

  return 0;  
}
