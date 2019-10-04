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

  for(size_t ii=0;ii<d;ii++){
    Gcolumn[ii].val = ( 1.12-0.72*((double)ii-1.0)/(double)d )*(1.12-0.72*((double)jj-1.0)/(double)d)/(double)d;
    Gcolumn[ii].idx = (long)ii;
  }
  Gcolumn.curr_size_ = d;

  return 0;
}


int main() {
  size_t Nit = 1000;      // number of iterations after burn in
  size_t Brn = 100;      // number of burn in iterations (these
                         // are not included in trajectory averages)
  size_t m = 2;      // compression parameter (after compression vectors have
                         // no more than m non-zero entries)
  size_t bw = 2;         // upper bound on the number of entries in each
                         // column of matrix

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

  // Initialize a sparse matrix;

  SparseMatrix<long, double> A(3,3);

  SparseVector<long, double> x(9);

  v.curr_size_ = 2;
  v[0].val = 1.0;
  v[0].idx = 1;
  v[1].val = 2.0;
  v[1].idx = 4;

  A.set_col(v,2);

  v.curr_size_ = 2;
  v[0].val = 3.0;
  v[0].idx = 0;
  v[1].val = 4.0;
  v[1].idx = 3;

  A.set_col(v,1);



  A.print_ccs();

  v.curr_size_ = 2;
  v[0].val = 5.0;
  v[0].idx = 0;
  v[1].val = 6.0;
  v[1].idx = 2;

  A.set_col(v,1);

  A.print_ccs();

  A.print_crs();

  A.print_ccs();

  v.curr_size_ = 2;
  v[0].val = 7.0;
  v[0].idx = 4;
  v[1].val = 8.0;
  v[1].idx = 5;

  A.set_col(v,0);

  A.print_crs();

  A.print_ccs();

  A.get_row(1,x);

  print_vector(x);

  A.col_sums(x);

  print_vector(x);

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
