// A driver to run Fast Randomized Iteration
// applied to solving Ax=b
// (c) Jonathan Weare 2019

#include <iostream>
#include <cstdlib>
#include <math.h>
#include "fri_3.h"

using namespace std;

// A subroutine that returns the ii-th column of
// the transfer matrix of the 2D Ising model.
int transfer(SparseVector<long, double> &col, const long ii){

  int n = 50;
  double BB = 0.01;
  double TT = 2.2;
  
  double aa, bb, cc, invaa, invbb, invcc;
  long L;
  
  col.curr_size_ = 2;

  aa = exp((2.0-BB)/TT);
  bb = exp(-BB/TT);
  cc = exp((-2.0-BB)/TT);
  invaa = exp(-(2.0-BB)/TT);
  invbb = exp(BB/TT);
  invcc = exp((2.0+BB)/TT);

  L = (long)1<<n;
  
  if (ii < (L>>2)){
    col[0].val = aa;
    col[0].idx = (ii<<1);

    col[1].val = bb;
    col[1].idx = (ii<<1)+1;
  }
  else if (ii>= (L>>2) && ii< (L>>1)){
    col[0].val = bb;
    col[0].idx = (ii<<1);

    col[1].val = cc;
    col[1].idx = (ii<<1)+1;
  }

  else if (ii>=(L>>1) && ii< (L>>1)+(L>>2)){
    col[0].val = invaa;
    col[0].idx = (ii<<1)-L;

    col[1].val = invbb;
    col[1].idx = (ii<<1)+1-L;
  }

  else if (ii>= (L>>1)+(L>>2)){
    col[0].val = invbb;
    col[0].idx = (ii<<1)-L;

    col[1].val = invcc;
    col[1].idx = (ii<<1)+1-L;
  } else {
    printf("problem in Ising transfer matrix\n");
  }

  return 0;
}


int main() {
  size_t Nit = 1000;      // number of iterations after burn in
  size_t Brn = 100;      // number of burn in iterations (these
                         // are not included in trajectory averages)
  size_t m = 1000000;      // compression parameter (after compression vectors have
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

  // Initialize timings.
  clock_t start, end;
  double cpuTime;
  
  // Run Brn iterations without accumulating trajectory
  // averages to discard initial transient.
  double lambda;
  for (size_t t = 0; t < Brn; t++) {

    // Perform and time a compression and multiplication.
    start = clock();
    compressor.compress(v, m);
    sparse_gemv(1.0, transfer, bw, v, 0.0, vnew);
    end = clock();

    // Finish the iteration.
    lambda = vnew.norm();
    normalize(vnew);
    v = vnew;

    // Print an iterate summary.
    printf("burn: %ld\t lambda: %lf\t nonzeros: %ld\t time / iteration: %lf\n",
       t+1, lambda, v.curr_size_, ((double)end - start)/CLOCKS_PER_SEC);
  }

  // Generate a trajectory of Nit iterations and 
  // accumulate trajectory averages.
  double lambda_ave = 0;
  for (size_t t = 0; t < Nit; t++){

    // Perform and time a compression and multiplication.
    start = clock();
    compressor.compress(v, m);
    sparse_gemv(1.0, transfer, bw, v, 0.0, vnew);
    end = clock();

    // Finish the iteration.
    lambda = vnew.norm();
    double eps = 1.0 / (1.0 + t);
    lambda_ave = (1.0 - eps) * lambda_ave + eps * lambda;
    normalize(vnew);
    v = vnew;

    // Print an iterate summary.
    printf("iteration: %ld\t lambda: %lf\t average: %lf\t nonzeros: %ld\t time / iteration: %lf\n",
       t+1, lambda, lambda_ave, v.curr_size_, ((double)end - start)/CLOCKS_PER_SEC);
  }

  return 0;  
}
