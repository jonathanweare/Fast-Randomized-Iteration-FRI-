// A driver to run Fast Randomized Iteration
// applied to computing the dominant
// eigenvalue (the partition function)
// of the transfer matrix associated with
// the 2D Ising model
// (c) Jonathan Weare 2015

#include <iostream>
#include <cstdlib>
#include <math.h>
#include <bitset>
#include "fri_2.h"
// #include "fri_index.h"

using namespace std;

const int N = 2;
const int M = 2;
const double tt = 1.0;
const double UU = 1.0;

// A subroutine that returns the ii-th column of
// the transfer matrix of the 2D Ising model.

// typedef std::bitset<2*N> multiindex;

template<std::size_t N>
bool operator<(const std::bitset<N>& x, const std::bitset<N>& y)
{
    for (int i = N-1; i >= 0; i--) {
        if (x[i] ^ y[i]) return y[i];
    }
    return false;
}


// template<std::size_t N>
// bool operator<(const std::bitset<N>& x, const std::bitset<N>& y)
// {
//     for (int i = N-1; i >= 0; i--) {
//         if (x[i] ^ y[i]) return y[i];
//     }
//     return false;
// }

template<std::size_t N>
int column(SparseVector<std::bitset<N>, double> &col, const std::bitset<N> ii){

  return 0;
}




// int column(SparseVector<multiindex, double> &col, const multiindex ii){
  
//   int n = 50;
//   double BB = 0.01;
//   double TT = 2.2;

//   double aa, bb, cc, invaa, invbb, invcc;
//   long jj, L;
  
//   col.curr_size_ = 2;

//   for(jj=0;jj<col.curr_size_;jj++)
//     col[jj].idx.clear(); 

//   aa = exp((2.0-BB)/TT);
//   bb = exp(-BB/TT);
//   cc = exp((-2.0-BB)/TT);
//   invaa = exp(-(2.0-BB)/TT);
//   invbb = exp(BB/TT);
//   invcc = exp((2.0+BB)/TT);

//   L = (long)1<<n;
  
//   if (ii[0] < (L>>2)){
//     col[0].val = aa;
//     col[0].idx.push_back(ii[0]<<1);

//     col[1].val = bb;
//     col[1].idx.push_back((ii[0]<<1)+1);
//   }


//   else if (ii[0]>= (L>>2) && ii[0]< (L>>1)){
//     col[0].val = bb;
//     col[0].idx.push_back(ii[0]<<1);

//     col[1].val = cc;
//     col[1].idx.push_back((ii[0]<<1)+1);
//   }

//   else if (ii[0]>=(L>>1) && ii[0]< (L>>1)+(L>>2)){
//     col[0].val = invaa;
//     col[0].idx.push_back((ii[0]<<1)-L);

//     col[1].val = invbb;
//     col[1].idx.push_back((ii[0]<<1)+1-L);
//   }

//   else if (ii[0]>= (L>>1)+(L>>2)){
//     col[0].val = invbb;
//     col[0].idx.push_back((ii[0]<<1)-L);

//     col[1].val = invcc;
//     col[1].idx.push_back((ii[0]<<1)+1-L);
//   } else {
//     printf("problem in Ising transfer matrix\n");
//   }

//   return 0;
// }


int main() {
  size_t Nit = 100;      // number of iterations after burn in
  size_t Brn = 100;      // number of burn in iterations (these
                         // are not included in trajectory averages)
  size_t m = 32768;      // compression parameter (after compression vectors have
                         // no more than m non-zero entries)
  size_t bw = 2;         // upper bound on the number of entries in each
                         // column of matrix

  // Initialize iterate vectors.
  SparseVector<std::bitset<2*N>, double> v(bw * m);
  SparseVector<std::bitset<2*N>, double> vnew(bw * m);
  vnew.curr_size_ = 1;
  v.curr_size_ = 1;                                                    // initial vector
  v[0].val = 1.0;
  for (int ii=0;ii<N;ii++)
    v[0].idx[ii] = true;
  normalize(v);
  // Initialize a seeded random compressor.
  std::random_device rd;
  Compressor<std::bitset<2*N>, double> compressor(bw * m, rd());


  std::bitset<4> a (5);
  std::cout << a << "\n";

  std::bitset<4> b (6);
  std::cout << b << "\n";

  std::cout << (a<b) << "\n";

  exit(1);

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
    sparse_gemv(1.0, column, bw, v, 0.0, vnew);
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
    sparse_gemv(1.0, column, bw, v, 0.0, vnew);
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
