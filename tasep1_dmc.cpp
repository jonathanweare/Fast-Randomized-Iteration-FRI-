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

using namespace std;

template< size_t N>
bool operator<(const bitset<N>& x, const bitset<N>& y)
{
    for (int i = N-1; i >= 0; i--) {
        if (x[i] ^ y[i]) return y[i];
    }
    return false;
}

#include "fri_2.h"

const size_t N = 100;
const size_t M = 50;
const double alpha = 0.005;
const double UU = -2.0;

typedef bitset<N> multiindex;

// A subroutine that returns the ii-th column of
// the transfer matrix of the 2D Ising model.
int column(SparseVector<multiindex, double> &col, const multiindex ii){
  
  size_t jj;
  size_t X=0;
  double eUU = exp(UU);

  for(jj=0;jj<N-1;jj++){
    if( ii[jj] & !ii[jj+1] ){
      col[X].idx = ii;
      col[X].idx[jj]=false;
      col[X].idx[jj+1]=true;
      X++;
    } 
  }
  if( ii[N-1] & !ii[0] ){
    col[X].idx = ii;
    col[X].idx[N-1]=false;
    col[X].idx[0]=true;
    X++;
  }

  for(jj=0;jj<X;jj++)
    col[jj].val = alpha*eUU/M;

  col[X].idx = ii;
  col[X].val = 1.0 - (double)X*alpha/M;

  if ( col[X].val >0 ){
    col.curr_size_ = X+1;
  } else{
    col.curr_size_= X;
  }
 
  assert( (double)abs(1 + X*alpha*(eUU-1.0)/M-col.norm())<1e-12 );

  return 0;
}


int main() {
  const size_t Nit = 1000000;      // number of iterations
  const size_t m = 1000;      // compression parameter (after compression vectors have
                         // no more than m non-zero entries)
  const size_t bw = M+1;         // upper bound on the number of entries in each
                         // column of matrix

  // Initialize iterate vectors.
  SparseVector<multiindex, double> v(2*m);
  SparseVector<multiindex, double> vnew(2*m);
  SparseVector<multiindex, double> vmid(2*m);
  vmid.curr_size_ = 1;
  vnew.curr_size_ = 1;
  // Set initial m initial walkers with equal position
  v.curr_size_ = m;
  v[0].val = 1.0;
  v[0].idx = 0;
  // for(size_t jj=0;jj<M;jj++) v[0].idx[2*jj]=true;
  for(size_t jj=0;jj<M;jj++) v[0].idx[jj]=true;
  for(size_t ii=0;ii<m;ii++) v[ii] = v[0];

  // Inialize random number generator
  std::mt19937_64 gen_;
  std::uniform_real_distribution<> uu_;

  // Initialize timings.
  clock_t start, end;
  double cpuTime;

  // Generate a trajectory of Nit iterations and 
  // accumulate trajectory averages.
  double lambda, lambda_ave = 0, w, U, column_sum;
  size_t kk;
  for (size_t t = 0; t < Nit; t++){

    // Perform and time a compression and multiplication.
    start = clock();

    // For each walker sample a new position according to the normalized columns.
    SparseVector<multiindex, double> single_row_by_column_adds(bw);
    for (size_t jj = 0; jj < v.curr_size_; jj++) {
      column(single_row_by_column_adds, v[jj].idx);
      column_sum = single_row_by_column_adds.norm();
      U = column_sum*uu_(gen_);
      w=0;
      kk = -1;
      while (w<U){
        kk++;
        w += single_row_by_column_adds[kk].val;
      }
      assert( (kk>=0) & (kk<single_row_by_column_adds.curr_size_) );
      // Set the weight of each walker equal to the column sum.
      v[jj].val = column_sum;
      v[jj].idx = single_row_by_column_adds[kk].idx;
    }
    

    // Resample the walkers according to the column sums so that expected number of walkers is m.
    // double total_sum = v.norm();
    // size_t n_entry, n_entry_adds = 0;
    // for (size_t jj=0;jj<v.curr_size_;jj++){
    //   n_entry = (size_t)floor(m*v[jj].val/total_sum+uu_(gen_));
    //   for( size_t ii=0; ii<n_entry; ii++){
    //     vnew[n_entry_adds].val = 1.0;
    //     vnew[n_entry_adds].idx = v[jj].idx;
    //     n_entry_adds++;
    //   }
    // }
    // vnew.curr_size_ = n_entry_adds;



    // Resample the walkers uniformly so that expected number of walkers is m.
    size_t n_entry, n_entry_adds = 0;
    for (size_t jj=0;jj<v.curr_size_;jj++){
      n_entry = (size_t)floor((double)m/v.curr_size_+uu_(gen_));
      for( size_t ii=0; ii<n_entry; ii++){
        vmid[n_entry_adds].val = v[jj].val;
        vmid[n_entry_adds].idx = v[jj].idx;
        n_entry_adds++;
      }
    }
    vmid.curr_size_ = n_entry_adds;

    // Resample the walkers according to the column sums with no additional control on population size.
    n_entry_adds = 0;
    double total_sum = v.norm();
    for (size_t jj=0;jj<vmid.curr_size_;jj++){
      n_entry = (size_t)floor(vmid[jj].val+uu_(gen_));
      for( size_t ii=0; ii<n_entry; ii++){
        vnew[n_entry_adds].val = 1.0;
        vnew[n_entry_adds].idx = vmid[jj].idx;
        n_entry_adds++;
      }
    }
    vnew.curr_size_ = n_entry_adds;



    end = clock();

    // Finish the iteration.
    lambda = (total_sum/v.curr_size_ - 1.0 + alpha)/alpha;
    double eps = 1.0 / (1.0 + t);
    lambda_ave = (1.0 - eps) * lambda_ave + eps * lambda;
    normalize(vnew);
    v = vnew;

    // Print an iterate summary.
    printf("iteration: %ld\t lambda: %lf\t average: %lf\t nonzeros: %ld\t time / iteration: %lf\n",
       t+1, log(lambda), log(lambda_ave), v.curr_size_, ((double)end - start)/CLOCKS_PER_SEC);
  }

  // cout << "\n";
  // for(size_t ii=0; ii<v.curr_size_;ii++){
  //   cout << ii << "\t" << v[ii].val*m << "\t" << v[ii].idx << "\n";
  // }


  return 0;  
}
