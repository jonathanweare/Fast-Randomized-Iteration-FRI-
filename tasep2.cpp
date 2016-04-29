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

// #include "fri_index.h"

using namespace std;


// A subroutine that returns the ii-th column of
// the transfer matrix of the 2D Ising model.

// typedef std::bitset<2*N> multiindex;

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
const double alpha = 1.0/M;
const double S = -2.0/N;

//typedef bitset<numeric_limits<unsigned long long>::digits> multiindex;
typedef bitset<N> multiindex;

// template<std::size_t N>
// bool operator<(const std::bitset<N>& x, const std::bitset<N>& y)
// {
//     for (int i = N-1; i >= 0; i--) {
//         if (x[i] ^ y[i]) return y[i];
//     }
//     return false;
// }


int column(SparseVector<multiindex, double> &col, const multiindex ii){
  
  size_t jj;
  size_t X=0;
  double eS = exp(S);
  
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
    col[jj].val = alpha*eS;

  col[X].idx = ii;
  col[X].val = 1.0 - X*alpha;

  if ( col[X].val >0 ){
    col.curr_size_ = X+1;
  } else{
    col.curr_size_= X;
  }

  return 0;
}


int main() {

  const size_t Nit = 100;      // number of iterations after burn in

  const size_t Brn = 0;      // number of burn in iterations (these
                         // are not included in trajectory averages)
  const size_t m = 10;      // compression parameter (after compression vectors have
                         // no more than m non-zero entries)
  const size_t bw = M+1;         // upper bound on the number of entries in each
                         // column of matrix

  // Initialize iterate vectors.
  SparseVector<multiindex, double> v(2*bw * m);
  SparseVector<multiindex, double> vnew(2*bw * m);
  vnew.curr_size_ = 1;
  v.curr_size_ = 1;                                                    // initial vector
  v[0].val = 1.0;
  v[0].idx = 0;
  // for(size_t jj=0;jj<M;jj++) v[0].idx[2*jj]=true;
  for(size_t jj=0;jj<M;jj++) v[0].idx[jj]=true;
  normalize(v);

  std::vector<size_t> col_locs(2*m+1);

  // Initialize a seeded random compressor.
  std::random_device rd;
  Compressor<multiindex, double> compressor(bw * m, rd());

  std::mt19937_64 gen_;
  std::uniform_real_distribution<> uu_;


  // std::bitset<4> a (5);
  // std::cout << a << "\n";

  // std::bitset<4> b (6);
  // std::cout << b << "\n";

  // std::cout << (a<b) << "\n";

  // exit(1);

  // Initialize timings.
  clock_t start, end;
  double cpuTime;
  
  // Run Brn iterations without accumulating trajectory
  // averages to discard initial transient.
  double lambda;
  size_t ii, jj, entries_added;
  for (size_t t = 0; t < Brn; t++) {

    // Perform and time a compression and multiplication.
    start = clock();
    sparse_gemv_cmp(1.0, column, bw, v, 0.0, vnew, col_locs);
    end = clock();

    // Finish the iteration.
    lambda = vnew.norm();
    normalize(vnew);
    v = vnew;

    // cout << "\n";
    // cout << col_locs.size() << "\n\n";

    // for(ii=0;ii<col_locs.size();ii++)
    //   cout << col_locs[ii] << "\n";
    // cout << "\n";

    entries_added = 0;
    for( ii=0; ii<col_locs.size()-1; ii++){
      // cout << ii << "\t" << v.curr_size_ << "\n";
      for (jj=0; jj<col_locs[ii+1]-col_locs[ii]; jj++){
        vnew[jj] = v[col_locs[ii]+jj];
        // cout << ii << "\t" << jj << "\t" << col_locs[ii] << "\t" << col_locs[ii+1] << "\n";
      }
      // cout << "\n";
      vnew.curr_size_ = col_locs[ii+1]-col_locs[ii];
      compressor.compress(vnew, (size_t)floor(m*vnew.norm()+uu_(gen_)));
      for( jj=0; jj<vnew.curr_size_; jj++){
        v[entries_added] = vnew[jj];
        entries_added++;
      }
    }
    v.curr_size_ = entries_added;

    // Print an iterate summary.
    printf("burn: %ld\t lambda: %lf\t nonzeros: %ld\t time / iteration: %lf\n",
       t+1, (lambda-1.0)/alpha, v.curr_size_, ((double)end - start)/CLOCKS_PER_SEC);
  }

  // Generate a trajectory of Nit iterations and 
  // accumulate trajectory averages.
  double lambda_ave = 0;
  for (size_t t = 0; t < Nit; t++){

    // Perform and time a compression and multiplication.
    start = clock();
    sparse_gemv_cmp(1.0, column, bw, v, 0.0, vnew, col_locs);
    end = clock();

    // Finish the iteration.
    lambda  = vnew.norm();
    double eps = 1.0 / (1.0 + t);
    lambda_ave = (1.0 - eps) * lambda_ave + eps * lambda;
    normalize(vnew);
    v = vnew;

    entries_added = 0;
    for( ii=0; ii<col_locs.size()-1; ii++){
      // cout << ii << "\t" << v.curr_size_ << "\n";
      for (jj=0; jj<col_locs[ii+1]-col_locs[ii]; jj++){
        vnew[jj] = v[col_locs[ii]+jj];
        // cout << ii << "\t" << jj << "\t" << col_locs[ii] << "\t" << col_locs[ii+1] << "\n";
      }
      // cout << "\n";
      vnew.curr_size_ = col_locs[ii+1]-col_locs[ii];
      compressor.compress(vnew, (size_t)floor(m*vnew.norm()+uu_(gen_)));
      cout << ii << "\t" << (size_t)floor(m*vnew.norm()) << "\n";
      for( jj=0; jj<vnew.curr_size_; jj++){
        v[entries_added] = vnew[jj];
        entries_added++;
      }
    }
    v.curr_size_ = entries_added;

    // Print an iterate summary.
    printf("iteration: %ld\t lambda: %lf\t average: %lf\t nonzeros: %ld\t time / iteration: %lf\n",
       t+1, (lambda-1.0)/alpha, (lambda_ave-1.0)/alpha, v.curr_size_, ((double)end - start)/CLOCKS_PER_SEC);
  }

  return 0;  
}
