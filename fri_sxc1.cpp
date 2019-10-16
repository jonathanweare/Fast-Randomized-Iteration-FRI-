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
int A1column(SparseVector<long, double> &col, const long jj, const size_t d){

  assert(d<=col.max_size_);

  double val;

  for(size_t ii=0;ii<d;ii++){
    val = ( 1.12-0.72*((double)ii)/(double)d )*(1.12-0.72*((double)jj)/(double)d)/(double)d;
    col.set_entry((long)ii,val);
  }

  return 0;
}


int main() {
  size_t d = 200;         // full dimension 
  size_t Nspls = 1<<0;      // number of independent samples of the estimator to generate
  size_t Nit = 2;      // number of iterations after burn in
  size_t m = 50;      // compression parameter (after compression vectors have
                         // no more than m non-zero entries)
  size_t bw = d;         // upper bound on the number of entries in each
                         // column of matrix
  size_t seed = 1;        // seed for RNG.  Can choose random seet below.

  // Initialize iterate vectors and submatrix.
  SparseMatrix<long, double> A(d,d);
  SparseVector<long, double> xtrue(d);
  SparseVector<long, double> b(d);
  SparseVector<long, double> y(2*d);
  SparseVector<long, double> z(d);
  SparseVector<long, double> x(2*d);
  SparseVector<long, double> bias(2*d);

  std::vector<bool> preserve(d);
  SparseVector<long, double> col_norms(m);
  std::valarray<double> col_budgets(m);

  for(size_t jj=0;jj<d;jj++){
  	bias.set_entry((long)jj,(double)0);
  }
  
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

  // and the error in a dot product.
  double finst, ftrue, fbias=0, fvar=0;

  //If you want you can build and print the whole matrix.
  // for(size_t jj=0; jj<d; jj++){
  //   x.set_entry((long)jj,1.0);
  // }
  // A.sparse_colwisemv(A1column, d, bw, x, A);
  //A.print_ccs();

  // The true solution vector xtrue
  for(size_t jj=0; jj<d; jj++){
    xtrue.set_entry((long)jj,1.0/(2.25-1.45*((double)jj)/(double)d));
  }

  // b = (I-G)*xtrue
  sparse_colwisemv(A1column, d, bw, xtrue, A);
  A.row_sums(b);
  x = xtrue;
  x += b;
  b = x;

  // compute the true Neumann sum up to Nit powers of G starting from b
  x = b;
  y = b;

	for (size_t jj=0; jj<Nit; jj++){
    sparse_colwisemv(A1column, d, bw, y, A);
    A.row_sums(y);
    x += y;
  }
  xtrue = x;
  ftrue = xtrue.sum();

  // Generate Nspls independent samples of the estimatator of the Neumann sum
	start = clock();
  for (size_t spl = 0; spl<Nspls; spl++){
  	// Compute the Neumann sum up to Nit powers of G starting from b
  	x = b;
  	y = b;


    compressor.compress(y, m);
  	for (size_t jj=0; jj<Nit; jj++){
      sparse_colwisemv(A1column, d, bw, y, A);
    	A.row_sums(y);
      x += y;

      compressor.preserve(y, m, preserve);

      for (size_t ii=0; ii<y.size(); ii++){
        //std::cout<< ii<<" "<<preserve[ii]<<std::endl;
        if( preserve[ii]==false ){
          size_t nummax =0;
          double valmax = abs((A.get_row_entry(ii,0)).val);
          for (size_t kk=1; kk<A.row_size(ii); kk++){
            if ( abs((A.get_row_entry(ii,kk)).val)>valmax ){
              A.set_row_value(ii,nummax,0);
              valmax = abs((A.get_row_entry(ii,0)).val);
              nummax = kk;
            }
            else{
              A.set_row_value(ii,kk,0);
            }
          }
          A.set_row_value(ii,nummax,y[ii].val);
          y.set_value(ii,0);
        }
        else{
          for (size_t kk=0; kk<A.row_size(ii); kk++){
            A.set_row_value(ii,kk,0);
          }
        }
      }
      // A.print_ccs();
      // y.print();

      y.remove_zeros();

      A.col_norms(col_norms);
      // col_norms.print();

      for(size_t ii=0; ii<m; ii++){
        col_budgets[ii] = col_norms[ii].val;
      }
      resample_piv(col_budgets, m, &generator);

      for(size_t ii=0; ii<A.ncols(); ii++){
        // std::cout << ii<<" "<<col_budgets[ii]<<std::endl;
        if( col_budgets[ii]>0 ){
          A.get_col(ii,z);
          compressor.compress(z,(size_t)col_budgets[ii]);
          A.set_col(z,y[ii].idx);
        }
      }
      A.row_sums(z);

      y+=z;
  	}

  	// Update the bias vector and compute the l2 error of the approximate solution.
    x -= xtrue;
    bias += x;
    finst = x.sum();
    fbias += finst;
    fvar += finst*finst;

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
  fbias /= (double)Nspls;
  fvar /= (double)Nspls;

  // Print result.
  //printf("%le\n",l1bias);
  printf("Bias and standard deviation after %lu samples of %lu iterations:  %le +/- %le, \t %le\n",
    Nspls, Nit, fbias,2*sqrt(fvar/(double)Nspls), sqrt(fvar));
  printf("Time elapsed:  %lf\n",((double)end - start)/CLOCKS_PER_SEC);


  return 0;  
}
