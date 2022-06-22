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
int RW1dcolumn(SparseVector<long, double> &col, const long jj, const size_t d){

  assert(2<=col.max_size_);

  if(jj>0 and jj<d-1){
    col.set_entry(jj-1,0.5);
    col.set_entry(jj+1,0.5);
  }
  if(jj==0 or jj==d-1){
    col.set_entry(jj,1.0);
  }

  return 0;
}


int main() {
  size_t d = 10;         // full dimension 
  size_t Nspls = 1<<0;      // number of independent samples of the estimator to generate
  size_t Nit = (size_t)floor(7);      // number of iterations after burn in
  size_t m = 4;      // compression parameter (after compression vectors have
                         // no more than m non-zero entries)
  size_t bw = 2;         // upper bound on the number of entries in any
                         // column or row of matrix
  size_t seed = 0;        // seed for RNG.  Can choose random seet below.

  // Initialize iterate vectors and submatrix.
  SparseMatrix<long, double> A(bw*m,m,bw);
  SparseVector<long, double> b(2);
  SparseVector<long, double> f(2);
  SparseVector<long, double> y(2*bw*m);
  SparseVector<long, double> z(d);
  SparseVector<long, double> x(d+bw*m);
  SparseVector<long, double> yave(d+bw*m);

  std::vector<bool> preserve(bw*m);
  std::vector<bool> Apres(bw*m);
  size_t npres;
  std::vector<double> col_norms(m);
  std::valarray<double> col_budgets(m);
  
  // Initialize a seeded random compressor.
  std::random_device rd;
  std::mt19937_64 generator;
  generator = std::mt19937_64(seed);
  //seed = rd();
  Compressor<long, double, std::mt19937_64> compressor(bw * m, generator);

  // Initialize timings.
  clock_t start, end;
  double cpuTime;

  // b is indicator on center point
  long center = (long)floor(d/2.0);
  if( 2*center == d ){
    b.set_entry(center-1,0.5);
    b.set_entry(center,0.5);
  }
  else{
    b.set_entry(center,1.0);
  }

  // we'll measure the error in a dot product.
  double finst, ftrue, fbias=0, fvar=0, sinst, sbias=0, svar=0;
  f.set_entry((long)0,-1.0);
  f.set_entry((long)d-1,1.0);
  ftrue = 0;

  // f.print();
  // b.print();

  // Generate Nspls independent samples of the estimatator of the Neumann sum
	start = clock();
  for (size_t spl = 0; spl<Nspls; spl++){
  	// Compute the Neumann sum up to Nit powers of G starting from b
  	x = b;
  	y = b;

    // compressor.compress(y,m);

  	for (size_t jj=0; jj<Nit; jj++){

      // y.print();

      sparse_colwisemv(RW1dcolumn, d, bw, y, A);
    	A.row_sums(y);
      x += y;

      std::cout<<std::endl;
      std::cout<<"iteration: "<<jj<<std::endl;
      // std::cout<<"before compression: "<<std::endl;
      // y.print();

      // npres = compressor.preserve(y, m, preserve);
      // if(npres>=0){
      //   std::cout<<"iteration: "<<jj<<"  npres: "<<npres<<std::endl;
      //   y.print();
      //   std::cout<<"y sum: "<<y.sum()<<std::endl;
      //   std::cout<<std::endl;
      // }

      // double y_sum = y.sum();

      // if ( npres < y.size() ){
      //   for (size_t ii=0; ii<y.size(); ii++){
      //     if( preserve[ii]==true ){
      //       A.set_all_row_values(ii,0);
      //     }
      //     else{
      //       y.set_value(ii,0);
      //     }
      //   }



      //   // if(npres>=0){
      //   //   A.print_ccs();
      //   //   y.print();
      //   // }

      //   // y.print();
      //   // A.print_ccs();

      //   // A.row_sums(z);
      //   // y_sum = z.sum();

      //   // compressor.col_assign_abs(A);

      //   // A.row_sums(z);

      //   // if(abs(y_sum-z.sum())>1e-6){
      //   //   std::cout<<y_sum<<" "<<z.sum()<<std::endl;
      //   //   A.print_ccs();
      //   //   z.print();
      //   //   assert(0);
      //   // }

      //   compressor.col_sample_pres(A, m-npres);

      //   A.row_sums(z);

      //   // z.print();

      //   y+=z;

      //   // if(npres>0){
      //   //   A.print_ccs();
      //   //   y.print();
      //   //   std::cout<<y.sum()<<std::endl;
      //   // }
      // }

      // std::cout<<"pass1"<<std::endl;
      // compressor.col_assign_abs(A);
      // std::cout<<"pass2"<<std::endl;

      // if(npres>0){
      //   A.print_ccs();
      // }

      A.print_ccs();
      y.print();

      // std::cout<<y.size()<<"  "<<A.nrows()<<std::endl;


      npres = compressor.preserve(y, m, preserve);

      // std::cout<<std::endl;
      // for(size_t jj=0; jj<y.size(); jj++){
      //   std::cout<<y[jj].idx<<"  "<<preserve[jj]<<std::endl;
      // }

      // std::cout<<std::endl;
      // std::cout<<m<<"  "<<npres<<"  "<<m-npres<<std::endl;

      compressor.col_sample_preserve2(A,m, preserve);
      A.row_sums(y);
      y.remove_zeros();

      // std::cout<<y.size()<<std::endl;

      // A.print_ccs();
      // y.print();

      // if(npres>=0){
      //   A.print_ccs();
      //   y.print();
      //   std::cout<<"y sum: "<<y.sum()<<std::endl;
      //   std::cout<<std::endl;
      // }

      // std::cout<<"after compression: "<<std::endl;
      // y.print();
  	}

  	// Update the estimates of the dot product and variance.
    sinst = y.sum();
    sbias += sinst-1.0;
    svar += (sinst-1.0)*(sinst-1.0);
    finst = y.dot(f);
    fbias += (finst-ftrue);
    fvar += (finst-ftrue)*(finst-ftrue);

    yave += y;
	}
	end = clock();

  sbias /= (double)Nspls;
  svar /= (double)(Nspls-1);

  fbias /= (double)Nspls;
  fvar /= (double)(Nspls-1);

  yave /= (double)Nspls;

  y.print();
  yave.print();
  std::cout<<yave.sum()<<std::endl;

  // Print result.
  //printf("%le\n",l1bias);
  printf("Sum bias and standard deviation after %lu samples of %lu iterations:  %le +/- %le, \t %le\n",
    Nspls, Nit, sbias,2*sqrt(svar/(double)Nspls), sqrt(svar));
  printf("Dot product bias and standard deviation after %lu samples of %lu iterations:  %le +/- %le, \t %le\n",
    Nspls, Nit, fbias,2*sqrt(fvar/(double)Nspls), sqrt(fvar));
  printf("Time elapsed:  %lf\n",((double)end - start)/CLOCKS_PER_SEC/60.0);


  return 0;  
}
