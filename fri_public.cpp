// A simple and somewhat generic implementation of Fast
// Randomized Iteration (FRI) as introduced by
// Lim and Weare.
// Meant for educational purposes.
// If you use this code please cite it by its
// DOI on Github and Zenodo; find it in the README 
// of www.github.com/jonathanweare/Fast-Randomized-Iteration-FRI- .
// (c) Jonathan Weare 2015

#include <iostream> 
#include <cstdlib>
#include <cmath>
#include <random>
#include <utility>
#include <algorithm>
#include "fri_public.h"

using namespace std;




// sparse_daxpy computes y <-- alpha x + y where x and y are sparse vectors with real valued entries
// y is assumed to be the first ny entries of an array of length nx + ny.  y is overwritten
// on output.
int sparse_daxpy(double alpha,
		 spentry<double> *x, long *nx,
		 spentry<double> *y, long *ny)
{

  long jj, cy = *nx, nw=0;

  // shift the entries of y to the end of its array
  for (jj=*ny-1;jj>-1;jj--)
      y[jj+*nx] = y[jj];

  *ny += *nx;

  // add in entries of either x or y with smallest indices
  for (jj=0;jj<*nx;jj++){
    while (cy<*ny && y[cy].loc<=x[jj].loc){
      y[nw] = y[cy];
      cy++;
      nw++;
    }
    if (nw>0 && x[jj].loc==y[nw-1].loc){
      y[nw-1].val += alpha*x[jj].val;
    }
    else {
      y[nw].loc = x[jj].loc;
      y[nw].val = alpha*x[jj].val;
      nw++;
    }
  }

  while (cy<*ny){
    y[nw] = y[cy];
    cy++;
    nw++;
  }

  *ny = nw;
  
  return 0;
}


// sparse_dgemv computes y <-- alpha A x + beta y where x and y are sparse vectors with real entries
// and the matrix A is specified by a routine Acolumn that returns a single column of A.  bw is an
// upper bound on the number of non-zero entries in any column of A.  y is
// assumed to be the first ny entries in an array of length ny + bw nx.  y is overwritten upon output.
int sparse_dgemv(double alpha,
		int (*Acolumn)(spentry<double> *col, long *nrows, long jj),
		long bw, spentry<double> *x, long *nx, double beta,
		spentry<double> *y, long *ny)
{
  long jj, ii, nw, na1;

  spentry<double> *a1;
  a1 = (spentry<double>*)malloc(bw*sizeof(spentry<double>));
  if (!a1) {
    std::cerr << "Memory cannot be allocated\n";
    return 1; //raise MemoryError()
  }

  if (beta != 0){
    for (ii=0;ii<*ny;ii++)
      y[ii].val *= beta;
    nw = *ny;
  }else{
    nw = 0;
  }

  // make a list of all the entries in A scaled by the entry of x corresponding to the column
  // containing the particular entry of A.
  for (jj=0;jj<*nx;jj++){
    Acolumn(a1,&na1,x[jj].loc);
    for(ii=0;ii<na1;ii++){
      y[nw].val = x[jj].val*a1[ii].val;
      y[nw].loc = a1[ii].loc;
      nw++;
    }
  }

  // sort the list according to their indices
  heapsort(y,nw,spcomparebyloc);

  // sum values corresponding to like indices and collapse the list so the indices are unique.
  *ny = 0;
  jj = 0;
  while (jj < nw){
    y[*ny] = y[jj];
    ii = jj+1;
    while (ii<nw && y[ii].loc == y[*ny].loc){
      y[*ny].val += y[ii].val;
      ii++;
    }
    jj = ii;
    (*ny)++;
  }

  free(a1);

  return 0;
}





// dscompatmult first compresses (with parameter m) the input vector x and then calls
// sparse_dgemv
int dspcompmatmult(long m, double alpha,
		  int (*Acolumn)(spentry<double> *col, long *nrows, long jj),
		  long bw, spentry<double> *x, long *nx,double beta,
		  spentry<double> *y, long *ny)
{
  
  long jj, nx2 = 0;
  
  double *xcopy;
  xcopy = (double*)malloc((*nx)*sizeof(double));
  if (!xcopy) {
    std::cerr << "Memory cannot be allocated\n";
    return 1; //raise MemoryError()
  }

  // store the modulus of the values of x in xcopy
  for (jj=0;jj<*nx;jj++)
    xcopy[jj] = fabs(x[jj].val);

  // spentry<double> *xabs;
  // xabs = (spentry<double>*)malloc((*nx)*sizeof(spentry<double>));
  // if (!xabs) {
  //   std::cerr << "Memory cannot be allocated\n";
  //   return 1; //raise MemoryError()
  // }

  // for (jj=0;jj<*nx;jj++){
  //   xabs[jj].val = fabs(x[jj].val);
  //   xabs[jj].loc = jj;
  // }

  // compress xcopy, i.e. randomly set entris of x to zero using the compress routine
  compress( xcopy , *nx , m);

  // write the resulting compressed vector back into x, omitting entries that have been set to zero.
  for(jj=0;jj<*nx;jj++){
    if(xcopy[jj]!=0){
      if(x[jj].val>0)
	x[nx2].val = xcopy[jj];
      else if (x[jj].val<0)
	x[nx2].val = -xcopy[jj];
      else
	printf("why does x have zero entries\n");
      x[nx2].loc = x[jj].loc;
      nx2++;
    }
  }

  *nx = nx2;

  // apply matrix vector multiplication with compressed x in place of original x
  sparse_dgemv(alpha, Acolumn, bw, x, nx, beta, y, ny);
	  
  free(xcopy);

  return 0;
}




// a standard max heapsort
template <typename Type>
int heapsort(Type *base, long L, int (*compr)(Type& a, Type& b))
{

  long jj;
  
  heapify(base, L, compr);

  for(jj=0;jj<L-1;jj++)
    pullroot(base,L-jj,compr);

  return 0;
}

// rearrange the entries in base so that they form a max heap.  comparison is generic and is
// handled by the compr input function.
template <typename Type>
int heapify(Type *base, long L, int (*compr)(Type& a, Type& b))
{
  
  long mm;
  
  if ((L & 1) == 0)
    mm = (L-2)>>1;
  else
    mm = (L-1)>>1;
        
  while (mm>=0) {
    siftdown(base, mm, L, compr);
    mm-=1;
  }
        
  return 0;
}

// puts the largest entry in base (the root) at the end of base and re-heapifies the rest of the entries
template <typename Type>
int pullroot(Type *base, long L, int (*compr)(Type& a, Type& b))
{

  std::swap( base[0], base[L-1] );
  
  siftdown(base,0,L-1, compr);
    
  return 0;
}

// a standard max siftdown
template <typename Type>
int siftdown(Type *base, long start, long end, int (*compr)(Type& a, Type& b))
{
    
  long root;
  long swap;
  long jj;
    
  root = start;
  jj = (root<<1)+1;

  while (jj<end) {
    swap = root;
    if (compr( base[swap], base[jj] )<0)
      swap = jj;
    
    jj+=1;

    if ( (jj<end) && compr( base[swap], base[jj] )<0)
      swap = jj;
    
    if (swap != root) {
      std::swap( base[swap], base[root] );
      root = swap;
      jj = (root<<1)+1;
    } else
      jj = end;
  }
  
  return 0; 
}

// use as compr argument when you want to sort an array of spentry elements according to the
// element indices.
template <typename Type>
int spcomparebyloc( Type& a, Type& b){

  if (a.loc<b.loc) return -1;

  if (a.loc>b.loc) return 1;
  
  return 0;
}


// use as compr argument when you want to sort an array of spentry elements according to the
// element values.  If spentry values use a custom type make sure < and > are overloaded for that type.
template <typename Type>
int spcomparebyval( Type& a, Type& b){

  if (a.val<b.val) return -1;

  if (a.val>b.val) return 1;
  
  return 0;
}



// compress a vector of real entries of length L.  n is the compression parameter.
// for this scheme the compressed vector will have n or fewer non-zero entries.
int compress( double *y , long L , long n) {
  
  double nrm1, r;
  long ii, nz;
    
  double Tol = 1e-15;
  static std::random_device rd;
  static std::mt19937_64 gen(rd());
  static std::uniform_real_distribution<> uu(0, 1);

  // first make sure y really has more than n non-zero entries
  nrm1 = 0;
  nz = L;
  for (ii=0;ii<L;ii++) {
    if (y[ii]<0) {
      std::cerr << "In compressHybrid y argument should have positive entries\n";
      return 1;
    }
    else if (y[ii]==0)
      nz -= 1;
    else
      nrm1 += y[ii];
  }
    
  if (nz<=n)
    return 0;
    
  long jj, kk, mm, imax;
  double w, dmax;

  // this vector of spentry elements is just to keep track of some element rearrangement that
  // occurs below.
  spentry<double>* yheap;
  yheap = (spentry<double>*)malloc(L*sizeof(spentry<double>));
  if (!yheap) {
    std::cerr << "Memory cannot be allocated\n";
    return 1; //raise MemoryError()
  }  
    
  for (ii=0;ii<L;ii++){
    yheap[ii].val = y[ii];
    yheap[ii].loc = ii;
  }
        
  dmax = 0;
  for (ii=0;ii<L;ii++)
    if (y[ii]>dmax) {
      imax = ii;
      dmax = y[ii];
    }

  std::swap( yheap[L-1], yheap[imax]);

  mm = 0;
  r = nrm1;

  // check if there are any elements large enough to be preserved exactly.  If so heapify
  // and pull large entries untill remaining entries are not large enough to be preserved exactly.
  if (n*dmax>=r) {
    mm = 1;
    r -= dmax;
    heapify(yheap,L-1,spcomparebyval);
    
    while (( (n-mm)*yheap[0].val>=r) && (mm<n)) {
      r -= yheap[0].val;
      pullroot(yheap,L-mm,spcomparebyval);
      mm++;
    }
  }

  // compute the norm of the vector of remaining small entries
  r = 0;
  for (ii=0;ii<L-mm;ii++)
    r += yheap[ii].val;

  // compress remaining small entries.  note that for small entries, entries that are not set to zero
  // are set to the norm of the vector of remaining small entries divided by the difference between n
  // and the number of large entries preserved exactly.
  if ((mm<n) && (r>nrm1*Tol)) {
    nz = mm;
    w = -uu(gen); //-genrand64_real3()
    jj = 0;
    ii = 0;
    while ( (nz<n) && (ii<L-mm) ) {
      w += ((n-mm)*yheap[ii].val)/r;
      kk=jj;
      while (jj <= (long)floor(w)) 
	jj++;
	    
      if (jj==kk)
	yheap[ii].val = 0;
      else {
	nz += 1;
	yheap[ii].val = r/(double)(n-mm);
	if (jj-kk>1)
	  std::cerr << "Too big\n";
      }
      ii++;
    }	
    while (ii<L-mm){
      yheap[ii].val = 0;
      ii++;
    }
  }
  else 
    for (ii=0;ii<L-mm;ii++)
      yheap[ii].val=0;
  

  // read the result back into y
  for(ii=0;ii<L;ii++)
    y[yheap[ii].loc] = yheap[ii].val;
  
    
  nz = 0;
  for (ii=0;ii<L;ii++)
    if (y[ii]>0)
      nz += 1;
                    
  if (nz>n)
    std::cerr << "Too many nonzeros in compress\n"; //', nz,n,mm, nrm1, nrm1-r, w, L
      
  free(yheap);

  return 0;
  
}
