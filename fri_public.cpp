#include <iostream> 
#include <cstdlib>
#include <cmath>
#include <random>
#include <utility>
#include <algorithm>
#include "fri_public.h"

using namespace std;


int sparse_daxpy(double alpha,
		 spentry<double> *x, long *nx,
		 spentry<double> *y, long *ny)
{

  long jj, cy = *nx, nw=0;

  for (jj=*ny-1;jj>-1;jj--)
      y[jj+*nx] = y[jj];

  *ny += *nx;

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

  for (jj=0;jj<*nx;jj++){
    Acolumn(a1,&na1,x[jj].loc);
    for(ii=0;ii<na1;ii++){
      y[nw].val = x[jj].val*a1[ii].val;
      y[nw].loc = a1[ii].loc;
      nw++;
    }
  }
  
  heapsort(y,nw,spcomparebyloc);
  
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

  compress( xcopy , *nx , m);

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
  
  sparse_dgemv(alpha, Acolumn, bw, x, nx, beta, y, ny);
	  
  free(xcopy);

  return 0;
}





template <typename Type>
int heapsort(Type *base, long L, int (*compr)(Type& a, Type& b))
{

  long jj;
  
  heapify(base, L, compr);

  for(jj=0;jj<L-1;jj++)
    pullroot(base,L-jj,compr);

  return 0;
}

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

template <typename Type>
int pullroot(Type *base, long L, int (*compr)(Type& a, Type& b))
{

  std::swap( base[0], base[L-1] );
  
  siftdown(base,0,L-1, compr);
    
  return 0;
}

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


template <typename Type>
int spcomparebyloc( Type& a, Type& b){

  if (a.loc<b.loc) return -1;

  if (a.loc>b.loc) return 1;
  
  return 0;
}


template <typename Type>
int spcomparebyval( Type& a, Type& b){

  if (a.val<b.val) return -1;

  if (a.val>b.val) return 1;
  
  return 0;
}





// int dShiftDown(double *key, long* heap, long start, long end) {
    
//   long root;
//   long swap;
//   long ii;
//   long jj; 
    
//   root = start;
//   jj = (root<<1)+1;

//   while (jj<end) {
//     swap = root;
//     if (key[heap[swap]]<key[heap[jj]])
//       swap = jj;
	
//     jj+=1;
	
//     if ( (jj<end) && (key[heap[swap]]<key[heap[jj]]) )
//       swap = jj;
	
//     if (swap != root) {
//       ii = heap[swap];
//       heap[swap] = heap[root];
//       heap[root] = ii;
//       root = swap;
//       jj = (root<<1)+1;
//     } else
//       jj = end;
//   }
            
//   return 0;
    
// }
    
// int dHeapify(double *key, long *heap, long L) {

//   long mm;
    
//   if ((L & 1) == 0)
//     mm = (L-2)>>1;
//   else
//     mm = (L-1)>>1;
        
//   while (mm>=0) {
//     dShiftDown(key, heap, mm, L);
//     mm-=1;
//   }
        
//   return 0;
// }
    
// int dPullRoot(double *key, long *heap, long L) {
  
//   long ii;
    
//   ii = heap[0];
//   heap[0] = heap[L-1];
//   heap[L-1] = ii;
//   dShiftDown(key,heap,0,L-1);
    
//   return 0;
// }

// int compress( double *y , long L , long n) {
  
//   double nrm1, r;
//   long ii, nz;
    
//   double Tol = 1e-15;
//   static std::random_device rd;
//   static std::mt19937_64 gen(rd());
//   static std::uniform_real_distribution<> uu(0, 1);

//   // gen.seed(10);
	
//   nrm1 = 0;
//   nz = L;
//   for (ii=0;ii<L;ii++) {
//     if (y[ii]<0) {
//       std::cerr << "In compressHybrid y argument should have positive entries\n";
//       return 1;
//     } else {
//       if (y[ii]==0)
// 	nz -= 1;
//       else
// 	nrm1 += y[ii];
//     }
//   }
    
//   if (nz<=n)
//     return 0;

//   // printf("pass\n");
    
//   long jj, kk, mm, imax;
//   double w, dmax;
    
//   long *heap;
//   heap = (long*)malloc(L*sizeof(long));
    
//   if (!heap) {
//     std::cerr << "Memory cannot be allocated\n";
//     return 1; //raise MemoryError()
//   }
    
//   for (ii=0;ii<L;ii++)
//     heap[ii] = ii;
        
//   dmax = 0;
//   for (ii=0;ii<L;ii++)
//     if (y[ii]>dmax) {
//       imax = ii;
//       dmax = y[ii];
//     }
        
//   heap[L-1] = imax;
//   heap[imax] = L-1;

//   mm = 0;
//   r = nrm1;
    
//   if (n*dmax>=r) {
//     mm = 1;
//     r -= dmax;
//     dHeapify(y,heap,L-1);
    
//     while (( (n-mm)*y[heap[0]]>=r) && (mm<n)) {
//       r -= y[heap[0]];
//       dPullRoot(y,heap,L-mm);
//       mm+=1;
//     }
//   }
            
//   r = 0;
//   for (ii=0;ii<L-mm;ii++)
//     r = r + y[heap[ii]];
        
//   if ((mm<n) && (r>nrm1*Tol)) {
//     nz = mm;
//     w = -uu(gen); //-genrand64_real3()
//     jj = 0;
//     ii = 0;
//     while ( (nz<n) && (ii<L-mm) ) {
//       w = w + ((n-mm)*y[heap[ii]])/r;
//       kk=jj;
//       while (jj <= (long)floor(w)) 
// 	jj += 1;
	    
//       if (jj==kk)
// 	y[heap[ii]] = 0;
//       else {
// 	nz += 1;
// 	y[heap[ii]] = r/(double)(n-mm);
// 	if (jj-kk>1) {
// 	  std::cerr << "Too big\n";
// 	  ii=ii; //print n, mm, r, w, (y[heap[ii]])/r, ii, jj, kk
// 	}
//       }
//       ii++;
//     }
	
//     while (ii<L-mm){
//       y[heap[ii]] = 0;
//       ii++;
//     }
//   } else {
//     for (ii=0;ii<L-mm;ii++)
//       y[heap[ii]]=0;
//   }
    
    
//   nz = 0;
//   for (ii=0;ii<L;ii++)
//     if (y[ii]>0)
//       nz += 1;
                    
//   if (nz>n)
//     std::cerr << "Too many nonzeros in compress\n"; //', nz,n,mm, nrm1, nrm1-r, w, L
      
//   free(heap);

//   return 0;
  
// }




int compress( double *y , long L , long n) {
  
  double nrm1, r;
  long ii, nz;
    
  double Tol = 1e-15;
  static std::random_device rd;
  static std::mt19937_64 gen(rd());
  static std::uniform_real_distribution<> uu(0, 1);
	
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
            
  r = 0;
  for (ii=0;ii<L-mm;ii++)
    r += yheap[ii].val;
        
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
