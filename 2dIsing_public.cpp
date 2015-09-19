#include <iostream>
#include <cstdlib>
#include <math.h>
#include "fri_public.h"


using namespace std;


// a subroutine that returns the ii-th column of the transfer matrix of the 2D Ising model
int transfer(spentry<double> *col, long *nrows, long ii){

  int n = 50;
  double BB = 0.01;
  double TT = 2.2;
  
  double aa, bb, cc, invaa, invbb, invcc;
  long L;
  
  *nrows = 2;

  aa = exp((2.0-BB)/TT);
  bb = exp(-BB/TT);
  cc = exp((-2.0-BB)/TT);
  invaa = exp(-(2.0-BB)/TT);
  invbb = exp(BB/TT);
  invcc = exp((2.0+BB)/TT);

  L = (long)1<<n;
  
  if (ii < (L>>2)){
    col[0].val = aa;
    col[0].loc = (ii<<1);

    col[1].val = bb;
    col[1].loc = (ii<<1)+1;
  }
  else if (ii>= (L>>2) && ii< (L>>1)){
    col[0].val = bb;
    col[0].loc = (ii<<1);

    col[1].val = cc;
    col[1].loc = (ii<<1)+1;
  }

  else if (ii>=(L>>1) && ii< (L>>1)+(L>>2)){
    col[0].val = invaa;
    col[0].loc = (ii<<1)-L;

    col[1].val = invbb;
    col[1].loc = (ii<<1)+1-L;
  }

  else if (ii>= (L>>1)+(L>>2)){
    col[0].val = invbb;
    col[0].loc = (ii<<1)-L;

    col[1].val = invcc;
    col[1].loc = (ii<<1)+1-L;
  }
  else
    printf("problem in ising\n");

  return 0;
}




int main()
{

  long Nit = 10000;              // number of iterations after burn in
  long Brn = 500;                // number of burn in iterations (these are not included in trajectory averages)
  long m = (long)1<<24;          // compression parameter (after compression vectors have no more than m non-zero entries)
  long bw = 2;                   // upper bound on the number of entries in each column of matrix

  long nv, nvnew, t, jj;
  double vsum, lambda=0, eps;
  
  // The struct spentry is declared in fri_public.h and contains a value and an index.  
  // A sparse vector v is stored as an integer (say nv) along with an array with elements of type spentry.

  spentry<double> *v = new spentry<double>[bw*m];            // v is the current iterate
  spentry<double> *vnew = new spentry<double>[bw*m];         // vnew is the updeated iterate

  for (jj=0;jj<bw*m;jj++)
    vnew[jj].loc = -1;
  nvnew = 1;

  nv = 1;                                                    // initial vector
  v[0].val = 1.0;
  v[0].loc = 0;

  vsum=0;
  for(jj=0;jj<nv;jj++)
    vsum += v[jj].val;

  for(jj=0;jj<nv;jj++)
    v[jj].val /= vsum;

  clock_t start, end;
  double cpuTime;




  
// run Brn iterations without accumulating trajectory averages to discard initial transient
  for (t=0;t<Brn;t++){

    start = clock();
    dspcompmatmult(m, 1.0,transfer, bw, v, &nv, 0.0,
		   vnew, &nvnew);
    end = clock();

    vsum=0;
    for(jj=0;jj<nvnew;jj++)
      vsum += vnew[jj].val;

    for(jj=0;jj<nvnew;jj++)
      vnew[jj].val /= vsum;

    for(jj=0;jj<nvnew;jj++)
      v[jj] = vnew[jj];
    nv = nvnew;

    printf("burn: %ld\t lambda: %lf\t nonzeros: %ld\t elapsed time: %lf\n",
	   t+1, vsum, nv, ((double)end - start)/CLOCKS_PER_SEC);
  }




// generate a trajectory of Nit iterations and accumulate trajectory averages  
  for (t=0;t<Nit;t++){

    eps = 1.0/(1.0+t);


    start = clock();
    dspcompmatmult(m, 1.0,transfer, bw, v, &nv, 0.0,
		vnew, &nvnew);
    end = clock();

    vsum=0;
    for(jj=0;jj<nvnew;jj++)
      vsum += vnew[jj].val;

    lambda = (1.0-eps)*lambda + eps*vsum;

    for(jj=0;jj<nvnew;jj++)
      vnew[jj].val /= vsum;

    for(jj=0;jj<nvnew;jj++)
      v[jj] = vnew[jj];
    nv = nvnew;

    printf("iteration: %ld\t lambda: %lf\t average: %lf\t nonzeros: %ld\t elapsed time: %lf\n",
	   t+1, vsum, lambda, nv, ((double)end - start)/CLOCKS_PER_SEC);
    
  }


  
  delete v;
  delete vnew;
   
  return 0;
  
}
