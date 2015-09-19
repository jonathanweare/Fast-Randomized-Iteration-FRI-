#include <iostream>
#include <cstdlib>
#include <math.h>
#include "fri_public.h"


using namespace std;


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

  long Nit = 10000;
  long Brn = 500;
  long m = (long)1<<24;
  long bw = 2;

  long nv, nvnew, t, jj;
  double vsum, lambda=0, eps;

  spentry<double> *v = new spentry<double>[bw*m];
  spentry<double> *vnew = new spentry<double>[bw*m]; 

  for (jj=0;jj<bw*m;jj++)
    vnew[jj].loc = -1;
  nvnew = 1;

  nv = 1;
  v[0].val = 1.0;
  v[0].loc = 0;

  vsum=0;
  for(jj=0;jj<nv;jj++)
    vsum += v[jj].val;

  for(jj=0;jj<nv;jj++)
    v[jj].val /= vsum;

  clock_t start, end;
  double cpuTime;




  

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
