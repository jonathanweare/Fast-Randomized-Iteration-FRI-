#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ran.h"
#include "genestat1.h"


int main( int argc, char *argv[]){
	
	double t,s,ss,aa,tmp,dlta=0.0,t_tmp=0.0,p,r,tinc,q=1.05;
	int i,j,l,cnt,m,n,k,fillchk;
	int *bg,*lng,**lkup;
	site_t *list_t,*list_n;
	point_t *site;
	store_t **tree;
	long idum;
	FILE *fp;
	
	if( argc != 4){
		printf("Usage: list size, seed, s\n");
		exit(0);
	}
	
	m = atoi(argv[1]);
	idum= atoi(argv[2]);
	s = atof(argv[3]);
//	p = atof(argv[4]);
//	fp=fopen(argv[4],"w");

	aa=1.0;
	n=4;
	ss=0.1*s;
	k=n/2;

	site=buildtree(n,&idum);
	list_t=buildlist(k,&idum);
	list_n=buildlist(k,&idum);
	tree=(store_t **) alloc2d(sizeof(store_t),m,n);
	bg=(int *) malloc(sizeof(int)*k);
	lng=(int *) malloc(sizeof(int)*k);
	lkup=(int **) alloc2d(sizeof(int),k,m);
	
	for(i=0;i<k;i++) 
		{bg[i]=0;
		lng[i]=0;
		}

	calctree(site,n,aa,ss);
	calclist(list_t,k);
	calclist(list_n,k);


//	FILLING UP THE LIST
	j=0;t=0.0;cnt=0;fillchk=0;
	while(cnt<m){
		j=site[0].asite-1;
		if(lng[j]<m/k){
			lng[j]++;
			lkup[j][(bg[j]+lng[j]-1)%m]=fillchk;
			rvsxchg(tree[fillchk],site,n);
			fillchk++;
			}
		else	{
			l=lkup[j][bg[j]];
			bg[j]=(bg[j]+1)%m;
			lkup[j][(bg[j]+lng[j]-1)%m]=l;
			rvsxchg(tree[l],site,n);
			}
		
		l=1;
		tinc=onerxn(site,list_t,tree,lkup,bg,lng,n,m,aa,s,&idum,&l);
		t+=tinc;
		updatelist(list_t+j,tinc);
		cnt++;
	}

	for(j=0;j<k;j++)
		if(lng[j]>=m/(10*k)) updatelist(list_n+j,lng[j]);


//	END FILLING UP THE LIST
	
	cnt=0;t=t_tmp=0.0;j=0;
	while(ss<s){
/*		i=clonelist(list_n,&idum);
		j=site[0].asite-1;
		l=lkup[i][bg[i]];
		lng[i]--;
		bg[i]=(bg[i]+1)%m;
		if(lng[i]<m/(10*k)) {
				tinc=0.0-list_n[i].dlta;
				updatelist(list_n+i,tinc);
				}
		else 
			updatelist(list_n+i,-1.0);

		lng[j]++;
		lkup[j][(bg[j]+lng[j]-1)%m]=l;

		if(lng[j]>=m/(k*10)){
					tinc=lng[j]-list_n[j].dlta;
					updatelist(list_n+j,tinc);
					}

		rvsxchg(tree[l],site,n); 
*/
		j=site[0].asite-1;
		if(lng[j]<m/k){
			lng[j]++;
			lkup[j][(bg[j]+lng[j]-1)%m]=fillchk;
			rvsxchg(tree[fillchk],site,n);
			fillchk++;
			}
		else	{
			l=lkup[j][bg[j]];
			bg[j]=(bg[j]+1)%m;
			lkup[j][(bg[j]+lng[j]-1)%m]=l;
			rvsxchg(tree[l],site,n);
			}
		
		l=0;
		tinc=onerxn(site,list_t,tree,lkup,bg,lng,n,m,aa,s,&idum,&l);
		t+=tinc;
		updatelist(list_t+j,tinc);
		
		cnt++;

		if(cnt%(m*100)==0){
			ss=ss+0.1*s;
			cnt=0;
		}

		
	}

	cnt=0;t=t_tmp=0.0;j=0;
	while(fillchk<m){
		j=site[0].asite-1;
		lng[j]++;
		lkup[j][(bg[j]+lng[j]-1)%m]=fillchk;
		rvsxchg(tree[fillchk],site,n);
		fillchk++;
		
		l=0;
		tinc=onerxn(site,list_t,tree,lkup,bg,lng,n,m,aa,s,&idum,&l);
		t+=tinc;
		updatelist(list_t+j,tinc);
		
		cnt++;
		if(cnt%(m*100)==0) {
			tmp=0.0;
			for(i=0;i<k;i++) tmp+=(exp(s)-1)*aa*(i+1)*list_t[i].dlta;
			dlta+=(t-t_tmp)*tmp/list_t[0].total;
			t_tmp=t;
			printf("%e %e %e\n",t,tmp/list_t[0].total,dlta/t);
			fflush(stdout);
			cnt=0;
		}

		if(cnt%(m*1)==0)
			for(i=0;i<k;i++){
				list_t[i].dlta=list_t[i].dlta/q;
				list_t[i].total=list_t[i].total/q;

			}

		
	}
	while(t<10000000000.0){
		i=clonelist(list_n,&idum);
		j=site[0].asite-1;
		l=lkup[i][bg[i]];
		lng[i]--;
		bg[i]=(bg[i]+1)%m;
		if(lng[i]<m/(10*k)) {
				tinc=0.0-list_n[i].dlta;
				updatelist(list_n+i,tinc);
				}
		else 
			updatelist(list_n+i,-1.0);

		lng[j]++;
		lkup[j][(bg[j]+lng[j]-1)%m]=l;

		if(lng[j]>=m/(k*10)){
					tinc=lng[j]-list_n[j].dlta;
					updatelist(list_n+j,tinc);
					}

		rvsxchg(tree[l],site,n);
		
		l=0;
		tinc=onerxn(site,list_t,tree,lkup,bg,lng,n,m,aa,s,&idum,&l);
		t+=tinc;
		updatelist(list_t+j,tinc);
		
		cnt++;
		if(cnt%(m*100)==0) {
			tmp=0.0;
			for(i=0;i<k;i++) tmp+=(exp(s)-1)*aa*(i+1)*list_t[i].dlta;
			dlta+=(t-t_tmp)*tmp/list_t[0].total;
			t_tmp=t;
			printf("%e %e %e\n",t,tmp/list_t[0].total,dlta/t);
			fflush(stdout);
			cnt=0;
		}

		if(cnt%(m*1)==0)
			for(i=0;i<k;i++){
				list_t[i].dlta=list_t[i].dlta/q;
				list_t[i].total=list_t[i].total/q;

			}

		
	}


}


double onerxn(point_t *pnow, site_t *list_t, store_t **tree,int **lkup, int *bg,
		int *lng,int n, int m, double aa, double s, long *idum,int *l){


	double r1, r2, tinc,atotal;
	int i,cnt,tmp;
	point_t *anow;

	atotal = aa*exp(s)*(pnow->asite)+(1-*l)*aa*(1-exp(s))*(pnow->asite);
//	printf("atota=%e l=%e\n",atotal,(1-*l)*sg*aa*(1-exp(s))*(pnow->asite));fflush(stdout);
		

	r1=ran1(idum);
	tinc =  -1/(atotal)*log(r1);
//		printf("tinc=%e\n",tinc);fflush(stdout);
	r2=ran1(idum)*atotal/(aa*exp(s));

	
//	i=pickrxn(pnow,(1-*l)*(exp(-s)-1),&r2);
//		printf("i=%d\n",i);fflush(stdout);


	if(r2>(pnow->asite)){
		cnt=clonelist(list_t,idum);
		r1=ran1(idum);
		i=(bg[cnt]+floor(lng[cnt]*r1));
		i=i%m;
		xchg(pnow,tree[lkup[cnt][i]],n);
		return tinc;}
		
	else {
		anow=pnow;
		while((anow->prop)<r2){
			r2=r2-(anow->prop);
			if((anow->d1->asite)>=r2) anow=anow->d1;
				else {r2=r2-(anow->d1->asite);anow=anow->d2;}
		}
		i=anow->i;
//		if(pnow[(i+1)%n].na!=0) {printf("Exiting (Code 1)\n");exit(0);}
			
		pnow[i].na--;pnow[(i+1)%n].na++;
		
		tmp=-pnow[i].prop;
		pnow[i].prop=0.0;
		updatetree(pnow+i,tmp);

		tmp=-pnow[(i+1)%n].prop;
		pnow[(i+1)%n].prop=calcprop(pnow+((i+1)%n),pnow+((i+2)%n),aa,s);
		tmp=pnow[(i+1)%n].prop-tmp;
		updatetree(pnow+((i+1)%n),tmp);

		tmp=-pnow[(i+n-1)%n].prop;
		pnow[(i+n-1)%n].prop=calcprop(pnow+((i+n-1)%n),pnow+i,aa,s);
		tmp=pnow[(i+n-1)%n].prop-tmp;
		updatetree(pnow+((i+n-1)%n),tmp);
		
		return tinc;
	
	}

}


int pickrxn(point_t *pnow, double sg, double *r2){

	int i;
	
	if((sg*(pnow->asite))>=*r2) return 2;

	*r2=*r2-(sg*(pnow->asite));
	if((pnow->asite) >= *r2)
			return 0;
	return -1;

}


double calcprop( point_t *pnow, point_t *next, double aa, double s){

		double atot = 0.0;
		int na,na_next;
		
		na=pnow->na;
		na_next=next->na;

		if((na>0)&&(na_next==0)) atot = 1;
		else atot = 0.0;


		return atot;

}
void calctree(point_t *site, int n, double aa, double s){

	int i;

	for(i=n-1;i>=0;i--){
		site[i].prop=calcprop(site+i,site+((i+1)%n), aa,s);
		site[i].asite+=site[i].prop;
		if(i>0)
			site[i].par->asite+=site[i].asite;
	}
}

int clonelist(site_t *list,long *idum){

	int i;
	site_t *pnow;
	double r;

	r=ran1(idum)*(list->total);
	pnow=list;

	while( (i=picksite(pnow,r))!=0 ){
		if(i == 1) {
			r=r-(pnow->dlta);
			pnow=pnow->d1;
			}
		else if(i == 2) {
			r=r-(pnow->dlta)-(pnow->d1->total);
			pnow=pnow->d2;
			}
		}
	return pnow->i;

}
int prunelist(int m, long *idum){

	int i;
	double r;

	r=ran1(idum);
	i=floor(r*m);

	return i;

}

int picksite(site_t *list,double r){
	
	double anow=0.0;

	anow+=list->dlta;
	if(anow >= r) return 0;
	anow+=(list->d1->total);
	if(anow >=r) return 1;
	else return 2;

}




void updatelist(site_t *pnow, double tmp){

	pnow->dlta+=tmp;
	pnow->total+=tmp;
	while((pnow->par)!=NULL){
		pnow->par->total+=tmp;
		pnow=pnow->par;
	}
}

void updatetree(point_t *pnow,int temp){

	pnow->asite+=temp;
	while((pnow->par)!=NULL){
		pnow->par->asite+=temp;
		pnow=pnow->par;
	}
}


void calclist(site_t *list, int m){

	
	int i;


	for( i=m-1 ;i>=0; i--){
		list[i].total+=list[i].dlta;
		if(i>0) {
			list[i].par->total+=list[i].total;
		}
	}
}
	



site_t *buildlist(int m,long *idum ){
	
	site_t *list;
	int i;

 	list = (site_t *) malloc(m*sizeof(site_t));

	for( i = 0 ; i <m ; i ++){

		if(i==0) list[i].par=NULL;
		
		else list[i].par=(list+(i-1)/2);
		if(2*(i+1) < m){
			list[i].d1=(list+2*i+1);
			list[i].d2=(list+2*i+2);
		}
		else if(2*i+1 < m) {
			list[i].d1=(list+2*i+1);
			list[i].d2=NULL;
			}
		else {
			list[i].d1=NULL;
			list[i].d2=NULL;
			}

		list[i].total=0.0;
		list[i].dlta=0.0;
		list[i].i=i;
	}

	return list;


}

point_t *buildtree(int n,long *idum ){
	
	point_t *tree;
	int i,nn;

 	tree = (point_t *) malloc(n*sizeof(point_t));

	for( i = 0 ; i <n ; i ++){

		if(i==0) tree[i].par=NULL;
		
		else tree[i].par=(tree+(i-1)/2);
		if(2*(i+1) < n){
			tree[i].d1=(tree+2*i+1);
			tree[i].d2=(tree+2*i+2);
		}
		else if(2*i+1 < n) {
			tree[i].d1=(tree+2*i+1);
			tree[i].d2=NULL;
			}
		else {
			tree[i].d1=NULL;
			tree[i].d2=NULL;
			}

		tree[i].asite=0;
		tree[i].na=0;
		tree[i].i=i;
	}
	nn=n/2;
	while(nn>0){
		i=floor(ran1(idum)*n);
		if (tree[i].na==0) {tree[i].na=1;nn--;}
	}

	return tree;


}

void xchg(point_t *site,store_t *tree,int n){
	int i;

	for(i=0;i<n;i++){
		site[i].asite=tree[i].asite;
		site[i].prop=tree[i].prop;
		site[i].na=tree[i].na;
	}

}
void rvsxchg(store_t *site,point_t *tree,int n){
	int i;

	for(i=0;i<n;i++){
		site[i].asite=tree[i].asite;
		site[i].prop=tree[i].prop;
		site[i].na=tree[i].na;
	}

}


int iranf(long *idum, int lo, int hi){
	int range, val;
	range = hi -lo +1;
	val = lo + (int) (range*ran1(idum)) ;
	return val;
}

/*
 *  Function: ran1
 *  --------------------------------------------------------------------
 *  random number generator (uniform distribution between 0 and 1)
 */

double ran1(long *idum) {
  int j;
  long k;
  static long iy=0;
  static long iv[NTAB];
  float temp;
  if (*idum <= 0 || !iy) {
    if (-(*idum) < 1) *idum=1;
    else *idum = -(*idum);
    for (j=NTAB+7;j>=0;j--) { 
      k=(*idum)/IQ;
      *idum=IA*(*idum-k*IQ)-IR*k;
      if (*idum < 0) *idum += IM;
      if (j < NTAB) iv[j] = *idum;
    }
    iy=iv[0];
  }
  k=(*idum)/IQ; 
  *idum=IA*(*idum-k*IQ)-IR*k; 
  if (*idum < 0) *idum += IM;
  j=iy/NDIV; 
  iy=iv[j]; 
  iv[j] = *idum;
  if ((temp=AM*iy) > RNMX) return RNMX; 
  else return temp;
}
void **alloc2d(int varsize, int n, int p) {
    int k ;
    void **a ;

    if ((a = (void **) malloc(n*sizeof(void *))) == NULL) { 
        printf ("Memory Error in alloc2d.\n") ;
        exit(1) ;
    }

    for (k = 0 ; k < n ; k++) {
        if ((a[k] = (void *) malloc(p*varsize)) == NULL) { 
            printf ("Memory Error in alloc2d.\n") ;
            exit(1) ;
        }
    }
    return a ;
}



void ***alloc3d(int varsize, int n, int p, int q) {
    int k,j ;
    void ***a ;

    if ((a = (void ***) malloc(n*sizeof(void **))) == NULL) { 
        printf ("Memory Error in alloc2d.\n") ;
        exit(1) ;
    }

    for (k = 0 ; k < n ; k++) {
        if ((a[k] = (void **) malloc(p*sizeof(void *))) == NULL) { 
            printf ("Memory Error in alloc2d.\n") ;
            exit(1) ;
        }
		for( j = 0 ; j < p ; j++){
			if ((a[k][j] = (void *) malloc(q*varsize)) == NULL) { 
				printf ("Memory Error in alloc2d.\n") ;
				exit(1) ;
			}
		}

    }
    return a ;
}


