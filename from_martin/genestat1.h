typedef struct _site_t{
	struct _site_t *d1;
	struct _site_t *d2;
	struct _site_t *par;
	double dlta;
	double total;
	int i;
} site_t;
typedef struct _point_t{
	struct _point_t *d1;
	struct _point_t *d2;
	struct _point_t *par;
	int asite;
	int prop;
	int na;
	int i;
} point_t;
typedef struct _store_t{
	int asite;
	char prop;
	char na;
} store_t;

int iranf(long *idum, int lo, int hi);
double ran1(long *idum) ;
point_t *buildtree(int n,long *idum );
site_t *buildlist(int m,long *idum );
void calclist(site_t *list, int m);
void updatetree(point_t *pnow, int temp);
void updatelist(site_t *pnow, double tmp);
int picksite(site_t *list,double r);
int prunelist(int m, long *idum);
int clonelist(site_t *list,long *idum);
void calctree(point_t *site, int n, double aa, double s);
double calcprop( point_t *pnow, point_t *next, double aa, double s);
int pickrxn(point_t *site, double sg, double *r2);
double onerxn(point_t *pnow,site_t *list_t, store_t **tree,int **lkup,int *bg,int *lng,int n, 
	int m, double aa, double s, long *idum,int *l);
void xchg(point_t *site,store_t *tree, int n);
void rvsxchg(store_t *site,point_t *tree, int n);

void ***alloc3d(int varsize, int n, int p, int q);
void **alloc2d(int varsize, int n, int p);

