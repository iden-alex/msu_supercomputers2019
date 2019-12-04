#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))
#define N 130

double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k,nk;
double eps;

double A [N][N][N];

void relax();
void init();
void verify(); 


int main(int an, char **as)
{
    double time0, time1;
	int it;
	init();
    time0 = omp_get_wtime();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) 
            break;
	}
    printf("Time in seconds=%lf\t", omp_get_wtime() - time0);
	verify();
	return 0;
}


void init()
{ 

	for(k=0; k<=N-1; k++)
    for(j=0; j<=N-1; j++)
    for(i=0; i<=N-1; i++)
    {
        if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
            A[i][j][k]= 0.;
        else A[i][j][k]= ( 4. + i + j + k) ;
    }
} 

void relax()
{
#pragma omp parallel private(nk) shared(A)
    {
        for(nk=3; nk<=3*N-6; ++nk) {
#pragma omp for schedule(static) private(i, j, k) reduction(max: eps)
            for(i=Max(1, nk-2*N+4); i<=Min(N-2, nk-2); ++i) {
                for(j=Max(1, nk-i-N+2); j<=Min(N-2, nk-i-1); ++j) { 
                    double e;
                    k = nk - j -i;
                    e=A[i][j][k];
                    A[i][j][k]=(A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]+A[i][j][k-1]+A[i][j][k+1])/6.;
                    eps=Max(eps, fabs(e-A[i][j][k]));
                }
            }
        }
    }
}

void verify()
{ 
	double s;

	s=0.;
	for(k=0; k<=N-1; k++)
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}
	printf("  S = %f\n",s);

}



