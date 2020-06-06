#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#define  Max(a,b) ((a)>(b)?(a):(b))

#define    N   130
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;

double (*A) [N][N];
double (*B) [N][N];
double (*C) [N][N];

void relax(int nl, int rank, int ranks);
void init(int nl, int rank, int ranks);
void verify(int nl, int rank, int ranks); 

int main(int an, char **as)
{
    double total_time = 0;
    int ranks, rank;
    MPI_Init(&an, &as);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Barrier(MPI_COMM_WORLD);

    int num_layers = N / ranks;
    A = malloc(num_layers * N * N * sizeof(A[0][0][0]));
    if (ranks > 1) {
        B = malloc(N * N * sizeof(B[0][0][0]));
        C = malloc(N * N * sizeof(C[0][0][0]));
    }

	init(num_layers, rank, ranks);
    double time = MPI_Wtime();
	for(int it=1; it<=itmax; it++)
	{
		eps = 0;
		relax(num_layers, rank, ranks);
		if (rank == 0) {
            printf( "it=%4i   eps=%f\n", it,eps);
		    if (eps < maxeps) break;
        }
	}
    time = MPI_Wtime() - time;
    MPI_Reduce(&time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Num_processors = %d :: N = %d :: TIME = %lf\n", ranks, N, total_time);
    }

	verify(num_layers, rank, ranks);

    free(A);
    if (ranks > 1) {
        free(B);
        free(C);
    }
    MPI_Finalize();

	return 0;
}


void init(int nl, int rank, int ranks)
{ 
	for(i = 0; i < nl; ++i) {
    	for(j = 0; j < N; ++j) {
        	for(k = 0; k < N; ++k) {
    		    if(j==0 || j==N-1 || k==0 || k==N-1 ||(rank + ranks * i==0) || (rank + ranks * i==N-1)) {
	    	        A[i][j][k]= 0;
                } else {
                    A[i][j][k]= 4 + (rank + ranks * i) + j + k;
                }
            }
        }
	}
} 


void relax(int nl, int rank, int ranks)
{
    double loc = 0.;
	for(i=0; i<nl; ++i) {
        if ((i == 0 && rank == 0) || (i == nl - 1 && rank == ranks - 1)) {
            continue;
        }
        if (ranks > 1 && (rank != 1 || i != 0)) {
            if (!rank) {
                MPI_Send(&A[i][0][0], N * N, MPI_DOUBLE, ranks - 1, 0, MPI_COMM_WORLD);
            } else {
                MPI_Send(&A[i][0][0], N * N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            }
        }

        if (rank == ranks - 2 && i == nl - 1) {
            for (j = 0; j < N; ++j) {
                for (k = 0; k < N; ++k) {
                    C[0][j][k] = 0;
                }
            }
        } else if (ranks > 1) {
            MPI_Recv(&C[0][0][0], N * N, MPI_DOUBLE, (rank + 1) % ranks, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    	for(j=1; j<N-1; ++j) {
            if (ranks > 1 && (rank != 1 || i != 0)) {
                if (!rank) {
                    MPI_Recv(&B[0][j][0], N, MPI_DOUBLE, ranks - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(&B[0][j][0], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

        	for(k=1; k<N-1; ++k) { 
                double e = A[i][j][k];
                if (ranks == 1) {
                    A[i][j][k] = (A[i-1][j][k] + A[i+1][j][k] + A[i][j - 1][k] + A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6.;
                } else {
                    A[i][j][k] = (B[0][j][k] + C[0][j][k] + A[i][j - 1][k] + A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6.;    
                }
        		loc=Max(loc, fabs(e - A[i][j][k]));
            }
            if (ranks > 1 && (rank != ranks - 2 || i != nl - 1)) {
                MPI_Send(&A[i][j][0], N, MPI_DOUBLE, (rank + 1) % ranks, 1, MPI_COMM_WORLD);
            }
        }
	}
    MPI_Reduce(&loc, &eps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}

void verify(int nl, int rank, int ranks)
{ 
	double s = 0.;
    double loc_sum = 0.;
	for(i=0; i<=nl-1; i++) {
    	for(j=0; j<=N-1; j++) {
        	for(k=0; k<=N-1; k++) {
        		loc_sum = loc_sum + A[i][j][k]*(rank+ranks*i+1)*(j+1)*(k+1)/(N*N*N);
            }
        }
	}
    MPI_Reduce(&loc_sum, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
    	printf("  S = %f\n",s);
    }
}
