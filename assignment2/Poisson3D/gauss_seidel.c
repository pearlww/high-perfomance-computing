/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void gauss_seidel(double *** u,  double *** f, int N, int iter_max, double tolerance) {
    double p= 1.0/6.0;
    double d = INFINITY;
    int iter=0;
    int i,j,k;
    double delta=2.0/(N+1);
    double u_old;
#pragma omp parallel num_threads(100)
{
    while(d > tolerance && iter < iter_max){
        d=0;
#pragma omp for schedule(static,1) ordered(2) private(j,k)
        for(i=1;i<=N;i++){
            for(j=1;j<=N;j++){
#pragma omp ordered depend(sink: i-1,j) depend(sink: i,j-1)
                for(k=1;k<=N;k++){
                    u_old = u[i][j][k];
                    u[i][j][k] = u[i-1][j][k] + u[i+1][j][k]
                                +u[i][j-1][k] + u[i][j+1][k]
                                +u[i][j][k-1] + u[i][j][k+1]
                                +delta*delta*f[i][j][k];
                    u[i][j][k] *= p;
                    d += ((u[i][j][k]-u_old)*(u[i][j][k]-u_old));        
                }
#pragma omp ordered depend(source)
            }
        }
        d = sqrt(d);
    
        if(iter%20==0)
            printf("iter: %d  loss: %f \n", iter, d);
        iter += 1;
    }
}
}

