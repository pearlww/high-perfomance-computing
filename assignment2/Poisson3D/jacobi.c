/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void jacobi(double *** u, double *** u_old, double *** f, int N, int iter_max, double tolerance) {
    double p= 1.0/6.0;
    double d = INFINITY;
    int iter=0;
    int i,j,k=0;
    double delta=2.0/(N+1);
    double *** tmp;

    while(d > tolerance && iter < iter_max){
    #pragma omp barrier
        d=0;
        #pragma omp parallel shared(f, u, u_old,d) private(i, j) firstprivate(k)
        {
        #pragma omp for reduction(+ : d)
        for(i=1;i<=N;i++){
            for(j=1;j<=N;j++){
                for(k=1;k<=N;k++){
                    u[i][j][k] = u_old[i-1][j][k] + u_old[i+1][j][k]
                                +u_old[i][j-1][k] + u_old[i][j+1][k]
                                +u_old[i][j][k-1] + u_old[i][j][k+1]
                                +delta*delta*f[i][j][k];
                    u[i][j][k] *= p;
                    d += ((u[i][j][k]-u_old[i][j][k])*(u[i][j][k]-u_old[i][j][k]));       
                }
            }
        }
        #pragma omp barrier
        #pragma omp single
        {
            d = sqrt(d);
            tmp = u_old;
            u_old= u;
            u=tmp;
        } 
        }
        if(iter%20==0)
            printf("iter: %d  loss: %f \n", iter, d);

        iter += 1;
    }

}
