/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int gauss_seidel(double *** u,  double *** f, int N, int iter_max, double tolerance) {
    double p= 1.0/6.0;
    double d = INFINITY;
    int k=0;
    double delta=2.0/(N+1);
    double u_old;

    while(d > tolerance*tolerance && k < iter_max){
        d=0;
        for(int i=1;i<=N;i++){
            for(int j=1;j<=N;j++){
                for(int k=1;k<=N;k++){
                    u_old = u[i][j][k];
                    u[i][j][k] = u[i-1][j][k] + u[i+1][j][k]
                                +u[i][j-1][k] + u[i][j+1][k]
                                +u[i][j][k-1] + u[i][j][k+1]
                                +delta*delta*f[i][j][k];
                    u[i][j][k] *= p;
                    d += ((u[i][j][k]-u_old)*(u[i][j][k]-u_old));        
                }
            }
        }
        // printf("loss: %f \n", d);
        k += 1;
    }

    printf("loss: %f \n", d);
    return k;
}

