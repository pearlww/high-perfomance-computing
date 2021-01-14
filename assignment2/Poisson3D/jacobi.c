/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int jacobi(double *** u, double *** u_old, double *** f, int N, int iter_max, double	tolerance) {
    double p= 1.0/6.0;
    double d = INFINITY;
    int k=0;
    double delta=2.0/(N+1);
    double *** tmp;

    while(d > tolerance && k < iter_max){
        d=0;
        for(int i=1;i<=N;i++){
            for(int j=1;j<=N;j++){
                for(int k=1;k<=N;k++){
                    u[i][j][k] = u_old[i-1][j][k] + u_old[i+1][j][k]
                                +u_old[i][j-1][k] + u_old[i][j+1][k]
                                +u_old[i][j][k-1] + u_old[i][j][k+1]
                                +delta*delta*f[i][j][k];
                    u[i][j][k] *= p;
                    d += ((u[i][j][k]-u_old[i][j][k])*(u[i][j][k]-u_old[i][j][k]));       
                }
            }
        }
        d = sqrt(d);
        tmp = u_old;
        u_old= u;
        u=tmp;
        // printf("loss: %f \n", d);
        k += 1;
    }

    printf("loss: %f \n", d);
    return k;

}
