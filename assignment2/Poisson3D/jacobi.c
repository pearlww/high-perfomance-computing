/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>

void jacobi(double *** u, double *** f, int N, int iter_max, double	tolerance) {
    double p= 1.0/6.0;
    double d = INFINITY;
    int k=0;
    double 	***u_old;

    while(d > tolerance*tolerance && k < iter_max){
        u_old=u;
        d=0;
        for(int i=1;i<=N;i++){
            for(int j=1;j<=N;j++){
                for(int k=1;k<=N;k++){
                    u[i][j][k] = u_old[i-1][j][k] + u_old[i+1][j][k]\
                                +u_old[i][j-1][k] + u_old[i][j+1][k]\
                                +u_old[i][j][k-1] + u_old[i][j][k+1]\
                                +f[i][j][k];
                    u[i][j][k] *= p;
                    d += (u[i][j][k]-u_old[i][j][k])*(u[i][j][k]-u_old[i][j][k]);           
                }
            }
        }
        k += 1;
    }

}
