/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

__global__
void jacobi_seq(double *** u, double *** u_old, double *** f, int N) {
    double p= 1.0/6.0;
    int i,j,k=0;
    double delta=2.0/(N+1);

    for(i=1;i<=N;i++){
        for(j=1;j<=N;j++){
            for(k=1;k<=N;k++){
                u[i][j][k] = u_old[i-1][j][k] + u_old[i+1][j][k]
                            +u_old[i][j-1][k] + u_old[i][j+1][k]
                            +u_old[i][j][k-1] + u_old[i][j][k+1]
                            +delta*delta*f[i][j][k];
                u[i][j][k] *= p;
                //d += ((u[i][j][k]-u_old[i][j][k])*(u[i][j][k]-u_old[i][j][k]));       
            }
        }
    }
    //d = sqrt(d);
}

__global__
void jacobi_nat(double *** u, double *** u_old, double *** f, int N) {
    double p= 1.0/6.0;
    double delta=2.0/(N+1);


    int i = (blockIdx.x * blockDim.x + threadIdx.x)+1; // remember +1!!
	int j = (blockIdx.y * blockDim.y + threadIdx.y)+1;
    int k = (blockIdx.z * blockDim.z + threadIdx.z)+1;

    u[i][j][k] = u_old[i-1][j][k] + u_old[i+1][j][k]
                +u_old[i][j-1][k] + u_old[i][j+1][k]
                +u_old[i][j][k-1] + u_old[i][j][k+1]
                +delta*delta*f[i][j][k];
    u[i][j][k] *= p;
    //d += ((u[i][j][k]-u_old[i][j][k])*(u[i][j][k]-u_old[i][j][k]));       
    //d = sqrt(d);

}
__inline__ __device__ 
double warpReduceSum(double value){ 
    for (int i = 32; i > 0; i /= 2)         
        value += __shfl_down_sync(-1, value, i);
    return value;
}

__global__
void jacobi_nat_with_norm(double ***u, double ***u_old, double ***f, int N, double *d) {
    double p= 1.0/6.0;
    double delta=2.0/(N+1);


    int i = (blockIdx.x * blockDim.x + threadIdx.x)+1;
	int j = (blockIdx.y * blockDim.y + threadIdx.y)+1;
    int k = (blockIdx.z * blockDim.z + threadIdx.z)+1;
    //printf("%d %d %d\n", i,j,k);
    u[i][j][k] = u_old[i-1][j][k] + u_old[i+1][j][k]
                +u_old[i][j-1][k] + u_old[i][j+1][k]
                +u_old[i][j][k-1] + u_old[i][j][k+1]
                +delta*delta*f[i][j][k];
    u[i][j][k] *= p;

    double value = (u[i][j][k]-u_old[i][j][k])*(u[i][j][k]-u_old[i][j][k]);
    //printf("%lf ", value);
    //value = warpReduceSum(value); 
    //if (threadIdx.x % 32 == 0)
    atomicAdd(d, value);    

}

__global__
void jacobi_gpu0(double *** u, double *** u_old, double *** peer_u_old, double *** f, int N) {
    double p= 1.0/6.0;
    double delta=2.0/(N+1);

    int i = (blockIdx.x * blockDim.x + threadIdx.x)+1; // remember +1!!
	int j = (blockIdx.y * blockDim.y + threadIdx.y)+1;
    int k = (blockIdx.z * blockDim.z + threadIdx.z)+1;

    if(k==N/2){
        u[i][j][k] = u_old[i-1][j][k] + u_old[i+1][j][k]
                    +u_old[i][j-1][k] + u_old[i][j+1][k]
                    +u_old[i][j][k-1] + peer_u_old[i][j][k+1]
                    +delta*delta*f[i][j][k];       
    }
    else{
        u[i][j][k] = u_old[i-1][j][k] + u_old[i+1][j][k]
                    +u_old[i][j-1][k] + u_old[i][j+1][k]
                    +u_old[i][j][k-1] + u_old[i][j][k+1]
                    +delta*delta*f[i][j][k];
    }
    u[i][j][k] *= p;
    //d += ((u[i][j][k]-u_old[i][j][k])*(u[i][j][k]-u_old[i][j][k]));       
    //d = sqrt(d);
}

__global__
void jacobi_gpu1(double *** u, double *** u_old, double *** peer_u_old, double *** f, int N) {
    double p= 1.0/6.0;
    double delta=2.0/(N+1);

    int i = (blockIdx.x * blockDim.x + threadIdx.x)+1; // remember +1!!
	int j = (blockIdx.y * blockDim.y + threadIdx.y)+1;
    int k = (blockIdx.z * blockDim.z + threadIdx.z)+1;

    if(k==1){
        u[i][j][k] = u_old[i-1][j][k] + u_old[i+1][j][k]
                    +u_old[i][j-1][k] + u_old[i][j+1][k]
                    +peer_u_old[i][j][k-1] + u_old[i][j][k+1]
                    +delta*delta*f[i][j][k];       
    }
    else{
        u[i][j][k] = u_old[i-1][j][k] + u_old[i+1][j][k]
                    +u_old[i][j-1][k] + u_old[i][j+1][k]
                    +u_old[i][j][k-1] + u_old[i][j][k+1]
                    +delta*delta*f[i][j][k];
    }
    u[i][j][k] *= p;
    //d += ((u[i][j][k]-u_old[i][j][k])*(u[i][j][k]-u_old[i][j][k]));       
    //d = sqrt(d);
}
