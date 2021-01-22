/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "alloc3d_gpu.h"
#include "transfer3d_gpu.h"
#include "print.h"
#include "jacobi.h"

#define N_DEFAULT 100


int main(int argc, char *argv[]) {

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char    *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    
    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }

    // allocate memory in host
    printf("allocating memory in host \n");
    double 	***h_u = d_malloc_3d(N+2, N+2, N+2);
    double 	***h_u_old = d_malloc_3d(N+2, N+2, N+2);
    double 	***h_f = d_malloc_3d(N+2, N+2, N+2); 
    double  h_d;
    printf("finish\n");

    if (h_u == NULL || h_u_old == NULL || h_f==NULL ) {
        perror("array u: allocation failed \n");
        exit(-1);
    }

    //initialization
    printf("initialzing \n");
    for(int i=0;i<N+2;i++){
        for(int j=0;j<N+2;j++){
            for(int k=0;k<N+2;k++){
                if(i==0||i==N+1||k==0||k==N+1||j==N+1){
                    h_u[i][j][k] = 20;
                    h_u_old[i][j][k] = 20;
                    //printf("wall20 ");
                    }
                else if(j==0)
                    {h_u[i][j][k] = 0;
                    h_u_old[i][j][k] = 0;
                    //printf("wall0 ");
                    }
                else{
                    h_u_old[i][j][k] = start_T;}
            }
        }
    }

    double lx=0, ux=5.0/16.0*(N+2);
    double ly=0, uy=0.25*(N+2);
    double lz=1.0/6.0*(N+2), uz=0.5*(N+2);

    for(int i=0; i<N+2; i++){
        for(int j=0; j<N+2; j++){
            for(int k=0; k<N+2; k++){
                if(k>=lx && k<=ux && j>=ly && j<=uy && i>=lz && i<=uz)
                    h_f[i][j][k] = 200;
                else
                    h_f[i][j][k] = 0;
            }
        }
    }
    printf("finish \n");

    // Allocate memory in device
    printf("allocating memory in device \n");
    double 	***d_u = d_malloc_3d_gpu(N+2, N+2, N+2);
    double 	***d_u_old = d_malloc_3d_gpu(N+2, N+2, N+2);
    double 	***d_f = d_malloc_3d_gpu(N+2, N+2, N+2); 
    double  *d_d;
    cudaMalloc((void **) &d_d, sizeof(double));
    printf("finish \n");


    // copy matrix u and f from host to device memory
    printf("copying matrix u and f from host to device memory \n");
    transfer_3d(d_u, h_u, N+2, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d(d_u_old, h_u_old, N+2, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d(d_f, h_f, N+2, N+2, N+2, cudaMemcpyHostToDevice);
    printf("finish \n");


    //update
    h_d = INFINITY;
    int iter=0;

    int K= 8;
    dim3 dimGrid(N/K, N/K, N/K);
    dim3 dimBlock(K,K,K);
    double *** tmp;

    printf("runing kernal \n");
    while(h_d > tolerance*tolerance & iter < iter_max){
    
        // jacobi_seq<<<1,1>>>(d_u, d_u_old, d_f, N); // one thread
        // jacobi_nat<<<dimGrid,dimBlock>>>(d_u, d_u_old, d_f, N); //N^3 threads
        jacobi_nat_with_norm<<<dimGrid,dimBlock>>>(d_u, d_u_old, d_f, N, d_d);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_d, d_d, sizeof(double), cudaMemcpyDeviceToHost);

        tmp = d_u_old;
        d_u_old= d_u;
        d_u=tmp;
        if(iter%20==0)
            // printf("%d\n", iter);
            printf("iter: %d  loss: %f \n", iter, h_d);
        iter += 1;
    }
    printf("finish \n");

    //copy result u from device to host
    printf("copying result u from device to host\n");
    transfer_3d(h_u, d_u, N+2, N+2, N+2, cudaMemcpyDeviceToHost);
    printf("finish \n");


    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N+2, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N+2, h_u);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N+2, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N+2, h_u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    cudaFreeHost(h_u);
    cudaFreeHost(h_u_old);
    cudaFreeHost(h_f);
    cudaFree(d_u);
    cudaFree(d_u_old);
    cudaFree(d_f);

    return(0);
}
