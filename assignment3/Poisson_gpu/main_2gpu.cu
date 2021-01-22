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
    double 	***h_u = d_malloc_3d(N+2, N+2, N+2);
    double 	***h_u_old = d_malloc_3d(N+2, N+2, N+2);
    double 	***h_f = d_malloc_3d(N+2, N+2, N+2); 

    if (h_u == NULL || h_u_old == NULL || h_f==NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    //initialization
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
 
    // Allocate memory in device
    // cudaSetDevice(0);
    // cudaDeviceEnablePeerAccess(1,0);
    // double 	***d0_u = d_malloc_3d_gpu(N+2, N+2, N/2+1);
    // double 	***d0_u_old = d_malloc_3d_gpu(N+2, N+2, N/2+1);
    // double 	***d0_f = d_malloc_3d_gpu(N+2, N+2, N/2+1); 
    // // copy matrix u and f from host to device0 memory
    // transfer_3d(d0_u, h_u, N+2, N+2, N/2+1, cudaMemcpyHostToDevice);
    // transfer_3d(d0_u_old, h_u_old, N+2, N+2, N/2+1, cudaMemcpyHostToDevice);
    // transfer_3d(d0_f, h_f, N+2, N+2, N/2+1, cudaMemcpyHostToDevice);

    // cudaSetDevice(1);
    // cudaDeviceEnablePeerAccess(0,0);
    // double 	***d1_u = d_malloc_3d_gpu(N+2, N+2, N/2+1);
    // double 	***d1_u_old = d_malloc_3d_gpu(N+2, N+2, N/2+1);
    // double 	***d1_f = d_malloc_3d_gpu(N+2, N+2, N/2+1); 
    // // copy matrix u and f from host to device1 memory
    // transfer_3d(d1_u, h_u+(N+2)*(N+2)*(N/2+1),  N+2, N+2, N/2+1, cudaMemcpyHostToDevice);
    // transfer_3d(d1_u_old, h_u_old+(N+2)*(N+2)*(N/2+1), N+2, N+2, N/2+1, cudaMemcpyHostToDevice);
    // transfer_3d(d1_f, h_f+(N+2)*(N+2)*(N/2+1), N+2, N+2, N/2+1, cudaMemcpyHostToDevice); 

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1,0);
    double 	***d0_u = d_malloc_3d_gpu(N+2, N+2, N+2);
    double 	***d0_u_old = d_malloc_3d_gpu(N+2, N+2, N+2);
    double 	***d0_f = d_malloc_3d_gpu(N+2, N+2, N+2); 
    // copy matrix u and f from host to device0 memory
    transfer_3d(d0_u, h_u, N+2, N+2, N/2+1, cudaMemcpyHostToDevice);
    transfer_3d(d0_u_old, h_u_old, N+2, N+2, N/2+1, cudaMemcpyHostToDevice);
    transfer_3d(d0_f, h_f, N+2, N+2, N/2+1, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0,0);
    double 	***d1_u = d_malloc_3d_gpu(N+2, N+2, N+2);
    double 	***d1_u_old = d_malloc_3d_gpu(N+2, N+2, N+2);
    double 	***d1_f = d_malloc_3d_gpu(N+2, N+2, N+2); 
    // copy matrix u and f from host to device1 memory
    transfer_3d(d1_u, h_u+(N+2)*(N+2)*(N/2+1),  N+2, N+2, N/2+1, cudaMemcpyHostToDevice);
    transfer_3d(d1_u_old, h_u_old+(N+2)*(N+2)*(N/2+1), N+2, N+2, N/2+1, cudaMemcpyHostToDevice);
    transfer_3d(d1_f, h_f+(N+2)*(N+2)*(N/2+1), N+2, N+2, N/2+1, cudaMemcpyHostToDevice); 


     //update
     double d = INFINITY;
     int iter=0;
 
     int K= 8;
     dim3 dimGrid((N)/K, (N)/K, (N)/K);
     dim3 dimBlock(K,K,K/2);
     double *** tmp;
 
     while(iter < iter_max){
         //d=0;
        
         jacobi_gpu0<<<dimGrid,dimBlock>>>(d0_u, d0_u_old, d1_u_old, d0_f, N);
         jacobi_gpu1<<<dimGrid,dimBlock>>>(d1_u, d1_u_old, d0_u_old, d1_f, N);

         cudaDeviceSynchronize();
         tmp = d0_u_old;
         d0_u_old= d0_u;
         d0_u=tmp;

         tmp = d1_u_old;
         d1_u_old= d1_u;
         d1_u=tmp; 

         if(iter%20==0)
             printf("%d\n", iter);
             //printf("iter: %d  loss: %f \n", iter, d);
         iter += 1;
     }

     //copy result u from device to host
     transfer_3d(h_u, d0_u, N+2, N+2, N/2+1, cudaMemcpyDeviceToHost);
     transfer_3d(h_u+(N+2)*(N+2)*(N/2+1), d1_u, N+2, N+2, N/2+1, cudaMemcpyDeviceToHost);
 
 
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
     cudaFree(d0_u);
     cudaFree(d0_u_old);
     cudaFree(d0_f);
     cudaFree(d1_u);
     cudaFree(d1_u_old);
     cudaFree(d1_f); 
     return(0);
 }
 