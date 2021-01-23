#include <stdio.h>
#include <stdlib.h>
#include "mandel.h"
#include "writepng.h"
#include <omp.h>

int main(int argc, char *argv[]) {

    int   width, height;
    int	  max_iter;
    int   *d_image, *h_image;

    width    = 2601;
    height   = 2601;
    max_iter = 400;

    // command line argument sets the dimensions of the image
    if ( argc == 2 ) width = height = atoi(argv[1]);

    double time = omp_get_wtime();
    cudaMallocHost((void **) &h_image, width * height * sizeof(int));
    cudaMalloc((void **) &d_image, width * height * sizeof(int));
    printf("warmup time = %.4f seconds\n", omp_get_wtime() - time);

    cudaMemcpy(d_image, h_image,  width * height * sizeof(int), cudaMemcpyHostToDevice);
    
    time = omp_get_wtime();
    int K= 32;
    dim3 dimGrid(width/K, height/K);
    dim3 dimBlock(K,K);
    mandel<<<dimGrid,dimBlock>>>(width, height, d_image, max_iter);
    cudaDeviceSynchronize();
    printf("kernel time (exclude transfer) = %.4f seconds\n", omp_get_wtime() - time);

    cudaMemcpy(h_image, d_image,  width * height * sizeof(int), cudaMemcpyDeviceToHost);

    writepng("mandelbrot.png", h_image, width, height);

    cudaFreeHost(h_image);
    cudaFree(d_image);

    return(0);
}
