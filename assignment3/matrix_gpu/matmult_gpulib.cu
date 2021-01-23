#include <cublas_v2.h>

extern "C" {
    void matmult_gpulib(int m, int n, int k, double *A, double *B, double *C) {
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    double alpha = 1.0, beta = 0.0;
    int lda=k, ldb=n, ldc=n;
    
    double *d_A, *d_B, *d_C; //variable on device

    int size_matrix_A = m * k * sizeof(double);
    cudaMalloc((void**)&d_A, size_matrix_A); // allocate memory on GPU
    int size_matrix_B = k * n * sizeof(double);
    cudaMalloc((void**)&d_B, size_matrix_B);
    int size_matrix_C = m * n * sizeof(double);
    cudaMalloc((void**)&d_C, size_matrix_C);

    //copy A and B to GPU
    cudaMemcpy(d_A, A, size_matrix_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_matrix_B, cudaMemcpyHostToDevice); 

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc);
    cudaDeviceSynchronize(); 
    
    //transfer C back to CPU
    cudaMemcpy(C, d_C, size_matrix_C, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); 

}
}


    
