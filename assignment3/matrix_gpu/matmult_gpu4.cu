#define STRIDE 4

__global__ void kernel4(int m, int n, int k, double *d_A, double *d_B, double *d_C){
    int j = (blockIdx.y * blockDim.y + threadIdx.y) * STRIDE; 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    if (i < m && j < n){
        for(int s = 0; s < k; s++){
            sum1 += d_A[i*k + s] * d_B[s*n + j];
            if (j+1 < n)
            sum2 += d_A[i*k + s] * d_B[s*n + j+1];
            if (j+2 < n)
            sum3 += d_A[i*k + s] * d_B[s*n + j+2];
            if (j+3 < n)
            sum4 += d_A[i*k + s] * d_B[s*n + j+3];
        }
        d_C[i*n + j] = sum1;
        if (j+1 < n)
        d_C[i*n + j+1] = sum2;
        if (j+2 < n)
        d_C[i*n + j+2] = sum3;
        if (j+3 < n)
        d_C[i*n + j+3] = sum4;
    }
}

extern "C" { 
	void matmult_gpu4(int m, int n, int k, double *A, double *B, double *C) { 

    
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

    dim3 dimBlock(16,16,1);
    dim3 dimGrid((m -1)/dimBlock.x+1,((n-1)/STRIDE-1)/dimBlock.y+1) ;  

    kernel4<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize(); 
    
    //transfer C back to CPU
    cudaMemcpy(C, d_C, size_matrix_C, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); }
    }