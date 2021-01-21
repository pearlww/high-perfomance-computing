__global__ void implementation1(int m, int n, int k, double *d_A, double *d_B, double *d_C){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            d_C[i*n + j] = 0;
        }
    }
    //mkn
    for(int i = 0; i < m; i++){
        for(int s = 0; s < k; s++){
            for(int j = 0; j < n; j++){
                d_C[i*n + j] += d_A[i*k + s] * d_B[s*n + j]; }
        }
    }
}

extern "C" { 
	void matmult_gpu1(int m, int n, int k, double *A, double *B, double *C) { 

    
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
    // Launch kernel using 1 thread per block and 1 block 
    implementation1<<<1,1>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize(); 
    
    //transfer C back to CPU
    cudaMemcpy(C, d_C, size_matrix_C, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); }
    }

