#define STRIDE 2

__global__ void kernel3(int m, int n, int k, double *d_A, double *d_B, double *d_C){
    int i = (blockIdx.y * blockDim.y + threadIdx.y) * STRIDE; 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    
    int sum1=0;
    for(int s1 = 0; s1 < STRIDE; s1++){
        sum1 = i + s1; 
        if (sum1 < m && j < n){
            d_C[sum1*n + j] = 0.0;
            for(int s = 0; s < k; s++){
                d_C[sum1*n + j] += d_A[sum1*k + s] * d_B[s*n + j];  }
        }   
    } 
}


extern "C" { 
	void matmult_gpu3(int m, int n, int k, double *A, double *B, double *C) { 

    
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
    dim3 dimGrid((m -1)/dimBlock.x+1,(n/STRIDE-1)/dimBlock.y+1) ; 

    kernel3<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize(); 
    
    //transfer C back to CPU
    cudaMemcpy(C, d_C, size_matrix_C, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); }
    }