#include <stdio.h>
#include <stdlib.h>


void cpu_mxv(int m, int n, int* a, int*  b, int* c){
    int sum;
    // #pragma omp parallel if(m>10|n>10) shared(m,n,a,b,c) private(i,j,sum)
    // #pragma omp for
    for(int i=0;i<m;i++){
        sum=0.0;
        for(int j=0;j<n;j++){
            sum += a[i*n+j]*b[j];
        }
        c[i]=sum;
    }
}

__global__ void gpu_mxv(int n, int* a, int* b, int* c){

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    //int row = blockIdx.y * blockDim.y + threadIdx.y;

    for(int i=0; i<n; i++){
        c[j] += a[j*n+i]*b[i];
    }
}

void validate(int m, int *h_c, int *h_cc, float gtime, float ctime){

    int all_ok = 1;
    for (int i = 0; i < m; ++i){
        //printf("h_c[%d]: %d  h_cc[%d]: %d\n", i, h_c[i], i, h_cc[i]);
        if(h_cc[i] != h_c[i]){
            all_ok = 0;
            break;
        }
    }
    // roughly compute speedup
    if(all_ok)
        printf("all results are correct!!!, speedup = %f\n", ctime/ gtime);
    else
        printf("incorrect results\n");
}


// main
int main(){
    int m = 500;
    int n = 1000; 
 
    // allocate memory in host
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, m*n*sizeof(int));
    cudaMallocHost((void **) &h_b, n*sizeof(int));
    cudaMallocHost((void **) &h_c, m*sizeof(int));
    cudaMallocHost((void **) &h_cc, m*sizeof(int));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 10;
        }
    }
    for( int i = 0; i < n; i++)
        h_b[i] = rand()% 10;	 
        
    // Allocate memory space on the device     
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, m*n*sizeof(int));
    cudaMalloc((void **) &d_b, n*sizeof(int));
    cudaMalloc((void **) &d_c, m*sizeof(int));

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a,  m*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b,  n*sizeof(int), cudaMemcpyHostToDevice);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);   
    gpu_mxv<<<10,50>>>(n, d_a, d_b, d_c); //m threads
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication on GPU: %f ms.\n",gpu_elapsed_time_ms);

     // copy result matrix C from device to host memory
    cudaMemcpy(h_c, d_c,  m*sizeof(int), cudaMemcpyDeviceToHost);

   // start to count execution time of CPU version
    cudaEventRecord(start, 0); 
    cpu_mxv(m, n, h_a, h_b, h_cc);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication on CPU: %f ms.\n",cpu_elapsed_time_ms);


    //compare the solution and claculate speed up
    validate(m, h_c, h_cc, gpu_elapsed_time_ms, cpu_elapsed_time_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}