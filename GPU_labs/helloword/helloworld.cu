#include <stdio.h>
#include <helper_cuda.h>

__global__ void helloword(){
    int *a = (int*) 0x10000; 
    *a = 0;

    printf("Hello world! I’m thread %d out of 64 in block %d. My global thread id is %d out of 256.\n", 
            threadIdx.x,
            blockIdx.x,
            blockIdx.x * blockDim.x + threadIdx.x);
}

int main(int argc, char *argv[])
{
    helloword<<<4,64>>>();
    checkCudaErrors(cudaDeviceSynchronize());

}