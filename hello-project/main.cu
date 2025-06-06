#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_from_gpu() {
    printf("Hello from GPU!\n");
}

int main() {
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    return 0;
}
