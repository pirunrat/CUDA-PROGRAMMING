#include <cuda_runtime.h>
#include <iostream>

__global__ void threshold_kernel(unsigned char* d_img, int size, unsigned char thresh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_img[idx] = (d_img[idx] > thresh) ? 255 : 0;
    }
}

// Mark for export on Windows
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" EXPORT void threshold_image(unsigned char* img, int size, unsigned char thresh) {
    unsigned char *d_img;
    cudaMalloc(&d_img, size);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    threshold_kernel<<<gridSize, blockSize>>>(d_img, size, thresh);
    cudaDeviceSynchronize();

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
}
