#include <cuda_runtime.h>
#include <cstdio>

// Export macro for Windows
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// Example kernel: assign region 255 if above threshold, else 0
__global__ void dummy_segment_kernel(unsigned char* d_img, int* d_labels, int width, int height, unsigned char thresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    if (d_img[idx] > thresh) {
        d_labels[idx] = 255;  // mark as foreground
    } else {
        d_labels[idx] = 0;    // mark as background
    }
}

extern "C" EXPORT void watershed_run(unsigned char* img, int* labels, int width, int height, unsigned char thresh) {
    int size_img = width * height * sizeof(unsigned char);
    int size_labels = width * height * sizeof(int);

    unsigned char* d_img;
    int* d_labels;

    cudaMalloc(&d_img, size_img);
    cudaMalloc(&d_labels, size_labels);

    cudaMemcpy(d_img, img, size_img, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    dummy_segment_kernel<<<grid, block>>>(d_img, d_labels, width, height, thresh);
    cudaDeviceSynchronize();

    cudaMemcpy(labels, d_labels, size_labels, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_labels);
}
