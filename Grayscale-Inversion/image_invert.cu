#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// ================= CUDA Kernel =================
__global__ void invertKernel(unsigned char* d_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        d_image[idx] = 255 - d_image[idx];
    }
}

// ================= Utility Functions =================

std::vector<unsigned char> simulateImage(int width, int height) {
    std::vector<unsigned char> image(width * height);
    for (int i = 0; i < width * height; ++i) {
        image[i] = static_cast<unsigned char>(i * 255 / (width * height));
    }
    return image;
}

void printImage(const std::vector<unsigned char>& image, int width, int height, const std::string& label) {
    std::cout << "\n" << label << ":\n";
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::cout << static_cast<int>(image[y * width + x]) << "\t";
        }
        std::cout << "\n";
    }
}

std::vector<unsigned char> invertImageCUDA(const std::vector<unsigned char>& input, int width, int height) {
    size_t img_size = width * height;
    unsigned char* d_image;

    // Allocate and copy to device
    cudaMalloc(&d_image, img_size);
    cudaMemcpy(d_image, input.data(), img_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    invertKernel<<<grid, block>>>(d_image, width, height);
    cudaDeviceSynchronize();

    // Copy result back
    std::vector<unsigned char> output(img_size);
    cudaMemcpy(output.data(), d_image, img_size, cudaMemcpyDeviceToHost);
    cudaFree(d_image);

    return output;
}

// ================= Main =================

int main() {
    const int width = 8;
    const int height = 8;

    auto image = simulateImage(width, height);
    printImage(image, width, height, "🔍 Simulated Image");

    std::cout << "\n🚀 Running CUDA kernel..." << std::endl;
    auto inverted = invertImageCUDA(image, width, height);
    printImage(inverted, width, height, "✅ Inverted Image");

    return 0;
}
