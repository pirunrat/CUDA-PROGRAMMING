#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void invertKernel(unsigned char* d_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        d_image[idx] = 255 - d_image[idx];
    }
}

int main() {
    std::cout << "🔍 Loading image..." << std::endl;
    cv::Mat image = cv::imread("me.jpg", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "❌ Failed to load 'me.jpg'. Check if the file exists." << std::endl;
        return -1;
    }

    std::cout << "✅ Loaded image size: " << image.cols << "x" << image.rows << std::endl;

    unsigned char* d_image;
    size_t img_size = image.rows * image.cols;

    cudaMalloc(&d_image, img_size);
    cudaMemcpy(d_image, image.data, img_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((image.cols + 15) / 16, (image.rows + 15) / 16);

    std::cout << "🚀 Running CUDA kernel..." << std::endl;
    invertKernel<<<grid, block>>>(d_image, image.cols, image.rows);
    cudaDeviceSynchronize();

    cv::Mat result(image.size(), CV_8UC1);
    cudaMemcpy(result.data, d_image, img_size, cudaMemcpyDeviceToHost);
    cudaFree(d_image);

    std::cout << "💾 Saving result to 'output.jpg'..." << std::endl;
    if (cv::imwrite("output.jpg", result)) {
        std::cout << "✅ Inverted image saved as 'output.jpg'" << std::endl;
    } else {
        std::cerr << "❌ Failed to save image." << std::endl;
    }

    // Optional: show result
    cv::imshow("Original", image);
    cv::imshow("Inverted", result);
    cv::waitKey(0);

    return 0;
}
