#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

__global__ void ncc_kernel(const unsigned char* img, int img_w, int img_h,
                           const unsigned char* tmpl, int tmpl_w, int tmpl_h,
                           float tmpl_mean, float tmpl_var,
                           float* result) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x + tmpl_w > img_w || y + tmpl_h > img_h) return;

    float sum_i = 0.0f;
    float sum_i2 = 0.0f;
    float sum_it = 0.0f;

    for (int j = 0; j < tmpl_h; ++j) {
        for (int i = 0; i < tmpl_w; ++i) {
            int img_idx = (y + j) * img_w + (x + i);
            int tmpl_idx = j * tmpl_w + i;

            float iv = img[img_idx];
            float tv = tmpl[tmpl_idx];

            sum_i += iv;
            sum_i2 += iv * iv;
            sum_it += iv * tv;
        }
    }

    int N = tmpl_w * tmpl_h;
    float mean_i = sum_i / N;
    float var_i = sum_i2 / N - mean_i * mean_i;

    if (var_i <= 1e-5f || tmpl_var <= 1e-5f) {
        result[y * img_w + x] = 0.0f;
    } else {
        result[y * img_w + x] = (sum_it - N * mean_i * tmpl_mean) / (N * sqrtf(var_i * tmpl_var));
    }
}

extern "C" EXPORT void match_template(const unsigned char* img, int img_w, int img_h,
                                      const unsigned char* tmpl, int tmpl_w, int tmpl_h,
                                      float tmpl_mean, float tmpl_var,
                                      float* result) {
    unsigned char *d_img, *d_tmpl;
    float *d_result;

    size_t img_size = img_w * img_h * sizeof(unsigned char);
    size_t tmpl_size = tmpl_w * tmpl_h * sizeof(unsigned char);
    size_t res_size = img_w * img_h * sizeof(float);

    cudaMalloc(&d_img, img_size);
    cudaMalloc(&d_tmpl, tmpl_size);
    cudaMalloc(&d_result, res_size);

    cudaMemcpy(d_img, img, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmpl, tmpl, tmpl_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((img_w + block.x - 1) / block.x, (img_h + block.y - 1) / block.y);
    ncc_kernel<<<grid, block>>>(d_img, img_w, img_h, d_tmpl, tmpl_w, tmpl_h, tmpl_mean, tmpl_var, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, res_size, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_tmpl);
    cudaFree(d_result);
}
