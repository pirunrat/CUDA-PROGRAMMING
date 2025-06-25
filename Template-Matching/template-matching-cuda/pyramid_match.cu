#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// ─────────────────────────────────────────────────────────────
// CUDA Kernel: Normalized Cross-Correlation
// ─────────────────────────────────────────────────────────────
__global__ void ncc_kernel(const unsigned char* img, int img_w, int img_h, 
                           const unsigned char* tmpl, int tmpl_w, int tmpl_h,
                           float tmpl_mean, float tmpl_var,
                           float* result) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x + tmpl_w > img_w || y + tmpl_h > img_h) return;

    float sum_i = 0.0f, sum_i2 = 0.0f, sum_it = 0.0f;

    for (int j = 0; j < tmpl_h; ++j) {
        for (int i = 0; i < tmpl_w; ++i) {
            int img_idx = (y + j) * img_w + (x + i);
            int tmpl_idx = j * tmpl_w + i;

            float iv = img[img_idx];
            float tv = tmpl[tmpl_idx];

            sum_i  += iv;
            sum_i2 += iv * iv;
            sum_it += iv * tv;
        }
    }

    int N = tmpl_w * tmpl_h;
    float mean_i = sum_i / N;
    float var_i  = sum_i2 / N - mean_i * mean_i;

    float score = 0.0f;
    if (var_i > 1e-5f && tmpl_var > 1e-5f) {
        score = (sum_it - N * mean_i * tmpl_mean) / (N * sqrtf(var_i * tmpl_var));
    }
    result[y * img_w + x] = score;
}

// ─────────────────────────────────────────────────────────────
// Device Memory Allocation & Launch Helpers
// ─────────────────────────────────────────────────────────────
void launch_ncc_kernel(const unsigned char* d_img, int img_w, int img_h,
                       const unsigned char* d_tmpl, int tmpl_w, int tmpl_h,
                       float tmpl_mean, float tmpl_var, float* d_result) {
    dim3 block(16, 16);
    dim3 grid((img_w + block.x - 1) / block.x, (img_h + block.y - 1) / block.y);

    ncc_kernel<<<grid, block>>>(d_img, img_w, img_h, d_tmpl, tmpl_w, tmpl_h,
                                tmpl_mean, tmpl_var, d_result);
    cudaDeviceSynchronize();
}

void allocate_and_copy_to_device(const unsigned char* h_data, size_t size, unsigned char** d_data) {
    cudaMalloc((void**)d_data, size);
    cudaMemcpy(*d_data, h_data, size, cudaMemcpyHostToDevice);
}

void allocate_device_result(float** d_result, size_t size) {
    cudaMalloc((void**)d_result, size);
}

void copy_result_to_host(float* h_result, float* d_result, size_t size) {
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
}

void free_device_memory(void* ptr) {
    cudaFree(ptr);
}

// ─────────────────────────────────────────────────────────────
// Entry Point (C interface)
// ─────────────────────────────────────────────────────────────
extern "C" EXPORT void match_template(const unsigned char* img, int img_w, int img_h,
                                      const unsigned char* tmpl, int tmpl_w, int tmpl_h,
                                      float tmpl_mean, float tmpl_var,
                                      float* result) {
    unsigned char *d_img = nullptr, *d_tmpl = nullptr;
    float *d_result = nullptr;

    size_t img_size  = img_w * img_h * sizeof(unsigned char);
    size_t tmpl_size = tmpl_w * tmpl_h * sizeof(unsigned char);
    size_t res_size  = img_w * img_h * sizeof(float);

    allocate_and_copy_to_device(img, img_size, &d_img);
    allocate_and_copy_to_device(tmpl, tmpl_size, &d_tmpl);
    allocate_device_result(&d_result, res_size);

    launch_ncc_kernel(d_img, img_w, img_h, d_tmpl, tmpl_w, tmpl_h,
                      tmpl_mean, tmpl_var, d_result);

    copy_result_to_host(result, d_result, res_size);

    free_device_memory(d_img);
    free_device_memory(d_tmpl);
    free_device_memory(d_result);
}
