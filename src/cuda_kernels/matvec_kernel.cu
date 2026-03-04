#include "cuda_kernels/llama_cuda_kernels.cuh"

#include <cuda_runtime.h>

namespace llama_cpp_ref {
namespace cuda_kernels {

__global__ void matvec_kernel(const float* w, const float* x, float* out, int n, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d) return;

    float acc = 0.0f;
    const float* w_row = w + (size_t)row * n;
    for (int j = 0; j < n; j++) {
        acc += w_row[j] * x[j];
    }
    out[row] = acc;
}

void launch_matvec(const float* w, const float* x, float* out, int n, int d, cudaStream_t stream) {
    int threads = 256;
    int blocks = (d + threads - 1) / threads;
    matvec_kernel<<<blocks, threads, 0, stream>>>(w, x, out, n, d);
}

} // namespace cuda_kernels
} // namespace llama_cpp_ref
