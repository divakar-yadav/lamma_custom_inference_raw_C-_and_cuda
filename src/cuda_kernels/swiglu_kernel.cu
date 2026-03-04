#include "cuda_kernels/llama_cuda_kernels.cuh"

#include <cuda_runtime.h>
#include <math.h>

namespace llama_cpp_ref {
namespace cuda_kernels {

__global__ void swiglu_kernel(float* hb, const float* hb2, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_dim) return;

    float x = hb[i];
    float silu = x * (1.0f / (1.0f + expf(-x)));
    hb[i] = silu * hb2[i];
}

void launch_swiglu(float* hb, const float* hb2, int hidden_dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, stream>>>(hb, hb2, hidden_dim);
}

} // namespace cuda_kernels
} // namespace llama_cpp_ref
