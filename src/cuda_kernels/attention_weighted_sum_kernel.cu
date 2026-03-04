#include "cuda_kernels/llama_cuda_kernels.cuh"

#include <cuda_runtime.h>

namespace llama_cpp_ref {
namespace cuda_kernels {

__global__ void attention_weighted_sum_kernel(
    const float* att,
    const float* value_cache_head,
    float* out,
    int head_size,
    int timesteps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= head_size) return;

    float acc = 0.0f;
    for (int t = 0; t < timesteps; t++) {
        const float* v = value_cache_head + (size_t)t * head_size;
        acc += att[t] * v[i];
    }
    out[i] = acc;
}

void launch_attention_weighted_sum(
    const float* att,
    const float* value_cache_head,
    float* out,
    int head_size,
    int timesteps,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (head_size + threads - 1) / threads;
    attention_weighted_sum_kernel<<<blocks, threads, 0, stream>>>(att, value_cache_head, out, head_size, timesteps);
}

} // namespace cuda_kernels
} // namespace llama_cpp_ref
