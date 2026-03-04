#include "cuda_kernels/llama_cuda_kernels.cuh"

#include <cuda_runtime.h>
#include <math.h>

namespace llama_cpp_ref {
namespace cuda_kernels {

__global__ void attention_scores_kernel(
    const float* q_head,
    const float* key_cache_head,
    float* scores,
    int head_size,
    int timesteps
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= timesteps) return;

    const float* k = key_cache_head + (size_t)t * head_size;
    float acc = 0.0f;
    for (int i = 0; i < head_size; i++) {
        acc += q_head[i] * k[i];
    }
    scores[t] = acc / sqrtf((float)head_size);
}

void launch_attention_scores(
    const float* q_head,
    const float* key_cache_head,
    float* scores,
    int head_size,
    int timesteps,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (timesteps + threads - 1) / threads;
    attention_scores_kernel<<<blocks, threads, 0, stream>>>(q_head, key_cache_head, scores, head_size, timesteps);
}

} // namespace cuda_kernels
} // namespace llama_cpp_ref
