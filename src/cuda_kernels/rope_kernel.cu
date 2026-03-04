#include "cuda_kernels/llama_cuda_kernels.cuh"

#include <cuda_runtime.h>
#include <math.h>

namespace llama_cpp_ref {
namespace cuda_kernels {

__global__ void rope_qk_kernel(float* q, float* k, int q_dim, int k_dim, int head_size, int pos) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 >= q_dim) return;

    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float c = cosf(val);
    float s = sinf(val);

    // rotate q
    float q0 = q[i];
    float q1 = q[i + 1];
    q[i] = q0 * c - q1 * s;
    q[i + 1] = q0 * s + q1 * c;

    // rotate k only where valid in kv dimension
    if (i + 1 < k_dim) {
        float k0 = k[i];
        float k1 = k[i + 1];
        k[i] = k0 * c - k1 * s;
        k[i + 1] = k0 * s + k1 * c;
    }
}

void launch_rope_qk(float* q, float* k, int q_dim, int k_dim, int head_size, int pos, cudaStream_t stream) {
    int pairs = (q_dim + 1) / 2;
    int threads = 256;
    int blocks = (pairs + threads - 1) / threads;
    rope_qk_kernel<<<blocks, threads, 0, stream>>>(q, k, q_dim, k_dim, head_size, pos);
}

} // namespace cuda_kernels
} // namespace llama_cpp_ref
