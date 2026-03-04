#include "cuda_kernels/llama_cuda_kernels.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace llama_cpp_ref {
namespace cuda_kernels {

__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int dim, float eps) {
    // Single-block reference kernel for one token vector.
    __shared__ float ss;
    if (threadIdx.x == 0) {
        float sumsq = 0.0f;
        for (int i = 0; i < dim; i++) {
            float v = x[i];
            sumsq += v * v;
        }
        ss = rsqrtf(sumsq / dim + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[i] = weight[i] * (x[i] * ss);
    }
}

void launch_rmsnorm(const float* x, const float* weight, float* out, int dim, float eps, cudaStream_t stream) {
    rmsnorm_kernel<<<1, 256, 0, stream>>>(x, weight, out, dim, eps);
}

} // namespace cuda_kernels
} // namespace llama_cpp_ref
