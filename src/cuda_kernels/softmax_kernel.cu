#include "cuda_kernels/llama_cuda_kernels.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace llama_cpp_ref {
namespace cuda_kernels {

__global__ void softmax_inplace_kernel(float* x, int size) {
    // Single-block reference softmax for one head score vector.
    __shared__ float max_val;
    __shared__ float sum;

    if (threadIdx.x == 0) {
        max_val = -FLT_MAX;
        for (int i = 0; i < size; i++) {
            max_val = fmaxf(max_val, x[i]);
        }
        float s = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i] = expf(x[i] - max_val);
            s += x[i];
        }
        sum = s;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        x[i] /= sum;
    }
}

void launch_softmax_inplace(float* x, int size, cudaStream_t stream) {
    softmax_inplace_kernel<<<1, 256, 0, stream>>>(x, size);
}

} // namespace cuda_kernels
} // namespace llama_cpp_ref
