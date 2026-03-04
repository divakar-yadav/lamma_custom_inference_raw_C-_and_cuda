#pragma once

#include <cuda_runtime.h>

namespace llama_cpp_ref {
namespace cuda_kernels {

// 1) RMSNorm
void launch_rmsnorm(
    const float* x,
    const float* weight,
    float* out,
    int dim,
    float eps,
    cudaStream_t stream
);

// 2) Matrix-vector multiply: out[d] = W[d, n] @ x[n]
void launch_matvec(
    const float* w,
    const float* x,
    float* out,
    int n,
    int d,
    cudaStream_t stream
);

// 3) RoPE rotation on q and k (k_dim can be < q_dim in GQA)
void launch_rope_qk(
    float* q,
    float* k,
    int q_dim,
    int k_dim,
    int head_size,
    int pos,
    cudaStream_t stream
);

// 4) Attention scores for one head over [0, pos]
void launch_attention_scores(
    const float* q_head,
    const float* key_cache_head, // contiguous [(pos+1), head_size]
    float* scores,
    int head_size,
    int timesteps,
    cudaStream_t stream
);

// 5) Softmax over one vector in place
void launch_softmax_inplace(
    float* x,
    int size,
    cudaStream_t stream
);

// 6) Weighted sum of values for one head: out = sum_t att[t] * v[t]
void launch_attention_weighted_sum(
    const float* att,
    const float* value_cache_head, // contiguous [timesteps, head_size]
    float* out,
    int head_size,
    int timesteps,
    cudaStream_t stream
);

// 7) SwiGLU elementwise: hb[i] = silu(hb[i]) * hb2[i]
void launch_swiglu(
    float* hb,
    const float* hb2,
    int hidden_dim,
    cudaStream_t stream
);

} // namespace cuda_kernels
} // namespace llama_cpp_ref
