#include "llama_engine.hpp"

#include <cmath>

namespace llama_cpp_ref {

void LlamaEngine::rmsnorm(float* o, const float* x, const float* weight, int size) {
    // RMSNorm uses root mean square only (no mean-centering like LayerNorm).
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / std::sqrt(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void LlamaEngine::softmax(float* x, int size) {
    // Numerically stable softmax: subtract max(logit) before exp().
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void LlamaEngine::matmul(float* xout, const float* x, const float* w, int n, int d) {
    // Reference GEMV: W[d,n] @ x[n] -> xout[d].
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

} // namespace llama_cpp_ref
