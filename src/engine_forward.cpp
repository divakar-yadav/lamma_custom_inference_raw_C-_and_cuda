#include "llama_engine.hpp"

#include <cmath>
#include <cstring>

namespace llama_cpp_ref {

std::vector<float> LlamaEngine::forward(int token, int pos) {
    float* x = state_.x;
    int dim = config_.dim;
    int kv_dim = (config_.dim * config_.n_kv_heads) / config_.n_heads;
    int kv_mul = config_.n_heads / config_.n_kv_heads;
    int hidden_dim = config_.hidden_dim;
    int head_size = dim / config_.n_heads;

    float* content_row = weights_.token_embedding_table + (size_t)token * dim;
    std::memcpy(x, content_row, (size_t)dim * sizeof(float));

    for (int l = 0; l < config_.n_layers; l++) {
        rmsnorm(state_.xb, x, weights_.rms_att_weight + (size_t)l * dim, dim);

        int loff = l * config_.seq_len * kv_dim;
        state_.k = state_.key_cache + loff + pos * kv_dim;
        state_.v = state_.value_cache + loff + pos * kv_dim;

        matmul(state_.q, state_.xb, weights_.wq + (size_t)l * dim * dim, dim, dim);
        matmul(state_.k, state_.xb, weights_.wk + (size_t)l * dim * kv_dim, dim, kv_dim);
        matmul(state_.v, state_.xb, weights_.wv + (size_t)l * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / std::pow(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = std::cos(val);
            float fci = std::sin(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? state_.q : state_.k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        for (int h = 0; h < config_.n_heads; h++) {
            float* q = state_.q + h * head_size;
            float* att = state_.att + (size_t)h * config_.seq_len;

            for (int t = 0; t <= pos; t++) {
                float* k = state_.key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= std::sqrt((float)head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float* xb = state_.xb + h * head_size;
            std::memset(xb, 0, (size_t)head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = state_.value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        matmul(state_.xb2, state_.xb, weights_.wo + (size_t)l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += state_.xb2[i];
        }

        rmsnorm(state_.xb, x, weights_.rms_ffn_weight + (size_t)l * dim, dim);

        matmul(state_.hb, state_.xb, weights_.w1 + (size_t)l * dim * hidden_dim, dim, hidden_dim);
        matmul(state_.hb2, state_.xb, weights_.w3 + (size_t)l * dim * hidden_dim, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; i++) {
            float val = state_.hb[i];
            val *= (1.0f / (1.0f + std::exp(-val)));
            val *= state_.hb2[i];
            state_.hb[i] = val;
        }

        matmul(state_.xb, state_.hb, weights_.w2 + (size_t)l * dim * hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += state_.xb[i];
        }
    }

    rmsnorm(x, x, weights_.rms_final_weight, dim);
    matmul(state_.logits, x, weights_.wcls, config_.dim, config_.vocab_size);

    return std::vector<float>(state_.logits, state_.logits + config_.vocab_size);
}

} // namespace llama_cpp_ref
