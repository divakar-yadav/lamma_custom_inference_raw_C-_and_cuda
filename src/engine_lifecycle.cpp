#include "llama_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace llama_cpp_ref {

LlamaEngine::LlamaEngine() = default;

LlamaEngine::~LlamaEngine() {
    if (data_ && data_ != MAP_FAILED) {
        munmap(data_, file_size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
    free_run_state();
}

void LlamaEngine::malloc_run_state() {
    int kv_dim = (config_.dim * config_.n_kv_heads) / config_.n_heads;
    // These buffers represent the "activation wave" for one decoded token.
    state_.x = (float*)std::calloc((size_t)config_.dim, sizeof(float));
    state_.xb = (float*)std::calloc((size_t)config_.dim, sizeof(float));
    state_.xb2 = (float*)std::calloc((size_t)config_.dim, sizeof(float));
    state_.hb = (float*)std::calloc((size_t)config_.hidden_dim, sizeof(float));
    state_.hb2 = (float*)std::calloc((size_t)config_.hidden_dim, sizeof(float));
    state_.q = (float*)std::calloc((size_t)config_.dim, sizeof(float));
    // k and v are not standalone allocations; they are aliases into KV cache rows.
    state_.k = nullptr;
    state_.v = nullptr;
    state_.key_cache = (float*)std::calloc((size_t)config_.n_layers * config_.seq_len * kv_dim, sizeof(float));
    state_.value_cache = (float*)std::calloc((size_t)config_.n_layers * config_.seq_len * kv_dim, sizeof(float));
    state_.att = (float*)std::calloc((size_t)config_.n_heads * config_.seq_len, sizeof(float));
    state_.logits = (float*)std::calloc((size_t)config_.vocab_size, sizeof(float));

    if (!state_.x || !state_.xb || !state_.xb2 || !state_.hb || !state_.hb2 || !state_.q ||
        !state_.key_cache || !state_.value_cache || !state_.att || !state_.logits) {
        throw std::runtime_error("malloc failed in run state");
    }
}

void LlamaEngine::free_run_state() {
    std::free(state_.x); state_.x = nullptr;
    std::free(state_.xb); state_.xb = nullptr;
    std::free(state_.xb2); state_.xb2 = nullptr;
    std::free(state_.hb); state_.hb = nullptr;
    std::free(state_.hb2); state_.hb2 = nullptr;
    std::free(state_.q); state_.q = nullptr;
    state_.k = nullptr;
    state_.v = nullptr;
    std::free(state_.att); state_.att = nullptr;
    std::free(state_.logits); state_.logits = nullptr;
    std::free(state_.key_cache); state_.key_cache = nullptr;
    std::free(state_.value_cache); state_.value_cache = nullptr;
}

void LlamaEngine::memory_map_weights(float* ptr, int shared_weights) {
    int head_size = config_.dim / config_.n_heads;
    unsigned long long n_layers = (unsigned long long)config_.n_layers;

    // Checkpoint layout is contiguous; we walk pointer offsets in exact export order.
    weights_.token_embedding_table = ptr;
    ptr += config_.vocab_size * config_.dim;
    weights_.rms_att_weight = ptr;
    ptr += n_layers * config_.dim;
    weights_.wq = ptr;
    ptr += n_layers * config_.dim * (config_.n_heads * head_size);
    weights_.wk = ptr;
    ptr += n_layers * config_.dim * (config_.n_kv_heads * head_size);
    weights_.wv = ptr;
    ptr += n_layers * config_.dim * (config_.n_kv_heads * head_size);
    weights_.wo = ptr;
    ptr += n_layers * (config_.n_heads * head_size) * config_.dim;
    weights_.rms_ffn_weight = ptr;
    ptr += n_layers * config_.dim;
    weights_.w1 = ptr;
    ptr += n_layers * config_.dim * config_.hidden_dim;
    weights_.w2 = ptr;
    ptr += n_layers * config_.hidden_dim * config_.dim;
    weights_.w3 = ptr;
    ptr += n_layers * config_.dim * config_.hidden_dim;
    weights_.rms_final_weight = ptr;
    ptr += config_.dim;
    ptr += config_.seq_len * head_size / 2; // old rope real
    ptr += config_.seq_len * head_size / 2; // old rope imag
    weights_.wcls = shared_weights ? weights_.token_embedding_table : ptr;
}

void LlamaEngine::load_model(const std::string& checkpoint_path) {
    FILE* file = std::fopen(checkpoint_path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("Could not open checkpoint: " + checkpoint_path);
    }

    if (std::fread(&config_, sizeof(Config), 1, file) != 1) {
        std::fclose(file);
        throw std::runtime_error("Failed reading Config header");
    }

    // Legacy format uses sign of vocab_size to encode tied output embedding.
    int shared_weights = config_.vocab_size > 0 ? 1 : 0;
    config_.vocab_size = std::abs(config_.vocab_size);

    std::fseek(file, 0, SEEK_END);
    file_size_ = std::ftell(file);
    std::fclose(file);

    fd_ = open(checkpoint_path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("open failed on checkpoint");
    }

    data_ = (float*)mmap(nullptr, (size_t)file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        throw std::runtime_error("mmap failed on checkpoint");
    }

    // Weights start immediately after Config header.
    float* weights_ptr = data_ + sizeof(Config) / sizeof(float);
    memory_map_weights(weights_ptr, shared_weights);
    malloc_run_state();
}

} // namespace llama_cpp_ref
