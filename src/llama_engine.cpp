#include "llama_engine.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace llama_cpp_ref {

// ---------------- Tokenizer ----------------

int Tokenizer::compare_tokens(const void* a, const void* b) {
    return std::strcmp(((const TokenIndex*)a)->str, ((const TokenIndex*)b)->str);
}

void Tokenizer::load(const std::string& tokenizer_path, int vocab_size) {
    vocab_size_ = vocab_size;
    vocab_.assign(vocab_size_, nullptr);
    vocab_scores_.assign(vocab_size_, 0.0f);
    sorted_vocab_.clear();
    sorted_ready_ = false;

    for (int i = 0; i < 256; i++) {
        byte_pieces_[i * 2] = (unsigned char)i;
        byte_pieces_[i * 2 + 1] = '\0';
    }

    FILE* file = std::fopen(tokenizer_path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("Failed to open tokenizer: " + tokenizer_path);
    }

    if (std::fread(&max_token_length_, sizeof(int), 1, file) != 1) {
        std::fclose(file);
        throw std::runtime_error("Failed reading max token length");
    }

    int len = 0;
    for (int i = 0; i < vocab_size_; i++) {
        if (std::fread(&vocab_scores_[i], sizeof(float), 1, file) != 1) {
            std::fclose(file);
            throw std::runtime_error("Failed reading vocab score");
        }
        if (std::fread(&len, sizeof(int), 1, file) != 1) {
            std::fclose(file);
            throw std::runtime_error("Failed reading token length");
        }
        vocab_[i] = (char*)std::malloc((size_t)len + 1);
        if (!vocab_[i]) {
            std::fclose(file);
            throw std::runtime_error("malloc failed for vocab token");
        }
        if (std::fread(vocab_[i], len, 1, file) != 1) {
            std::fclose(file);
            throw std::runtime_error("Failed reading token bytes");
        }
        vocab_[i][len] = '\0';
    }

    std::fclose(file);
}

int Tokenizer::str_lookup(const char* str) const {
    if (!sorted_ready_) {
        sorted_vocab_.resize((size_t)vocab_size_);
        for (int i = 0; i < vocab_size_; i++) {
            sorted_vocab_[(size_t)i].str = vocab_[(size_t)i];
            sorted_vocab_[(size_t)i].id = i;
        }
        std::qsort(sorted_vocab_.data(), (size_t)vocab_size_, sizeof(TokenIndex), compare_tokens);
        sorted_ready_ = true;
    }

    TokenIndex tok{const_cast<char*>(str), 0};
    TokenIndex* res = (TokenIndex*)std::bsearch(
        &tok, sorted_vocab_.data(), (size_t)vocab_size_, sizeof(TokenIndex), compare_tokens
    );
    return res ? res->id : -1;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool bos, bool eos) const {
    std::vector<int> tokens;
    tokens.reserve(text.size() + 8);

    if (bos) {
        tokens.push_back(1);
    }

    if (!text.empty()) {
        int dummy_prefix = str_lookup(" ");
        tokens.push_back(dummy_prefix);
    }

    size_t buf_cap = (size_t)max_token_length_ * 2 + 3;
    std::vector<char> str_buffer(buf_cap, 0);
    size_t str_len = 0;

    for (size_t cidx = 0; cidx < text.size(); cidx++) {
        unsigned char c = (unsigned char)text[cidx];
        if ((c & 0xC0) != 0x80) {
            str_len = 0;
        }

        if (str_len + 1 >= str_buffer.size()) {
            str_len = 0;
        }
        str_buffer[str_len++] = (char)c;
        str_buffer[str_len] = '\0';

        unsigned char next = (cidx + 1 < text.size()) ? (unsigned char)text[cidx + 1] : 0;
        if ((next & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer.data());
        if (id != -1) {
            tokens.push_back(id);
        } else {
            for (size_t i = 0; i < str_len; i++) {
                tokens.push_back((unsigned char)str_buffer[i] + 3);
            }
        }
        str_len = 0;
    }

    while (true) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (int)tokens.size() - 1; i++) {
            std::snprintf(str_buffer.data(), str_buffer.size(), "%s%s", vocab_[tokens[(size_t)i]], vocab_[tokens[(size_t)i + 1]]);
            int id = str_lookup(str_buffer.data());
            if (id != -1 && vocab_scores_[(size_t)id] > best_score) {
                best_score = vocab_scores_[(size_t)id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;
        }

        tokens[(size_t)best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    if (eos) {
        tokens.push_back(2);
    }

    return tokens;
}

std::string Tokenizer::decode_piece(int prev_token, int token) const {
    char* piece = vocab_[(size_t)token];
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }

    unsigned char byte_val = 0;
    if (std::sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)byte_pieces_ + byte_val * 2;
    }

    if (piece == nullptr || piece[0] == '\0') {
        return std::string();
    }

    if (piece[1] == '\0') {
        unsigned char bv = (unsigned char)piece[0];
        if (!(std::isprint(bv) || std::isspace(bv))) {
            return std::string();
        }
    }

    return std::string(piece);
}

// ---------------- LlamaEngine ----------------

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
    state_.x = (float*)std::calloc((size_t)config_.dim, sizeof(float));
    state_.xb = (float*)std::calloc((size_t)config_.dim, sizeof(float));
    state_.xb2 = (float*)std::calloc((size_t)config_.dim, sizeof(float));
    state_.hb = (float*)std::calloc((size_t)config_.hidden_dim, sizeof(float));
    state_.hb2 = (float*)std::calloc((size_t)config_.hidden_dim, sizeof(float));
    state_.q = (float*)std::calloc((size_t)config_.dim, sizeof(float));
    // k and v are aliases into key_cache/value_cache rows during forward().
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

    float* weights_ptr = data_ + sizeof(Config) / sizeof(float);
    memory_map_weights(weights_ptr, shared_weights);
    malloc_run_state();
}

void LlamaEngine::rmsnorm(float* o, const float* x, const float* weight, int size) {
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
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

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

int LlamaEngine::sample_argmax(const float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int LlamaEngine::sample_mult(const float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int LlamaEngine::compare_prob_index(const void* a, const void* b) {
    const ProbIndex* pa = (const ProbIndex*)a;
    const ProbIndex* pb = (const ProbIndex*)b;
    if (pa->prob > pb->prob) return -1;
    if (pa->prob < pb->prob) return 1;
    return 0;
}

int LlamaEngine::sample_topp(const float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }

    std::qsort(probindex, (size_t)n0, sizeof(ProbIndex), compare_prob_index);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

uint32_t LlamaEngine::random_u32(uint64_t* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

float LlamaEngine::random_f32(uint64_t* state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int LlamaEngine::sample(float* logits, float temperature, float topp, uint64_t* rng_state, std::vector<ProbIndex>& probindex) {
    if (temperature == 0.0f) {
        return sample_argmax(logits, config_.vocab_size);
    }

    for (int q = 0; q < config_.vocab_size; q++) {
        logits[q] /= temperature;
    }
    softmax(logits, config_.vocab_size);

    float coin = random_f32(rng_state);
    if (topp <= 0.0f || topp >= 1.0f) {
        return sample_mult(logits, config_.vocab_size, coin);
    }
    return sample_topp(logits, config_.vocab_size, topp, probindex.data(), coin);
}

std::string LlamaEngine::generate(
    const Tokenizer& tokenizer,
    const std::string& prompt,
    int steps,
    float temperature,
    float topp,
    uint64_t seed
) {
    std::vector<int> prompt_tokens = tokenizer.encode(prompt, true, false);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("Prompt tokenization returned empty sequence");
    }

    std::vector<ProbIndex> probindex((size_t)config_.vocab_size);
    uint64_t rng_state = seed;

    int token = prompt_tokens[0];
    int pos = 0;
    int next = token;

    std::string out;
    out.reserve((size_t)steps * 4);

    while (pos < steps) {
        auto logits_vec = forward(token, pos);
        float* logits = logits_vec.data();

        if (pos < (int)prompt_tokens.size() - 1) {
            next = prompt_tokens[(size_t)pos + 1];
        } else {
            next = sample(logits, temperature, topp, &rng_state, probindex);
        }
        pos++;

        if (next == 1) {
            break;
        }

        std::string piece = tokenizer.decode_piece(token, next);
        out += piece;
        token = next;
    }

    return out;
}

} // namespace llama_cpp_ref
