#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llama_cpp_ref {

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

class Tokenizer {
public:
    Tokenizer() = default;
    void load(const std::string& tokenizer_path, int vocab_size);
    std::vector<int> encode(const std::string& text, bool bos, bool eos) const;
    std::string decode_piece(int prev_token, int token) const;
    int vocab_size() const { return vocab_size_; }

private:
    struct TokenIndex {
        char* str;
        int id;
    };

    static int compare_tokens(const void* a, const void* b);
    int str_lookup(const char* str) const;

    int vocab_size_ = 0;
    unsigned int max_token_length_ = 0;
    std::vector<char*> vocab_;
    std::vector<float> vocab_scores_;
    mutable std::vector<TokenIndex> sorted_vocab_;
    mutable bool sorted_ready_ = false;
    unsigned char byte_pieces_[512]{};
};

class LlamaEngine {
public:
    LlamaEngine();
    ~LlamaEngine();

    void load_model(const std::string& checkpoint_path);
    const Config& config() const { return config_; }

    std::vector<float> forward(int token, int pos);

    std::string generate(
        const Tokenizer& tokenizer,
        const std::string& prompt,
        int steps,
        float temperature,
        float topp,
        uint64_t seed
    );

private:
    struct TransformerWeights {
        float* token_embedding_table = nullptr;
        float* rms_att_weight = nullptr;
        float* rms_ffn_weight = nullptr;
        float* wq = nullptr;
        float* wk = nullptr;
        float* wv = nullptr;
        float* wo = nullptr;
        float* w1 = nullptr;
        float* w2 = nullptr;
        float* w3 = nullptr;
        float* rms_final_weight = nullptr;
        float* wcls = nullptr;
    };

    struct RunState {
        float* x = nullptr;
        float* xb = nullptr;
        float* xb2 = nullptr;
        float* hb = nullptr;
        float* hb2 = nullptr;
        float* q = nullptr;
        float* k = nullptr;
        float* v = nullptr;
        float* att = nullptr;
        float* logits = nullptr;
        float* key_cache = nullptr;
        float* value_cache = nullptr;
    };

    struct ProbIndex {
        float prob;
        int index;
    };

    static void rmsnorm(float* o, const float* x, const float* weight, int size);
    static void softmax(float* x, int size);
    static void matmul(float* xout, const float* x, const float* w, int n, int d);
    static int sample_argmax(const float* probabilities, int n);
    static int sample_mult(const float* probabilities, int n, float coin);
    static int compare_prob_index(const void* a, const void* b);
    static int sample_topp(const float* probabilities, int n, float topp, ProbIndex* probindex, float coin);
    static uint32_t random_u32(uint64_t* state);
    static float random_f32(uint64_t* state);

    int sample(float* logits, float temperature, float topp, uint64_t* rng_state, std::vector<ProbIndex>& probindex);

    void malloc_run_state();
    void free_run_state();
    void memory_map_weights(float* ptr, int shared_weights);

    Config config_{};
    TransformerWeights weights_{};
    RunState state_{};

    int fd_ = -1;
    float* data_ = nullptr;
    ssize_t file_size_ = 0;
};

} // namespace llama_cpp_ref
