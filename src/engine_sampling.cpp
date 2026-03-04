#include "llama_engine.hpp"

#include <cstdlib>
#include <stdexcept>

namespace llama_cpp_ref {

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
