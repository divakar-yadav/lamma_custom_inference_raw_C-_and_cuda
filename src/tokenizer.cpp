#include "llama_engine.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace llama_cpp_ref {

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

} // namespace llama_cpp_ref
