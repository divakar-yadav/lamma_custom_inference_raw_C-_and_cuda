#include "llama_engine.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

using namespace llama_cpp_ref;

static void usage() {
    std::cerr << "Usage: llama_infer_cpp <checkpoint.bin> [options]\n"
              << "Options:\n"
              << "  --tokenizer <path>   tokenizer.bin path (default: tokenizer.bin)\n"
              << "  --prompt <text>      input prompt (default: empty)\n"
              << "  --steps <int>        number of generation steps (default: 128)\n"
              << "  --temperature <f>    sampling temperature (default: 1.0, 0.0=greedy)\n"
              << "  --topp <f>           top-p sampling (default: 0.9)\n"
              << "  --seed <int>         rng seed (default: time)\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string checkpoint = argv[1];
    std::string tokenizer_path = "tokenizer.bin";
    std::string prompt;
    int steps = 128;
    float temperature = 1.0f;
    float topp = 0.9f;
    uint64_t seed = (uint64_t)std::time(nullptr);

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc) temperature = std::atof(argv[++i]);
        else if (arg == "--topp" && i + 1 < argc) topp = std::atof(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = (uint64_t)std::strtoull(argv[++i], nullptr, 10);
        else {
            usage();
            return 1;
        }
    }

    try {
        LlamaEngine engine;
        engine.load_model(checkpoint);

        Tokenizer tokenizer;
        tokenizer.load(tokenizer_path, engine.config().vocab_size);

        std::string text = engine.generate(tokenizer, prompt, steps, temperature, topp, seed);
        std::cout << text << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }

    return 0;
}
