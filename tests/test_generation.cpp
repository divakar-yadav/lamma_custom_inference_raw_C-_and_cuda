#include "llama_engine.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace llama_cpp_ref;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: generation_test <stories260K.bin> <tok512.bin>\n";
        return 1;
    }

    const std::string expected =
        "Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\n"
        "Lily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"\n"
        "Lily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"\n"
        "Lily didn't want to help her mom, so she";

    LlamaEngine engine;
    engine.load_model(argv[1]);

    Tokenizer tok;
    tok.load(argv[2], engine.config().vocab_size);

    std::string actual = engine.generate(tok, "", 200, 0.0f, 0.9f, 1234);

    if (actual != expected) {
        std::cerr << "Generation output mismatch.\n";
        std::cerr << "Actual:\n" << actual << "\n";
        return 2;
    }

    std::cout << "Generation test passed.\n";
    return 0;
}
