#include "llama_engine.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llama_cpp_ref;

static void assert_eq(const std::vector<int>& actual, const std::vector<int>& expected, const std::string& name) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + ": size mismatch");
    }
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(name + ": mismatch at index " + std::to_string(i));
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: tokenizer_test <tokenizer.bin>\n";
        return 1;
    }

    Tokenizer tok;
    tok.load(argv[1], 32000);

    assert_eq(tok.encode("", true, false), {1}, "test0");
    assert_eq(tok.encode("I believe the meaning of life is", true, false), {1,306,4658,278,6593,310,2834,338}, "test1");
    assert_eq(tok.encode("Simply put, the theory of relativity states that ", true, false), {1,3439,17632,1925,29892,278,6368,310,14215,537,5922,393,29871}, "test2");

    std::cout << "Tokenizer tests passed.\n";
    return 0;
}
