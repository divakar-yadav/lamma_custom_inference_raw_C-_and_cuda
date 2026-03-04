# llama_inference_cpp

A brand-new C++17 repository for Llama-style inference from `.bin` checkpoints.

This project implements:
- pure C++ tokenizer + inference path (no Python runtime dependency)
- full forward pass (RMSNorm, RoPE, attention, SwiGLU, logits)
- deterministic regression tests for tokenization and generation
- explainer docs in TXT + PDF (LaTeX)

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run inference

```bash
./build/llama_infer_cpp /path/to/model.bin --tokenizer /path/to/tokenizer.bin --steps 128 --temperature 1.0 --topp 0.9
```

## Download test models

```bash
./scripts/download_test_models.sh
```

## Run tests

```bash
./build/tokenizer_test ./models/tokenizer.bin
./build/generation_test ./models/stories260K.bin ./models/tok512.bin
```

## Deterministic reference

`generation_test` validates output against a known good greedy (temperature=0) output from the tiny `stories260K` model.

## Docs

- Text: `docs/EXPLAINER.txt`
- PDF: `docs/EXPLAINER.pdf`
- LaTeX source: `docs/EXPLAINER.tex`
