#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"
mkdir -p "$MODEL_DIR"

base="https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K"

wget -nc "$base/stories260K.bin" -O "$MODEL_DIR/stories260K.bin"
wget -nc "$base/tok512.bin" -O "$MODEL_DIR/tok512.bin"

# Prefer local tokenizer.bin if llama2.c repo is present on this machine.
if [ -f "/home/azureuser/divakar_projects/llama2.c/tokenizer.bin" ]; then
  cp -f "/home/azureuser/divakar_projects/llama2.c/tokenizer.bin" "$MODEL_DIR/tokenizer.bin"
else
  wget -nc "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M/tokenizer.bin" -O "$MODEL_DIR/tokenizer.bin"
fi

echo "Downloaded models into $MODEL_DIR"
