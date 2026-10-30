#!/usr/bin/env bash
# Download the TinyStories V2 (GPT-4) text files into data/ (~2.2 GB).
set -euo pipefail
DATA="$(dirname "$0")/../data"
mkdir -p "$DATA" && cd "$DATA"
BASE=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
for f in TinyStoriesV2-GPT4-valid.txt TinyStoriesV2-GPT4-train.txt; do
  echo "downloading $f"
  curl -L -C - -o "$f" "$BASE/$f"
done
