"""Tokenize TinyStories into uint16 token files for training.

Reads data/TinyStoriesV2-GPT4-{train,valid}.txt, encodes each story with the
GPT-2 tokenizer, ends each story with <|endoftext|>, and writes
data/train.bin and data/val.bin.
"""
import os
from multiprocessing import Pool

import numpy as np
import tiktoken

DATA = os.path.join(os.path.dirname(__file__), "data")
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def encode_batch(stories):
    out = []
    for ids in enc.encode_ordinary_batch(stories):
        out.extend(ids)
        out.append(EOT)
    return np.array(out, dtype=np.uint16)


def stories_in(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return [s.strip() for s in text.split("<|endoftext|>") if s.strip()]


def prepare(src, dst):
    stories = stories_in(os.path.join(DATA, src))
    batches = [stories[i:i + 5000] for i in range(0, len(stories), 5000)]
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        parts = pool.map(encode_batch, batches)
    tokens = np.concatenate(parts)
    tokens.tofile(os.path.join(DATA, dst))
    print(f"{dst}: {len(stories):,} stories, {len(tokens):,} tokens")


if __name__ == "__main__":
    prepare("TinyStoriesV2-GPT4-valid.txt", "val.bin")
    prepare("TinyStoriesV2-GPT4-train.txt", "train.bin")
