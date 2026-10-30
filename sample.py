"""Generate stories from a trained checkpoint.

Example: python sample.py --prompt "Once upon a time" --n 3
"""
import argparse
import os

import tiktoken
import torch

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("--n", type=int, default=3, help="number of samples")
p.add_argument("--tokens", type=int, default=300)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--ckpt", default=os.path.join(HERE, "out/ckpt.pt"))
args = p.parse_args()

enc = tiktoken.get_encoding("gpt2")
ckpt = torch.load(args.ckpt, map_location="cpu")
model = GPT(GPTConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.to("cuda").eval()
print(f"checkpoint: step {ckpt['iter']}, best val loss {ckpt['best_val']:.3f}\n")

for i in range(args.n):
    idx = torch.tensor([enc.encode_ordinary(args.prompt)], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(idx, args.tokens, temperature=args.temperature,
                             stop_token=enc.eot_token)
    ids = [t for t in out[0].tolist() if t != enc.eot_token]
    print(f"=== sample {i + 1} ===\n{enc.decode(ids)}\n")
