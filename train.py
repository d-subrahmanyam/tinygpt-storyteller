"""Train a small GPT on TinyStories on the RX 7900 XT.

Saves a checkpoint to out/ckpt.pt every --ckpt-every steps and on Ctrl+C,
so a run can be stopped and continued later with --resume.

Examples:
  uv run train.py --iters 200 --eval-every 100      # quick smoke test
  uv run train.py                                   # full run
  uv run train.py --resume                          # continue a stopped run
"""
import argparse
import glob
import math
import os
import signal
import time

import numpy as np
import torch

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=12000)
p.add_argument("--batch", type=int, default=32, help="sequences per micro-batch")
p.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
p.add_argument("--block", type=int, default=512)
p.add_argument("--layers", type=int, default=8)
p.add_argument("--heads", type=int, default=8)
p.add_argument("--embd", type=int, default=512)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--eval-every", type=int, default=500)
p.add_argument("--ckpt-every", type=int, default=1000)
p.add_argument("--log-every", type=int, default=50)
p.add_argument("--compile", action="store_true", help="use torch.compile (slow first start)")
p.add_argument("--resume", action="store_true")
p.add_argument("--out", default=os.path.join(HERE, "out"))
args = p.parse_args()

torch.manual_seed(1337)
dev = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
if "7900" not in gpu_name:
    print(f"warning: training on {gpu_name}; set HIP_VISIBLE_DEVICES to pick the 7900 XT")
torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs(args.out, exist_ok=True)
ckpt_path = os.path.join(args.out, "ckpt.pt")

# --- data ---------------------------------------------------------------
train_data = np.memmap(os.path.join(HERE, "data/train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(HERE, "data/val.bin"), dtype=np.uint16, mode="r")


def get_batch(data):
    ix = np.random.randint(0, len(data) - args.block - 1, args.batch)
    x = torch.from_numpy(np.stack([data[i:i + args.block] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + args.block] for i in ix]).astype(np.int64))
    return x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)


# --- model / optimizer --------------------------------------------------
start_iter, best_val = 0, float("inf")
if args.resume and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(dev)
    model.load_state_dict(ckpt["model"])
    start_iter, best_val = ckpt["iter"], ckpt["best_val"]
else:
    ckpt = None
    cfg = GPTConfig(block_size=args.block, n_layer=args.layers, n_head=args.heads, n_embd=args.embd)
    model = GPT(cfg).to(dev)

decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95))
if ckpt:
    opt.load_state_dict(ckpt["opt"])
    del ckpt

train_model = torch.compile(model) if args.compile else model
tokens_per_step = args.batch * args.accum * args.block
print(f"GPU: {gpu_name}")
print(f"model: {model.num_params() / 1e6:.1f}M params, {cfg.n_layer} layers, "
      f"{cfg.n_embd} dim, context {cfg.block_size}")
print(f"data: {len(train_data) / 1e6:.0f}M train tokens, {tokens_per_step:,} tokens/step, "
      f"{args.iters} steps = {args.iters * tokens_per_step / 1e6:.0f}M tokens")


def lr_at(it):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    progress = min(1.0, (it - args.warmup) / max(1, args.iters - args.warmup))
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_loss(data, batches=40):
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / batches


def gpu_sensors():
    hw = glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/temp2_input")
    for path in hw:
        dev_dir = path.split("/hwmon/")[0]
        try:
            if open(f"{dev_dir}/device").read().strip() != "0x744c":
                continue
            temp = int(open(path).read()) // 1000
            watts = int(open(path.replace("temp2_input", "power1_average")).read()) / 1e6
            return f"{temp}C {watts:.0f}W"
        except OSError:
            pass
    return ""


def save(it):
    tmp = ckpt_path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it,
                "best_val": best_val, "config": vars(cfg)}, tmp)
    os.replace(tmp, ckpt_path)


stop = False


def on_sigint(*_):
    global stop
    stop = True
    print("\nstopping after this step and saving a checkpoint...")


signal.signal(signal.SIGINT, on_sigint)

# --- training loop ------------------------------------------------------
model.train()
x, y = get_batch(train_data)
t0 = time.time()
it = start_iter
for it in range(start_iter, args.iters):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)

    for _ in range(args.accum):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = train_model(x, y)
        x, y = get_batch(train_data)  # fetch next batch while the GPU works
        (loss / args.accum).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

    if (it + 1) % args.log_every == 0:
        torch.cuda.synchronize()
        dt = time.time() - t0
        t0 = time.time()
        tok_s = tokens_per_step * args.log_every / dt
        print(f"step {it + 1:6d} | loss {loss.item():.3f} | lr {lr_at(it):.2e} | "
              f"{tok_s / 1e3:.0f}k tok/s | {gpu_sensors()}", flush=True)

    if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
        val = eval_loss(val_data)
        best_val = min(best_val, val)
        print(f"--- step {it + 1}: val loss {val:.3f} (best {best_val:.3f})", flush=True)

    if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iters or stop:
        save(it + 1)
        if stop:
            print(f"saved checkpoint at step {it + 1}; continue with --resume")
            break

print("done")
