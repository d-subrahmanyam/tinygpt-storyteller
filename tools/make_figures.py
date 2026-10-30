"""Generate the charts and pictures used in docs/ from real logs, data and the model.

Run from the tinygpt folder after training:  python tools/make_figures.py
"""
import math
import os
import re
import sqlite3
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")
sys.path.insert(0, ROOT)
os.makedirs(OUT, exist_ok=True)

INK, MUTED, ACCENT, GREEN, ORANGE = "#2b2433", "#7a7184", "#7c4dff", "#0f8a6c", "#e8833a"
TOKEN_COLORS = ["#ffd6e0", "#d6ecff", "#dff7d8", "#fff1c9", "#eadcff"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.grid": True, "grid.alpha": 0.25,
})
enc = tiktoken.get_encoding("gpt2")


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def parse_train_log():
    steps, loss, toks, temp, val = [], [], [], [], []
    for line in open(os.path.join(ROOT, "logs", "train.log")):
        m = re.match(r"step\s+(\d+) \| loss ([\d.]+) \| lr \S+ \| (\d+)k tok/s \| (\d+)C", line)
        if m:
            steps.append(int(m[1])); loss.append(float(m[2]))
            toks.append(int(m[3])); temp.append(int(m[4]))
        m = re.match(r"--- step (\d+): val loss ([\d.]+)", line)
        if m:
            val.append((int(m[1]), float(m[2])))
    return steps, loss, toks, temp, val


def loss_curve():
    steps, loss, _, _, val = parse_train_log()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(steps, loss, color=ACCENT, alpha=0.55, lw=1.2, label="training loss (one batch)")
    vs, vl = zip(*val)
    ax.plot(vs, vl, "o-", color=GREEN, lw=2, ms=4, label="validation loss (unseen stories)")
    ax.axhline(math.log(50257), color=MUTED, ls=":", lw=1)
    ax.text(12000, math.log(50257) - 0.15, "random guessing = ln(50257) ≈ 10.8", ha="right", va="top", color=MUTED, fontsize=9)
    ax.annotate(f"final val loss {vl[-1]:.3f}", (vs[-1], vl[-1]), xytext=(-150, 60),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
    ax.annotate("step 1,000: already writes\nsentences and names", (1000, 1.807), xytext=(40, 70),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
    ax.set(title="How the loss fell during 70 minutes of training", xlabel="training step", ylabel="loss (lower is better)", ylim=(0.8, 11.5))
    ax.legend(loc="center right", frameon=False)
    save(fig, "loss-curve.png")


def lr_schedule():
    lr, warmup, iters = 1e-3, 500, 12000
    def lr_at(it):
        if it < warmup:
            return lr * (it + 1) / warmup
        p = min(1.0, (it - warmup) / (iters - warmup))
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    xs = np.arange(iters)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(xs, [lr_at(i) for i in xs], color=ORANGE, lw=2.2)
    ax.axvspan(0, warmup, color=ORANGE, alpha=0.1)
    ax.text(warmup / 2, 2e-4, "warm-up", ha="center", color=ORANGE, fontsize=10)
    ax.text(6500, 7.5e-4, "cosine decay: big steps early,\ncareful small steps at the end", color=INK, fontsize=10)
    ax.set(title="Learning-rate schedule", xlabel="training step", ylabel="learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    save(fig, "lr-schedule.png")


def throughput_and_temp():
    steps, _, toks, temp, _ = parse_train_log()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(steps, toks, color=ACCENT, lw=1.5, label="tokens/sec (thousands)")
    ax.set(ylabel="k tokens / sec", xlabel="training step", ylim=(0, 110),
           title="Speed and GPU hotspot temperature during training")
    ax2 = ax.twinx()
    ax2.plot(steps, temp, color=ORANGE, lw=1.5, label="hotspot °C")
    ax2.set_ylim(40, 110)
    ax2.set_ylabel("hotspot °C", color=ORANGE)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.text(200, 12, "dips = validation checks every 500 steps", color=MUTED, fontsize=9)
    fig.legend(loc="lower right", bbox_to_anchor=(0.88, 0.25), frameon=False)
    save(fig, "throughput-temp.png")


def stress_test():
    t, w, junc = [], [], []
    for line in open(os.path.join(ROOT, "logs", "stress_test.log")):
        m = re.match(r"\[\s*(\d+)s\]\s+(\d+)W .*junction=(\d+)C", line)
        if m:
            t.append(int(m[1])); w.append(int(m[2])); junc.append(int(m[3]))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(t, junc, color=ORANGE, lw=2, label="hotspot °C")
    ax.plot(t, w, color=ACCENT, lw=1, alpha=0.6, label="power (W)")
    ax.axhline(110, color="#d64545", ls=":", lw=1)
    ax.text(5, 113, "110 °C throttle point", color="#d64545", fontsize=9)
    ax.set(title="10-minute eGPU stress test: 0 errors", xlabel="seconds", ylim=(0, 320))
    ax.legend(loc="lower right", frameon=False)
    save(fig, "stress-test.png")


def token_picture():
    text = "Once upon a time, there was a unicorn named Poppy."
    ids = enc.encode_ordinary(text)
    fig, ax = plt.subplots(figsize=(11, 1.9))
    ax.axis("off")
    x = 0.0
    for i, t in enumerate(ids):
        piece = enc.decode([t]).replace(" ", "␣")
        width = 0.012 * len(piece) + 0.03
        ax.add_patch(plt.Rectangle((x, 0.45), width, 0.4, color=TOKEN_COLORS[i % 5], ec="#bbb"))
        ax.text(x + width / 2, 0.65, piece, ha="center", va="center", fontsize=13, family="monospace", color=INK)
        ax.text(x + width / 2, 0.25, str(t), ha="center", va="center", fontsize=9, color=MUTED, family="monospace")
        x += width + 0.008
    ax.set_xlim(-0.01, x)
    ax.set_ylim(0, 1)
    ax.text(0, 0.98, f"{len(text)} characters → {len(ids)} tokens   (␣ = space; grey = token id)",
            fontsize=10, color=MUTED, va="top")
    save(fig, "tokens-example.png")


def story_lengths():
    db = sqlite3.connect(os.path.join(ROOT, "studio", "library.db"))
    n = np.array([r[0] for r in db.execute(
        "SELECT n_tokens FROM stories WHERE id % 10 = 0")])  # every 10th story
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(n, bins=np.arange(0, 1200, 20), color=ACCENT, alpha=0.8)
    ax.axvline(512, color=ORANGE, lw=2)
    share = (n <= 512).mean()
    ax.text(525, ax.get_ylim()[1] * 0.8, f"context = 512 tokens\n{share:.0%} of stories fit whole", color=ORANGE, fontsize=10)
    ax.set(title="How long are TinyStories? (tokens per story)", xlabel="tokens", ylabel="stories (10% sample)")
    save(fig, "story-lengths.png")


def next_token_probs():
    import torch
    from model import GPT, GPTConfig
    ckpt = torch.load(os.path.join(ROOT, "out", "ckpt.pt"), map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    prompt = "Once upon a time, there was a little"
    with torch.no_grad():
        logits, _ = model(torch.tensor([enc.encode_ordinary(prompt)]))
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    top = torch.topk(probs, 12)
    labels = [repr(enc.decode([i]))[1:-1] for i in top.indices.tolist()]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(range(12)[::-1], top.values.tolist(), color=ACCENT)
    ax.set_yticks(range(12)[::-1], labels, family="monospace")
    for y, p in zip(range(12)[::-1], top.values.tolist()):
        ax.text(p + 0.003, y, f"{p:.1%}", va="center", fontsize=9, color=INK)
    ax.set(title=f'What comes next after "{prompt} ..."?', xlabel="probability the model assigns")
    ax.grid(axis="y", visible=False)
    save(fig, "next-token.png")


if __name__ == "__main__":
    loss_curve()
    lr_schedule()
    throughput_and_temp()
    stress_test()
    token_picture()
    story_lengths()
    next_token_probs()
