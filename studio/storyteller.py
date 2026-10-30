"""The Storyteller pipeline: Qwen writes an opening, your tiny GPT writes the
story, and Qwen checks the story against the request (retrying if it fails).

tell() yields small event dicts that the web app streams to the browser.
"""
import json
import os
import time
import urllib.request

import tiktoken
import torch

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "out", "ckpt.pt")
OLLAMA = "http://127.0.0.1:11434/api/chat"
QWEN = "qwen3:14b"
MAX_ATTEMPTS = 3
MAX_NEW_TOKENS = 400

OPENING_PROMPT = """You write the opening of a short story for small children, in the \
style of the TinyStories dataset: very simple words a 3-year-old knows. Start with \
"Once upon a time". Introduce the main character and the setting from the request. \
Write one or two short sentences, 30 words at most. Do not finish the story. \
Output only the sentences."""

CHECK_PROMPT = """You review a short children's story written for a request. \
Derive 2 to 4 short checks from the request (for example "Has a unicorn", \
"Happy ending", "Bedtime mood") and always include "Stays on topic" (the main \
character does not turn into something else). Decide fairly whether the story \
passes each check. Be strict about the ending if the request asks for one."""

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {"type": "array", "items": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "pass": {"type": "boolean"}},
            "required": ["label", "pass"]}},
        "summary": {"type": "string"},
    },
    "required": ["checks", "summary"],
}

enc = tiktoken.get_encoding("gpt2")


class TinyGPT:
    """Loads the trained checkpoint and generates one token at a time."""

    def __init__(self, path=CKPT):
        ckpt = torch.load(path, map_location="cpu")
        self.model = GPT(GPTConfig(**ckpt["config"]))
        self.model.load_state_dict(ckpt["model"])
        self.model.to("cuda").eval()
        self.info = {"params_m": round(self.model.num_params() / 1e6, 1), "step": ckpt["iter"],
                     "val_loss": round(ckpt["best_val"], 3), "context": self.model.cfg.block_size,
                     "gpu": torch.cuda.get_device_name(0)}
        del ckpt

    @torch.no_grad()
    def stream(self, prompt_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=0.8, top_k=50):
        block = self.model.cfg.block_size
        idx = torch.tensor([prompt_ids], device="cuda")
        for _ in range(max_new_tokens):
            if idx.size(1) >= block:
                return  # context window is full
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = self.model(idx)
            logits = logits[:, -1, :].float() / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            token = nxt.item()
            if token == enc.eot_token:
                return
            idx = torch.cat([idx, nxt], dim=1)
            yield token


def ask_qwen(system, user, temperature=0.7, schema=None):
    body = {"model": QWEN, "stream": False, "think": False, "keep_alive": "30m",
            "options": {"temperature": temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if schema:
        body["format"] = schema
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.load(r)
    return res["message"]["content"].strip(), res.get("eval_count", 0)


def tell(gpt: TinyGPT, request: str):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        yield {"type": "attempt", "attempt": attempt, "max": MAX_ATTEMPTS}

        # Step 1: Qwen writes the opening
        t0 = time.time()
        opening, qwen_tokens = ask_qwen(OPENING_PROMPT, request, temperature=0.9)
        yield {"type": "opening", "text": opening, "tokens": qwen_tokens,
               "seconds": round(time.time() - t0, 2)}

        # Step 2: tiny GPT continues it, streamed token by token
        prompt_ids = enc.encode_ordinary(opening)
        new_ids, sent = [], ""
        t0 = time.time()
        for token in gpt.stream(prompt_ids):
            new_ids.append(token)
            # Decode the whole continuation so multi-token characters come out whole.
            text = enc.decode(new_ids)
            if text.endswith("�"):
                continue
            elapsed = max(time.time() - t0, 1e-6)
            yield {"type": "token", "text": text[len(sent):], "count": len(new_ids),
                   "context": len(prompt_ids) + len(new_ids), "tok_s": round(len(new_ids) / elapsed)}
            sent = text
        seconds = time.time() - t0
        story = opening + enc.decode(new_ids)
        yield {"type": "story_done", "tokens": len(new_ids), "prompt_tokens": len(prompt_ids),
               "seconds": round(seconds, 2), "tok_s": round(len(new_ids) / max(seconds, 1e-6))}

        # Step 3: Qwen checks the story against the request
        verdict, _ = ask_qwen(CHECK_PROMPT, f"Request: {request}\n\nStory:\n{story}",
                              temperature=0.0, schema=CHECK_SCHEMA)
        result = json.loads(verdict)
        passed = bool(result["checks"]) and all(c["pass"] for c in result["checks"])
        yield {"type": "check", "checks": result["checks"], "summary": result["summary"],
               "passed": passed}
        if passed or attempt == MAX_ATTEMPTS:
            yield {"type": "done", "passed": passed, "attempts": attempt, "story": story,
                   "opening": opening}
            return
        yield {"type": "retry", "attempt": attempt}
