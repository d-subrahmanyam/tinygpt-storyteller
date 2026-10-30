# Chapter 8 · Story Studio: a big model and a small model as a team

Our model writes nice stories but can't follow a request like *"a bedtime story with a unicorn and a happy ending."* Qwen3 14B understands requests perfectly, but it's 290× bigger. **Story Studio** uses each model for what it's good at:

```mermaid
flowchart LR
    U["🧒 'A bedtime story with<br/>a unicorn and a happy ending'"] --> Q1["🦙 Qwen3 14B<br/>writes the OPENING<br/>(understands the request)"]
    Q1 --> T["🧠 Our 51M model<br/>writes the STORY<br/>(token by token, streamed)"]
    T --> Q2["🦙 Qwen3 14B<br/>CHECKS it against<br/>the request"]
    Q2 -->|"✅ all checks pass"| DONE["Show the story"]
    Q2 -->|"❌ e.g. sad ending"| RETRY{"Tried 3 times?"}
    RETRY -->|no| Q1
    RETRY -->|yes| DONE
```

Run it:

```bash
uv run python -m studio.build_index   # once: makes studio/library.db (3.6 GB, about 2 minutes)
uv run python -m studio.app           # open http://127.0.0.1:8000
```

## ✨ The Storyteller tab

![Storyteller](images/studio-storyteller.png)

1. **Qwen writes the opening.** A system prompt asks for one or two sentences "in the style of TinyStories… simple words a 3-year-old knows… start with 'Once upon a time'". Matching the training data's style matters: our model continues text it recognizes far better.
2. **Our model writes the story,** streamed to the browser token by token. The green highlighted text is Qwen's opening; everything after it is ours. The live counter shows:
   - **🔤 tokens** generated so far, plus how many came from the opening
   - **⚡ tokens/sec**: our model manages 120–310 tokens/sec on the RX 7900 XT
   - **🪟 context used**: how much of the 512-token window is filled. When it's full, the model can't see any more text, so generation stops.
3. **Qwen checks it.** It's asked to derive 2–4 checks from the request ("Has a unicorn", "Happy ending", "Bedtime mood"), always plus "Stays on topic" (catching the unicorn-turns-into-a-penguin problem from [chapter 7](07-results.md)). It answers in **structured JSON** via Ollama's `format` option, so the app never has to parse free text. If any check fails, we start again with a fresh opening, up to 3 times.

## 📚 The Library tab

Search all 2.7M training stories in milliseconds, open any story, and press **🔤 Show tokens** to see exactly how the model saw it (see [chapter 4](04-tokenizing.md)).

## How it's built

```mermaid
flowchart TB
    subgraph Browser["Browser · plain HTML/CSS/JS, no build step"]
        LIB["📚 Library tab"]
        ST["✨ Storyteller tab<br/>EventSource (streaming)"]
    end
    subgraph Server["studio/app.py · FastAPI on 127.0.0.1:8000"]
        API1["/api/search, /api/story, /api/tokens, /api/random"]
        API2["/api/tell → Server-Sent Events"]
        API3["/api/save"]
    end
    LIB --> API1
    ST --> API2
    ST --> API3
    API1 --> DB[("library.db<br/>SQLite + FTS5 full-text index<br/>2.7M stories, 3.6 GB")]
    API2 --> PIPE["storyteller.py<br/>tell() pipeline"]
    PIPE --> GPT["TinyGPT<br/>out/ckpt.pt on the GPU"]
    PIPE --> OLL["Ollama :11434<br/>qwen3:14b"]
    API3 --> SAVED["saved/stories.jsonl"]
```

### Streaming with Server-Sent Events

`tell()` in [storyteller.py](../studio/storyteller.py) is a Python **generator**: it `yield`s small events as things happen, and the web app forwards each one to the browser immediately:

```mermaid
sequenceDiagram
    participant B as Browser
    participant A as FastAPI
    participant Q as Ollama (Qwen3)
    participant M as Our model (GPU)
    B->>A: GET /api/tell?request=…
    A->>Q: write an opening
    Q-->>A: "Once upon a time, a little unicorn named Luna…"
    A-->>B: {"type":"opening", "text":…, "tokens":29}
    loop every new token
        A->>M: forward pass on the context
        M-->>A: next token
        A-->>B: {"type":"token", "text":" They", "count":1, "context":29, "tok_s":…}
    end
    A-->>B: {"type":"story_done", "tokens":115, "tok_s":314}
    A->>Q: check this story (answer as JSON)
    Q-->>A: {"checks":[{"label":"Happy ending","pass":true},…]}
    A-->>B: {"type":"check", …} then {"type":"done", …}
```

Two details worth stealing:

- **Decoding token by token is tricky.** One emoji can be 3 tokens, and decoding just the first gives a broken `�` character. So we decode the *whole* continuation each time and only send the new part once it doesn't end in `�`.
- **One GPU, one story at a time.** A lock makes a second request get a polite "busy" message instead of fighting over VRAM.

### Search with SQLite FTS5

[build_index.py](../studio/build_index.py) loads every story into SQLite with its word and token counts, then builds an **FTS5 full-text index** with Porter stemming, so "unicorn" also finds "unicorns". Searching 2.7M stories then takes under 0.1 s, with no search server needed.

## Why is our model "only" 120–310 tokens/sec?

It's much smaller than Qwen, so it *should* be faster still. The reason it isn't: **no KV cache.** For every new token, `generate()` re-runs the model over the *entire* story so far. Big inference engines like Ollama cache each token's attention keys and values, so each new token costs one position, not hundreds. Adding a KV cache is a great next exercise (see [chapter 9](09-whats-next.md)).

---
← [Chapter 7 · Results](07-results.md) · Next: [Chapter 9 · What's next →](09-whats-next.md)
