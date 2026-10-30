"""Build the searchable story library used by Story Studio's Library tab.

Reads the TinyStories text files and writes studio/library.db: an SQLite
table of every story (with word and token counts) plus a full-text search
index. Run once after downloading the data (takes a few minutes):

  uv run python -m studio.build_index
"""
import os
import sqlite3
import time
from multiprocessing import Pool

import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
DB = os.path.join(HERE, "library.db")
SOURCES = [("train", "TinyStoriesV2-GPT4-train.txt"), ("valid", "TinyStoriesV2-GPT4-valid.txt")]

enc = tiktoken.get_encoding("gpt2")


def count_batch(stories):
    tokens = enc.encode_ordinary_batch(stories)
    return [(s, len(s.split()), len(t) + 1) for s, t in zip(stories, tokens)]  # +1 for <|endoftext|>


def main():
    if os.path.exists(DB):
        os.remove(DB)
    db = sqlite3.connect(DB)
    db.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE stories (
            id INTEGER PRIMARY KEY,
            split TEXT NOT NULL,
            text TEXT NOT NULL,
            n_words INTEGER NOT NULL,
            n_tokens INTEGER NOT NULL
        );
        CREATE VIRTUAL TABLE stories_fts USING fts5(
            text, content='stories', content_rowid='id', tokenize='porter unicode61'
        );
    """)
    start = time.time()
    for split, name in SOURCES:
        with open(os.path.join(DATA, name), encoding="utf-8") as f:
            stories = [s.strip() for s in f.read().split("<|endoftext|>") if s.strip()]
        batches = [stories[i:i + 5000] for i in range(0, len(stories), 5000)]
        with Pool(max(1, os.cpu_count() - 2)) as pool:
            for rows in pool.imap(count_batch, batches):
                db.executemany(
                    "INSERT INTO stories (split, text, n_words, n_tokens) VALUES (?, ?, ?, ?)",
                    [(split, *r) for r in rows])
        db.commit()
        print(f"{split}: {len(stories):,} stories loaded ({time.time() - start:.0f}s)")

    print("building search index...")
    db.execute("INSERT INTO stories_fts (stories_fts) VALUES ('rebuild')")
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"done in {time.time() - start:.0f}s: {DB} ({os.path.getsize(DB) / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
