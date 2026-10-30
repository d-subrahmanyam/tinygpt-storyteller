"""Read-only access to the story library built by build_index.py."""
import html
import os
import re
import sqlite3

import tiktoken

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library.db")
enc = tiktoken.get_encoding("gpt2")

# snippet() markers that can't appear in story text; swapped for <mark> after escaping
START, END = "\x02", "\x03"


def _highlight(snippet):
    return html.escape(snippet).replace(START, "<mark>").replace(END, "</mark>")


class Library:
    def __init__(self, path=DB):
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found; run: uv run python -m studio.build_index")
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.total = self.db.execute("SELECT max(id) FROM stories").fetchone()[0]
        self.by_split = dict(self.db.execute("SELECT split, count(*) FROM stories GROUP BY split").fetchall())

    @staticmethod
    def _fts_query(q):
        # Quote each word so punctuation in user input can't break FTS syntax; words are ANDed.
        words = re.findall(r"\w+", q.lower())
        return " ".join(f'"{w}"' for w in words)

    def search(self, q, limit=20, offset=0):
        match = self._fts_query(q)
        if not match:
            return {"total": 0, "results": []}
        total = self.db.execute(
            "SELECT count(*) FROM stories_fts WHERE stories_fts MATCH ?", (match,)).fetchone()[0]
        rows = self.db.execute(
            f"""SELECT s.id, s.split, s.n_words, s.n_tokens,
                       snippet(stories_fts, 0, '{START}', '{END}', '…', 40) AS snip
                FROM stories_fts JOIN stories s ON s.id = stories_fts.rowid
                WHERE stories_fts MATCH ? LIMIT ? OFFSET ?""",
            (match, limit, offset)).fetchall()
        return {"total": total, "results": [
            {"id": r["id"], "split": r["split"], "n_words": r["n_words"],
             "n_tokens": r["n_tokens"], "snippet": _highlight(r["snip"])} for r in rows]}

    def get(self, story_id):
        r = self.db.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
        return dict(r) if r else None

    def random(self):
        r = self.db.execute(
            "SELECT * FROM stories WHERE id = abs(random()) % ? + 1", (self.total,)).fetchone()
        return dict(r)

    def tokens(self, story_id):
        story = self.get(story_id)
        if not story:
            return None
        ids = enc.encode_ordinary(story["text"]) + [enc.eot_token]
        return [{"id": t, "text": enc.decode([t])} for t in ids]
