"""Story Studio: browse the training stories and make new ones.

Run from the tinygpt folder:  uv run python -m studio.app
Then open http://127.0.0.1:8000
"""
import json
import os
import threading
import time

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")  # use the RX 7900 XT, not the iGPU

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .library import Library
from .storyteller import QWEN, TinyGPT, tell

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(HERE, "..", "saved", "stories.jsonl")

app = FastAPI(title="Story Studio")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
library = Library()
gpt = TinyGPT()
gpu_lock = threading.Lock()  # one story at a time on the GPU


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/info")
def info():
    return {"model": gpt.info, "qwen": QWEN, "library": library.by_split}


@app.get("/api/search")
def search(q: str, offset: int = 0):
    return library.search(q, offset=offset)


@app.get("/api/random")
def random_story():
    return library.random()


@app.get("/api/story/{story_id}")
def story(story_id: int):
    s = library.get(story_id)
    if not s:
        raise HTTPException(404, "no such story")
    return s


@app.get("/api/tokens/{story_id}")
def tokens(story_id: int):
    t = library.tokens(story_id)
    if t is None:
        raise HTTPException(404, "no such story")
    return t


@app.get("/api/tell")
def tell_story(request: str):
    def events():
        if not gpu_lock.acquire(blocking=False):
            yield f"data: {json.dumps({'type': 'error', 'message': 'Busy with another story, try again in a moment.'})}\n\n"
            return
        try:
            for event in tell(gpt, request):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:  # surface failures (e.g. Ollama not running) in the UI
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            gpu_lock.release()

    return StreamingResponse(events(), media_type="text/event-stream")


class SavedStory(BaseModel):
    request: str
    opening: str
    story: str
    checks: list
    passed: bool


@app.post("/api/save")
def save(item: SavedStory):
    os.makedirs(os.path.dirname(SAVED), exist_ok=True)
    with open(SAVED, "a", encoding="utf-8") as f:
        f.write(json.dumps({**item.model_dump(), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    return {"saved_to": os.path.relpath(SAVED, os.path.join(HERE, ".."))}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
