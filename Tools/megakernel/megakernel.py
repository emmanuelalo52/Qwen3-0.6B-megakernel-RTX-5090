"""
megakernel_server.py — OpenAI-compatible HTTP server backed by the Qwen megakernel.

Optimisations vs original:
  OPT 1  — Single tokenizer.encode() per request (original called it 3x)
  OPT 2  — _gen op lookup cached on Decoder.__init__
  OPT 3  — Partial KV-cache reset: zero only positions written, not full 235MB
  OPT 4  — prompt_tokens / output_tokens returned from generate(), no re-encode
  OPT 5  — run_in_executor: GPU work offloaded, event loop stays free
  OPT 6  — tokenizer.encode() inside executor thread (overlaps with async I/O)
  OPT 7  — uuid4() replaced with incrementing counter (~0.1ms saved)
  OPT 8  — time.time() cached per-second (~0.05ms saved)
  OPT 9  — ORJSONResponse: 2-4x faster JSON serialization than stdlib
  OPT 10 — access_log=False: no file write per request
  OPT 11 — asyncio.get_running_loop() replaces deprecated get_event_loop()
  OPT 12 — _build_prompt() moved inside executor thread, off the event loop
"""

import asyncio
import os
import time
import itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

load_env()

PORT       = int(os.getenv("PORT", "8000"))
MODEL_NAME = os.getenv("MODEL", "Qwen/Qwen3-0.6B")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))

from fastapi import FastAPI, HTTPException
try:
    from fastapi.responses import ORJSONResponse
except ImportError:
    from fastapi.responses import JSONResponse as ORJSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Qwen Megakernel Server", default_response_class=ORJSONResponse)

decoder   = None
_executor = ThreadPoolExecutor(max_workers=1)

# OPT 7: counter instead of uuid4()
_req_counter = itertools.count(1)

# OPT 8: cached timestamp
_cached_ts    = int(time.time())
_cached_ts_at = time.monotonic()

def _get_ts() -> int:
    global _cached_ts, _cached_ts_at
    now = time.monotonic()
    if now - _cached_ts_at >= 1.0:
        _cached_ts    = int(time.time())
        _cached_ts_at = now
    return _cached_ts


@app.on_event("startup")
async def load_model():
    global decoder
    print(f"[megakernel_server] Loading {MODEL_NAME} with megakernel...", flush=True)
    t0 = time.time()
    from Model.Qwen06B_architecture import Decoder
    decoder = Decoder(model_name=MODEL_NAME, verbose=True)
    print(f"[megakernel_server] Model loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"[megakernel_server] Serving on port {PORT}", flush=True)


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model",
                  "created": _get_ts(), "owned_by": "megakernel"}]
    }


def _build_prompt(messages: list[Message]) -> str:
    # OPT 12: called inside executor thread, not on the event loop
    # Use Qwen3 chat template so <|im_end|> (token 151645) triggers EOS.
    # Append closed <think> block to suppress chain-of-thought reasoning.
    import re as _re
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    prompt = decoder.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    # Strip any empty think block the template injects, then re-add it closed
    # so the model skips thinking and answers directly.
    prompt = _re.sub(r'<think>.*?</think>\n*', '', prompt, flags=_re.DOTALL)
    prompt = prompt.rstrip() + '\n<think>\n\n</think>\n\n'
    return prompt


def _run_generate(messages: list[Message], max_tok: int):
    """
    All blocking work runs here — inside the thread-pool executor.
    OPT 12: prompt building moved here so the event loop is free during it.
    OPT 6:  tokenizer.encode() happens inside generate(), also here.
    """
    prompt = _build_prompt(messages)
    return decoder.generate(prompt, max_tokens=max_tok)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if decoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    max_tok = req.max_tokens or MAX_TOKENS

    try:
        # OPT 11: get_running_loop() is faster than deprecated get_event_loop()
        # OPT 12: pass messages directly — prompt building happens in executor
        loop = asyncio.get_running_loop()
        answer, prompt_tokens, output_tokens = await loop.run_in_executor(
            _executor, _run_generate, req.messages, max_tok
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    req_id = next(_req_counter)

    return {
        "id":      f"chatcmpl-{req_id}",
        "object":  "chat.completion",
        "created": _get_ts(),
        "model":   MODEL_NAME,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens":      prompt_tokens + output_tokens,
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
        access_log=False,   # OPT 10: no file write per request
    )