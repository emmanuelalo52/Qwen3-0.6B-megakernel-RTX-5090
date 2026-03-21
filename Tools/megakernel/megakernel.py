import asyncio
import os
import time
import itertools
from contextlib import asynccontextmanager
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

decoder   = None
_executor = ThreadPoolExecutor(max_workers=1)

_req_counter = itertools.count(1)

_cached_ts    = int(time.time())
_cached_ts_at = time.monotonic()


def _get_ts() -> int:
    global _cached_ts, _cached_ts_at
    now = time.monotonic()
    if now - _cached_ts_at >= 1.0:
        _cached_ts    = int(time.time())
        _cached_ts_at = now
    return _cached_ts


# I replaced deprecated @app.on_event("startup") with the lifespan context manager — required by FastAPI >= 0.93.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global decoder
    print(f"[megakernel_server] Loading {MODEL_NAME} with megakernel...", flush=True)
    t0 = time.time()
    from Model.Qwen06B_architecture import Decoder
    decoder = Decoder(model_name=MODEL_NAME, verbose=True)
    print(f"[megakernel_server] Model loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"[megakernel_server] Serving on port {PORT}", flush=True)
    yield
    # Shutdown: nothing to clean up (CUDA context released with process)


app = FastAPI(
    title="Qwen Megakernel Server",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


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
    import re as _re
    msgs   = [{"role": m.role, "content": m.content} for m in messages]
    prompt = decoder.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    # Suppress Qwen3 chain-of-thought — strip existing <think> block and inject an empty one so the model skips the thinking phase.
    prompt = _re.sub(r'<think>.*?</think>\n*', '', prompt, flags=_re.DOTALL)
    prompt = prompt.rstrip() + '\n<think>\n\n</think>\n\n'
    return prompt


def _run_generate(messages: list[Message], max_tok: int):
    prompt = _build_prompt(messages)
    return decoder.generate(prompt, max_tokens=max_tok)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if decoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    max_tok = req.max_tokens or MAX_TOKENS

    try:
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
        access_log=False,
    )