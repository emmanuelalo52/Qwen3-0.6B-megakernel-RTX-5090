"""OpenAI-compatible server for the dynamic-persistent megakernel, for the vLLM comparison.

    python megakernel_dynamic/serve_dps.py                                 # fp16, default scheduler
    DPS_WEIGHTS=fp8 DPS_SCHED=static python megakernel_dynamic/serve_dps.py

Same API, prompt and request handling as Tools/megakernel/megakernel.py (the RTX 5090
server), so client_benchmark.py measures both the same way:
  * /v1/chat/completions, /v1/models, /health on PORT (default 8000);
  * the chat template with thinking disabled, which gives the same prompt tokens as vLLM
    with chat_template_kwargs={"enable_thinking": False};
  * greedy decoding, one request at a time (the kernel runs batch 1), whole request in one launch.

Token accounting follows vLLM: generation stops at <|im_end|>, which counts as a completion
token, and finish_reason is "stop" (EOS) or "length" (max_tokens).

Environment (also read from .env): MODEL, PORT, MAX_TOKENS, DPS_WEIGHTS (fp16 | fp8 | fp4),
DPS_SCHED (auto | atomic | static | clc), DPS_BACKEND (cuda | cutedsl), DPS_WARMUP (requests
run at startup, default 3, so the first timed request is not a cold one).
"""

import asyncio
import itertools
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


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

PORT = int(os.getenv("PORT", "8000"))
MODEL_NAME = os.getenv("MODEL", "Qwen/Qwen3-0.6B")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
WEIGHTS = os.getenv("DPS_WEIGHTS", "fp16")
SCHED = os.getenv("DPS_SCHED", "auto")
BACKEND = os.getenv("DPS_BACKEND", "cuda")
WARMUP = int(os.getenv("DPS_WARMUP", "3"))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

try:   # fastapi exports ORJSONResponse even without orjson, then fails on the first response
    import orjson  # noqa: F401
    from fastapi.responses import ORJSONResponse as JSONResponse
except ImportError:
    from fastapi.responses import JSONResponse

decoder = None
_executor = ThreadPoolExecutor(max_workers=1)   # the kernel runs one request at a time
_req_counter = itertools.count(1)


def build_prompt(tokenizer, messages) -> str:
    """Chat template with thinking disabled (same tokens as vLLM's enable_thinking=False)."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = re.sub(r"<think>.*?</think>\n*", "", prompt, flags=re.DOTALL)
    return prompt.rstrip() + "\n<think>\n\n</think>\n\n"


def run_generate(messages, max_tokens: int):
    """-> (text, prompt tokens, completion tokens incl. EOS, finish_reason)."""
    tok = decoder.tokenizer
    ids = tok.encode(build_prompt(tok, messages), add_special_tokens=False)
    out = decoder.generate_ids(ids, max_tokens, tok.eos_token_id)   # stops at EOS, which it leaves out
    hit_eos = len(out) < max_tokens
    return (tok.decode(out, skip_special_tokens=True), len(ids), len(out) + int(hit_eos),
            "stop" if hit_eos else "length")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global decoder
    from qwen_dps import DpsDecoder, load_weights

    print(f"[serve_dps] loading {MODEL_NAME}: weights={WEIGHTS} sched={SCHED} backend={BACKEND}", flush=True)
    t0 = time.time()
    weights, tokenizer, hf_model = load_weights(MODEL_NAME)
    del hf_model
    decoder = DpsDecoder(weights, tokenizer, backend=BACKEND, sched=SCHED, weight_format=WEIGHTS)
    for _ in range(WARMUP):
        run_generate([{"role": "user", "content": "Hello!"}], 32)
    mode, grid = decoder.last_launch if decoder.last_launch else ("?", "?")
    print(f"[serve_dps] ready in {time.time() - t0:.1f} s (scheduler mode {mode}, grid {grid}); "
          f"serving on port {PORT}", flush=True)
    yield


app = FastAPI(title="Qwen dynamic-persistent megakernel server", default_response_class=JSONResponse,
              lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.0   # decoding is always greedy
    stream: Optional[bool] = False


@app.get("/health")
async def health():
    return {"status": "ok" if decoder is not None else "loading"}


@app.get("/v1/models")
async def models():
    return {"object": "list",
            "data": [{"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "megakernel"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if decoder is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if req.stream:
        raise HTTPException(status_code=400, detail="streaming is not supported")
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        loop = asyncio.get_running_loop()
        text, n_prompt, n_out, finish = await loop.run_in_executor(
            _executor, run_generate, messages, req.max_tokens or MAX_TOKENS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "id": f"chatcmpl-{next(_req_counter)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning", access_log=False)
