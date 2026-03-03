from prompt import *
import os
import json
import statistics
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Using OpenAI SDK for the client
from openai import OpenAI

# load .env attributes
def load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        print(f"[warning] {path} not found, using defaults.")
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

load_env()

HOST         = os.getenv("HOST", "http://localhost") + ":" + os.getenv("PORT", "8000")
MODEL        = os.getenv("MODEL", "Qwen/Qwen3-0.6B")
NUM_REQUESTS = int(os.getenv("NUM_REQUESTS",   "100"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS",     "128"))
TEMPERATURE  = float(os.getenv("TEMPERATURE",  "0.0"))
LOG_FILE     = os.getenv("LOG_FILE",           "latency_log_baseline.json")
# Make CONCURRENCY configurable via .env; default kept as 1 to preserve previous behavior.
CONCURRENCY  = int(os.getenv("CONCURRENCY", "1"))

# api_key is required by the SDK but vLLM does not validate it.
client = OpenAI(
    base_url=f"{HOST}/v1",
    api_key="vllm-key",
)

def server_timeout(timeout:int = 180):
    import urllib.request
    print(f"[client] waiting for vLLM server at {HOST}", end="",flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{HOST}/health",timeout=3):
                print("ready \n")
                return
        except Exception:
            print(".",end="",flush=True)
            time.sleep(3)
    raise RuntimeError(f"\n[client] Server not ready after {timeout}s. "
        "Run bash serve.sh in another terminal first.")

def send_request(questions:str):
    # Send a request at a time (blocking call)
    t0 = time.perf_counter() # start
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":questions}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    latency_stop = time.perf_counter() - t0
    answer        = response.choices[0].message.content
    # note: response.usage may be None for some servers; guard just in case
    prompt_tokens = getattr(response.usage, "prompt_tokens", 0) if getattr(response, "usage", None) else 0
    output_tokens = getattr(response.usage, "completion_tokens", 0) if getattr(response, "usage", None) else 0

    return answer, latency_stop, prompt_tokens, output_tokens

def run_benchmark() -> None:
    print("=" * 65)
    print("Benchmark of Steps 1 and 2")
    print("=" * 65)
    print(f"  Server           : {HOST}")
    print(f"  Model            : {MODEL}")
    print(f"  Client           : openai SDK  (vLLM-compatible)")
    print(f"  Kernel           : vLLM standard CUDA  (baseline)")
    print(f"  dtype            : float16")
    print(f"  Requests         : {NUM_REQUESTS}")
    mode = "sequential" if CONCURRENCY <= 1 else "parallel"
    print(f"  Concurrency      : {CONCURRENCY}  ({mode})")
    print(f"  Max tokens/reply : {MAX_TOKENS}")
    print(f"  Temperature      : {TEMPERATURE}  (greedy)")
    print(f"  Log file         : {LOG_FILE}")
    print("=" * 65 + "\n")

    server_timeout()

    print(f"  {'Req':>3}  {'Latency (s)':>11}  {'In tok':>6}  {'Out tok':>7}  Question")
    print("  " + "-" * 63)

    log       = []
    latencies = []

    # Parallel runner (used when CONCURRENCY > 1)
    def run_parallel():
        # limit max_workers to NUM_REQUESTS at most to avoid creating useless threads
        max_workers = min(CONCURRENCY, NUM_REQUESTS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(send_request, PROMPTS[i]): i
                for i in range(NUM_REQUESTS)
            }

            # as_completed yields futures as they complete
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                question = PROMPTS[i]
                try:
                    answer, latency_s, p_tok, o_tok = future.result()
                    latencies.append(latency_s)
                    log.append({
                        "index":         i,
                        "question":      question,
                        "answer":        answer,
                        "latency_s":     round(latency_s, 4),
                        "prompt_tokens": p_tok,
                        "output_tokens": o_tok,
                        "error":         None,
                    })
                    print(f"  {i+1:>3}  {latency_s:>11.3f}s  {p_tok:>6}  {o_tok:>7}  {question[:35]}")
                except Exception as exc:
                    print(f"  {i+1:>3}  ERROR: {exc}")
                    log.append({
                        "index": i, "question": question,
                        "answer": None, "latency_s": None,
                        "prompt_tokens": 0, "output_tokens": 0,
                        "error": str(exc),
                    })

    # Sequential runner
    def run_sequential():
        for i in range(NUM_REQUESTS):
            question = PROMPTS[i]
            try:
                answer, latency_s, p_tok, o_tok = send_request(question)
                latencies.append(latency_s)
                log.append({
                    "index":         i,
                    "question":      question,
                    "answer":        answer,
                    "latency_s":     round(latency_s, 4),
                    "prompt_tokens": p_tok,
                    "output_tokens": o_tok,
                    "error":         None,
                })
                print(f"  {i+1:>3}  {latency_s:>11.3f}s  {p_tok:>6}  {o_tok:>7}  {question[:35]}")

            except Exception as exc:
                print(f"  {i+1:>3}  ERROR: {exc}")
                log.append({
                    "index": i, "question": question,
                    "answer": None, "latency_s": None,
                    "prompt_tokens": 0, "output_tokens": 0,
                    "error": str(exc),
                })

    # Choose runner based on CONCURRENCY
    if CONCURRENCY > 1:
        run_parallel()
    else:
        run_sequential()

    # Summary
    valid      = [l for l in latencies]
    failed     = NUM_REQUESTS - len(valid)
    avg        = statistics.mean(valid)       if valid else 0
    median     = statistics.median(valid)     if valid else 0
    std        = statistics.stdev(valid)      if len(valid) > 1 else 0
    minimum    = min(valid)                   if valid else 0
    maximum    = max(valid)                   if valid else 0
    throughput = len(valid) / sum(valid)      if valid else 0

    print("\n" + "=" * 65)
    print("LATENCY SUMMARY  vLLM standard CUDA baseline  float16")
    print("=" * 65)
    print(f"  Requests sent      : {NUM_REQUESTS}")
    print(f"  Successful         : {len(valid)}")
    print(f"  Failed             : {failed}")
    print(f"  Average latency    : {avg:.3f} s   <- primary metric")
    print(f"  Median latency     : {median:.3f} s")
    print(f"  Std deviation      : {std:.3f} s")
    print(f"  Min latency        : {minimum:.3f} s")
    print(f"  Max latency        : {maximum:.3f} s")
    print(f"  Throughput         : {throughput:.2f} req/s")
    print("=" * 65)

    # Save log
    output = {
        "meta": {
            "step":        "Step 2  vLLM standard CUDA baseline",
            "server":      HOST,
            "model":       MODEL,
            "client":      "openai SDK",
            "dtype":       "float16",
            "kernel":      "vLLM standard CUDA PagedAttention",
            "concurrency": CONCURRENCY,
            "max_tokens":  MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
        "summary": {
            "num_requests":     NUM_REQUESTS,
            "successful":       len(valid),
            "failed":           failed,
            "avg_latency_s":    round(avg,        4),
            "median_latency_s": round(median,     4),
            "std_latency_s":    round(std,        4),
            "min_latency_s":    round(minimum,    4),
            "max_latency_s":    round(maximum,    4),
            "throughput_rps":   round(throughput, 4),
        },
        "requests": log,
    }

    with open(LOG_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[client] Log saved to {LOG_FILE}")
    print("[client] avg_latency_s in this file is your Step 2 baseline number.")
    print("[client] For Step 3: point HOST in .env at the megakernel server and rerun.\n")


if __name__ == "__main__":
    run_benchmark()