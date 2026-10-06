#!/bin/bash
# =============================================================================
# bench_serving.sh
# Serving benchmark, megakernel vs vLLM, like the RTX 5090 table in README.md:
# serve each configuration in turn on PORT, send the 100 prompts from prompt.py
# with client_benchmark.py, stop it, then compare all logs with compare_benchmarks.py.
#
#   bash bench_serving.sh
#   DPS_CONFIGS="fp16:auto fp16:static" VLLM_MODES=eager MAX_TOKENS=128 bash bench_serving.sh
#
# Environment (also read from .env; explicit variables win):
#   PY=python          python with the megakernel extension built, fastapi, uvicorn, openai
#   VLLM_BIN=vllm      vLLM CLI; vLLM pins its own torch, so give it its own venv:
#                      VLLM_BIN=/venv/vllm/bin/vllm
#   DPS_CONFIGS        megakernel runs as "weights:scheduler" (weights fp16|fp8|fp4,
#                      scheduler auto|atomic|static|clc). Default: "fp16:auto fp8:auto fp4:auto"
#   VLLM_MODES         "eager" (no CUDA graphs, as in the RTX 5090 comparison) and/or
#                      "graphs" (vLLM's default). Default: "eager graphs"
#   MAX_TOKENS=32      max new tokens per request (the RTX 5090 table used 32)
#   NUM_REQUESTS=100   WARMUP=3 (untimed requests per server)   PORT=8000
#   OUT_DIR            default megakernel_dynamic/results/serving_<date>
# =============================================================================

set -eo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
    # .env fills in only what the caller did not set
    while IFS='=' read -r key val; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        [ -z "${!key+x}" ] && export "$key=$val"
    done < .env
fi
: "${PY:=python}"
: "${VLLM_BIN:=vllm}"
: "${DPS_CONFIGS=fp16:auto fp8:auto fp4:auto}"   # set empty to skip
: "${VLLM_MODES=eager graphs}"   # set empty to skip
: "${HOST:=http://localhost}"
: "${PORT:=8000}"
: "${MODEL:=Qwen/Qwen3-0.6B}"
: "${MAX_TOKENS:=32}"
: "${NUM_REQUESTS:=100}"
: "${WARMUP:=3}"
: "${TEMPERATURE:=0.0}"
: "${CONCURRENCY:=1}"
: "${OUT_DIR:=megakernel_dynamic/results/serving_$(date +%Y%m%d_%H%M)}"
export HOST PORT MODEL MAX_TOKENS NUM_REQUESTS WARMUP TEMPERATURE CONCURRENCY
mkdir -p "$OUT_DIR"

GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
{
    echo "date: $(date -Is)"; echo "gpu: $GPU"; nvidia-smi --query-gpu=driver_version,memory.total --format=csv
    echo "megakernel python: $("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    echo "vllm: $("$VLLM_BIN" --version 2>/dev/null || echo 'not found')"
    echo "MAX_TOKENS=$MAX_TOKENS NUM_REQUESTS=$NUM_REQUESTS WARMUP=$WARMUP CONCURRENCY=$CONCURRENCY"
} | tee "$OUT_DIR/env.txt"

SERVER_PID=""
stop_server() {
    [ -z "$SERVER_PID" ] && return 0
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    for _ in $(seq 60); do   # until the port is free
        curl -sf -o /dev/null "$HOST:$PORT/health" || break
        sleep 1
    done
    sleep 3   # let the GPU memory go before the next server starts
}
trap stop_server EXIT

wait_ready() {   # name, log, timeout seconds
    for _ in $(seq "$3"); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[bench] $1 exited during startup; last lines of $2:"; tail -20 "$2"; return 1
        fi
        curl -sf -o /dev/null "$HOST:$PORT/health" && return 0
        sleep 1
    done
    echo "[bench] $1 not ready after $3 s; see $2"; return 1
}

run_client() {   # label, log name
    SERVER_LABEL="$1" LOG_FILE="$OUT_DIR/$2.json" "$PY" client_benchmark.py | tee "$OUT_DIR/$2_client.txt"
}

for cfg in $DPS_CONFIGS; do
    w="${cfg%%:*}"; s="${cfg##*:}"
    name="dps_${w}_${s}"; log="$OUT_DIR/${name}_server.log"
    echo; echo "[bench] ===== megakernel: weights $w, scheduler $s ====="
    DPS_WEIGHTS="$w" DPS_SCHED="$s" DPS_WARMUP=3 "$PY" megakernel_dynamic/serve_dps.py > "$log" 2>&1 &
    SERVER_PID=$!
    wait_ready "megakernel server" "$log" 600
    DTYPE="$w" run_client "Megakernel ($w, $s)" "$name"
    stop_server
done

for mode in $VLLM_MODES; do
    eager=1; [ "$mode" = "graphs" ] && eager=0
    name="vllm_${mode}"; log="$OUT_DIR/${name}_server.log"
    echo; echo "[bench] ===== vLLM ($mode) ====="
    ENFORCE_EAGER=$eager VLLM_BIN="$VLLM_BIN" bash serve.sh > "$log" 2>&1 &
    SERVER_PID=$!
    wait_ready "vLLM" "$log" 900
    DTYPE=float16 run_client "vLLM ($mode)" "$name"
    stop_server
done

# Compare every run against the first vLLM mode.
base_mode="${VLLM_MODES%% *}"
baseline="$OUT_DIR/vllm_${base_mode}.json"
others=()
for f in "$OUT_DIR"/dps_*.json "$OUT_DIR"/vllm_*.json; do
    [ -f "$f" ] && [ "$f" != "$baseline" ] && others+=("$f")
done
if [ -n "$base_mode" ] && [ -f "$baseline" ] && [ ${#others[@]} -gt 0 ]; then
    echo; echo "[bench] ===== comparison ($GPU, max_tokens $MAX_TOKENS) ====="
    "$PY" compare_benchmarks.py --baseline "$baseline" "${others[@]}" | tee "$OUT_DIR/comparison.md"
else
    echo; echo "[bench] no vLLM baseline in this run: skipping the comparison"
fi
echo; echo "[bench] logs in $OUT_DIR"
