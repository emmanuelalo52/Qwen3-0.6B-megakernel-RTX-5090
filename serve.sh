#!/bin/bash
# =============================================================================
# serve.sh
# Step 2 as requested: Run vLLM with standard CUDA kernel as the baseline.
#
# Starts an OpenAI-compatible HTTP server on localhost:8000.
# The model runs with vLLM's default PagedAttention CUDA kernel (no megakernel).
# Wait for "Application startup complete" before running client_benchmark.py.
#
# Flags explained:
#   --dtype float16                  FP16 precision (matches megakernel dtype)
#   --max-model-len 2048             Max input+output token length per request
#   --max-num-seqs 1                 One request at a time — matches concurrency=1 benchmark
#   --max-num-batched-tokens 2048    Max tokens processed per scheduler step
#   --block-size 16                  KV cache block size in tokens (PagedAttention)
#   --gpu-memory-utilization 0.90    90% of VRAM reserved for KV cache blocks
#   --swap-space 4                   4 GB CPU RAM for preempted sequence KV swap (older vLLM only)
#   --kv-cache-dtype float16         KV cache stored in fp16 (matches model dtype)
#   --scheduling-policy fcfs         First-come-first-served (standard, no priority tricks)
#   --enforce-eager                  Disables CUDA graph capture — pure standard CUDA kernel (ENFORCE_EAGER=1, default)
#   --disable-log-requests           Keeps terminal clean during the 100-request benchmark (if this vLLM has it)
#
# Environment:
#   VLLM_BIN=vllm        vLLM's CLI. vLLM pins its own torch, so on a box where the megakernel
#                        needs a different one, install vLLM in its own venv and point here:
#                        VLLM_BIN=/venv/vllm/bin/vllm bash serve.sh
#   ENFORCE_EAGER=1      1: no CUDA graphs (the RTX 5090 comparison). 0: vLLM's default CUDA
#                        graphs, a stronger baseline.
# Flags that newer vLLM releases dropped (--swap-space) are passed only if the installed
# vLLM still lists them in its help.
# =============================================================================

set -eo pipefail

# Load .env if present
if [ -f .env ]; then
    set -o allexport; source .env; set +o allexport
    echo "[serve.sh] Loaded .env"
else
    echo "[serve.sh] .env not found — using defaults"
fi
: "${PORT:=8000}"
: "${MODEL:=Qwen/Qwen3-0.6B}"
: "${DTYPE:=float16}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_MODEL_LEN:=2048}"
: "${MAX_NUM_SEQS:=1}"
: "${BLOCK_SIZE:=16}"
: "${SWAP_SPACE:=4}"
: "${KV_CACHE_DTYPE:=float16}"
: "${VLLM_BIN:=vllm}"
: "${ENFORCE_EAGER:=1}"

# Which optional flags does this vLLM accept?
HELP="$("$VLLM_BIN" serve --help=all 2>/dev/null || "$VLLM_BIN" serve --help 2>/dev/null || true)"
supports() { grep -q -- "$1" <<<"$HELP"; }

ARGS=(
    --dtype                   "$DTYPE"
    --port                    "$PORT"
    --host                    0.0.0.0
    --gpu-memory-utilization  "$GPU_MEMORY_UTILIZATION"
    --max-model-len           "$MAX_MODEL_LEN"
    --max-num-seqs            "$MAX_NUM_SEQS"
    --max-num-batched-tokens  "$MAX_MODEL_LEN"
    --block-size              "$BLOCK_SIZE"
    --kv-cache-dtype          "$KV_CACHE_DTYPE"
    --scheduling-policy       fcfs
)
if supports --swap-space; then ARGS+=(--swap-space "$SWAP_SPACE"); fi
if supports --disable-log-requests; then ARGS+=(--disable-log-requests); fi
if [ "$ENFORCE_EAGER" = "1" ]; then
    ARGS+=(--enforce-eager)
    KERNEL_DESC="vLLM standard CUDA (enforce-eager, no CUDA graphs)"
else
    KERNEL_DESC="vLLM with CUDA graphs"
fi

echo "[serve.sh] ========================================"
echo "[serve.sh] vLLM                   : $VLLM_BIN ($("$VLLM_BIN" --version 2>/dev/null || echo '?'))"
echo "[serve.sh] Model                  : $MODEL"
echo "[serve.sh] Port                   : $PORT"
echo "[serve.sh] dtype                  : $DTYPE"
echo "[serve.sh] KV cache dtype         : $KV_CACHE_DTYPE"
echo "[serve.sh] GPU memory utilization : $GPU_MEMORY_UTILIZATION"
echo "[serve.sh] Max model length       : $MAX_MODEL_LEN"
echo "[serve.sh] Max num seqs           : $MAX_NUM_SEQS"
echo "[serve.sh] Block size             : $BLOCK_SIZE tokens"
echo "[serve.sh] Scheduling policy      : fcfs"
echo "[serve.sh] Kernel                 : $KERNEL_DESC"
echo "[serve.sh] Flags                  : ${ARGS[*]}"
echo "[serve.sh] ========================================"
echo ""

# Launch the server
exec "$VLLM_BIN" serve "$MODEL" "${ARGS[@]}"
