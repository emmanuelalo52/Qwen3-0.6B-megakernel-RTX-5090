#!/bin/bash
# =============================================================================
# serve.sh
# Step 2 as requested: Run vLLM with standard CUDA kernel as the baseline.
#
# Starts an OpenAI-compatible HTTP server on localhost:8000.
# The model runs with vLLM's default PagedAttention CUDA kernel (no megakernel).
# Wait for "Application startup complete" before running benchmark_client.py.
#
# Flags explained:
#   --dtype float16                  FP16 precision (matches megakernel dtype)
#   --max-model-len 2048             Max input+output token length per request
#   --max-num-seqs 1                 One request at a time — matches concurrency=1 benchmark
#   --max-num-batched-tokens 2048    Max tokens processed per scheduler step
#   --block-size 16                  KV cache block size in tokens (PagedAttention)
#   --gpu-memory-utilization 0.90    90% of VRAM reserved for KV cache blocks
#   --swap-space 4                   4 GB CPU RAM for preempted sequence KV swap (swapping)
#   --kv-cache-dtype float16         KV cache stored in fp16 (matches model dtype)
#   --scheduling-policy fcfs         First-come-first-served (standard, no priority tricks)
#   --enforce-eager                  Disables CUDA graph capture — pure standard CUDA kernel (no use of graph here)
#   --disable-log-requests           Keeps terminal clean during the 100-request benchmark (Free after 100 prompts)
# =============================================================================

set -eo pipefail

# Load .env if present
if [ -f .env ]; then
    set -o allexport; source .env; set +o allexport
    echo "[serve.sh] Loaded .env"
else
    echo "[serve.sh] .env not found — using defaults"
    PORT=8000
    MODEL="Qwen/Qwen3-0.6B"
    DTYPE="float16"
    GPU_MEMORY_UTILIZATION=0.90
    MAX_MODEL_LEN=2048
    MAX_NUM_SEQS=1
    BLOCK_SIZE=16
    SWAP_SPACE=4
    KV_CACHE_DTYPE="float16"
fi

echo "[serve.sh] ========================================"
echo "[serve.sh] Model                  : $MODEL"
echo "[serve.sh] Port                   : $PORT"
echo "[serve.sh] dtype                  : $DTYPE"
echo "[serve.sh] KV cache dtype         : $KV_CACHE_DTYPE"
echo "[serve.sh] GPU memory utilization : $GPU_MEMORY_UTILIZATION"
echo "[serve.sh] Max model length       : $MAX_MODEL_LEN"
echo "[serve.sh] Max num seqs           : $MAX_NUM_SEQS"
echo "[serve.sh] Block size             : $BLOCK_SIZE tokens"
echo "[serve.sh] Swap space             : $SWAP_SPACE GB"
echo "[serve.sh] Scheduling policy      : fcfs"
echo "[serve.sh] Kernel                 : vLLM standard CUDA (enforce-eager, no CUDA graphs)"
echo "[serve.sh] ========================================"
echo ""


# Launch the server
python -m vllm.entrypoints.openai.api_server \
    --model                   "$MODEL"                   \
    --dtype                   "$DTYPE"                   \
    --port                    "$PORT"                    \
    --host                    0.0.0.0                    \
    --gpu-memory-utilization  "$GPU_MEMORY_UTILIZATION"  \
    --max-model-len           "$MAX_MODEL_LEN"           \
    --max-num-seqs            "$MAX_NUM_SEQS"            \
    --max-num-batched-tokens  "$MAX_MODEL_LEN"           \
    --block-size              "$BLOCK_SIZE"              \
    --swap-space              "$SWAP_SPACE"              \
    --kv-cache-dtype          "$KV_CACHE_DTYPE"          \
    --scheduling-policy       fcfs                       \
    --enforce-eager                                      \
    --disable-log-requests