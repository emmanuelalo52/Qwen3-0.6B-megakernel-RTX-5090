# Qwen3-0.6B Megakernel - Reproduction Guide

Custom CUDA megakernel for Qwen3-0.6B inference on RTX 5090, benchmarked against vLLM's standard PagedAttention baseline.

## Benchmark Results (RTX 5090, float16, 32 max tokens)

| Metric | Megakernel | vLLM (enforce-eager) | Speedup |
|---|---|---|---|
| Avg latency | 0.052s | 0.120s | **2.3x** |
| Median latency | 0.052s | 0.154s | **2.9x** |
| Tokens/sec | 606.7 | 266.5 | **2.3x** |
| Throughput (end-to-end) | 402.6 tok/s | 199.3 tok/s | **2.0x** |
| Req/s | 17.04 | 8.33 | **2.0x** |
| Variance (min to max) | 0.048-0.058s | 0.048-0.162s | 10x tighter |

---

## Requirements

- NVIDIA RTX 5090 (sm_120 architecture)
- CUDA 13.0 driver (nvidia-smi should show Driver 580+)
- Python 3.12
- Ubuntu 24 (tested on Vast.ai container)

**Note:** The CUDA megakernel compiles for sm_120 (Blackwell architecture) and requires a physical RTX 5090. It cannot run on older GPUs or on macOS. Mac users must use a remote GPU instance - see the macOS section below.

---

## Repository Structure

```
qwen_megakernel/
├── megakernel_5090.cu          # Main CUDA megakernel (sm_120)
├── megakernel.py               # FastAPI OpenAI-compatible server
├── qwen_ops.cpp                # PyTorch C++ extension bindings
├── setup.py                    # Build script for CUDA extension
├── serve.sh                    # vLLM baseline server launcher
├── client_benchmark.py         # Benchmark client (OpenAI SDK)
├── prompt.py                   # 100 benchmark prompts
├── rmsnorm.cuh / rmsnorm.cu    # RMSNorm CUDA kernel
├── swiglu.cuh / swiglu.cu      # SwiGLU CUDA kernel
├── swiglu_binding.cpp          # SwiGLU PyTorch binding
└── Model/
    └── Qwen06B_architecture.py # Weight loading + Decoder class
```

---

## Architecture Deep-Dive

### Model Constants

Qwen3-0.6B uses the following architecture, hardcoded as compile-time constants in both the CUDA kernel and the Python loader to ensure a single source of truth:

| Constant | Value | Description |
|---|---|---|
| `NUM_LAYERS` | 28 | Transformer blocks |
| `HIDDEN_SIZE` | 1024 | Model hidden dimension |
| `INTERMEDIATE_SIZE` | 3072 | MLP intermediate dimension (SwiGLU) |
| `NUM_Q_HEADS` | 16 | Query attention heads |
| `NUM_KV_HEADS` | 8 | Key/Value heads (GQA — 2:1 ratio) |
| `HEAD_DIM` | 128 | Dimension per attention head |
| `Q_SIZE` | 2048 | `NUM_Q_HEADS × HEAD_DIM` |
| `KV_SIZE` | 1024 | `NUM_KV_HEADS × HEAD_DIM` |
| `MAX_SEQ_LEN` | 2048 | Maximum sequence length |
| `VOCAB_SIZE` | 151936 | Vocabulary size |
| `ROPE_THETA` | 1,000,000 | RoPE base frequency |

The model uses Grouped Query Attention (GQA) with a 2:1 Q-to-KV head ratio. Each KV head is shared by 2 Q heads. This halves the KV cache memory compared to standard multi-head attention.

### Weight Layout Per Layer

Each transformer layer contains exactly 11 weight tensors, stored in a fixed order. The `LDGLayerWeight` struct in the CUDA kernel declares these as raw `half*` pointers — the Python code must pack them in exactly this order or the kernel will read garbage:

```
[0] input_layernorm.weight          [HIDDEN_SIZE]
[1] self_attn.q_proj.weight         [Q_SIZE × HIDDEN_SIZE]
[2] self_attn.k_proj.weight         [KV_SIZE × HIDDEN_SIZE]
[3] self_attn.v_proj.weight         [KV_SIZE × HIDDEN_SIZE]
[4] self_attn.q_norm.weight         [HEAD_DIM]
[5] self_attn.k_norm.weight         [HEAD_DIM]
[6] self_attn.o_proj.weight         [HIDDEN_SIZE × Q_SIZE]
[7] post_attention_layernorm.weight  [HIDDEN_SIZE]
[8] mlp.gate_proj.weight            [INTERMEDIATE_SIZE × HIDDEN_SIZE]
[9] mlp.up_proj.weight              [INTERMEDIATE_SIZE × HIDDEN_SIZE]
[10] mlp.down_proj.weight           [HIDDEN_SIZE × INTERMEDIATE_SIZE]
```

Qwen3 uses tied embeddings: `lm_head_weight` is the same tensor as `embed_tokens.weight`. This saves approximately 290MB of GPU memory at float16.

---

## CUDA Kernel Internals

### File: `megakernel_5090.cu`

The megakernel is approximately 1,400 lines of CUDA C++ targeting `sm_120` (Blackwell). It implements the full Qwen3-0.6B decode pipeline — embedding lookup, 28 transformer layers, and LM head projection — inside a fixed pool of persistent GPU blocks, eliminating per-operation kernel launch overhead.

#### Kernel Launch Configuration

```
LDG_NUM_BLOCKS   = 170   // Persistent decode blocks (configurable at compile time)
LDG_BLOCK_SIZE   = 256   // Threads per decode block (8 warps)
LDG_LM_NUM_BLOCKS = 680  // LM head Phase 1 blocks
LDG_LM_BLOCK_SIZE = 256  // Threads per LM head block
LDG_ATTN_BLOCKS  = 16    // Blocks dedicated to attention (Q-norm + softmax)
LDG_LM_ROWS_PER_WARP = 8 // Vocab rows computed per warp per iteration
```

The decode grid runs 170 blocks × 256 threads = 43,520 threads simultaneously on the RTX 5090.

#### `LDGLayerWeight` Struct and ABI Versioning

```cpp
struct __align__(16) LDGLayerWeight {
    const half* input_layernorm_weight;
    const half* q_proj_weight;
    const half* k_proj_weight;
    const half* v_proj_weight;
    const half* q_norm_weight;
    const half* k_norm_weight;
    const half* o_proj_weight;
    const half* post_attn_layernorm_weight;
    const half* gate_proj_weight;
    const half* up_proj_weight;
    const half* down_proj_weight;
    const void* padding;   // makes struct 12 × 8 = 96 bytes, 16-byte aligned
};
```

The struct is `__align__(16)` and padded to 96 bytes (12 × 8-byte pointers) so that each layer's weight table begins on a 16-byte boundary. ABI version `2` tracks this exact layout. On startup `_assert_extension_compatibility()` in `Qwen06B_architecture.py` checks that the loaded `.so` reports ABI version 2; a stale build triggers a `RuntimeError` before any CUDA call can misread memory.

#### `AtomicGridSync` — Software Grid Barrier

Standard CUDA `__syncthreads()` synchronises threads within one block. The megakernel needs all 170 blocks to coordinate between pipeline stages (e.g., the QKV projection must finish writing before attention begins reading). CUDA cooperative groups provide a hardware grid barrier, but this requires `cudaLaunchCooperativeKernel` and has occupancy restrictions.

Instead the megakernel implements its own software barrier using two global atomics:

```cpp
struct AtomicGridSync {
    unsigned int *counter;     // blocks that have arrived
    unsigned int *generation;  // incremented when all blocks arrive
    unsigned int  nblocks;     // total blocks in the grid
    unsigned int  local_gen;   // last generation this block observed
    
    __device__ void sync() {
        __syncthreads();         // flush all threads in this block first
        if (threadIdx.x == 0) {
            // Fence ensures prior stores are visible before arrival is counted
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1) {
                // Last block resets counter and advances generation
                *counter = 0;
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                atomicAdd(generation, 1);
            } else {
                // Other blocks spin on generation with nanosleep to yield the warp
                while (*generation <= local_gen)
                    asm volatile("nanosleep.u32 256;" ::: "memory");
            }
            local_gen++;
        }
        __syncthreads();         // broadcast new generation to all threads in block
    }
};
```

The `nanosleep.u32 256` instruction (available from sm_80 / Ampere onward) yields the warp scheduler for ~256ns per iteration, allowing other warps to run instead of busy-spinning. The `fence.acq_rel.gpu` instructions ensure acquire/release semantics across the GPU global memory space — without them, the counter update and the subsequent data read are not ordered.

#### Decode Body — `ldg_decode_body`

The shared core logic, called by all three kernel variants:

**1. Embedding lookup** (block 0 only)
Block 0 reads `hidden_buffer[i] = embed_weight[token_id * HIDDEN_SIZE + i]` for all 1024 elements. All other blocks wait at `grid.sync()`.

**2. 28 transformer layers**, each consisting of:

- **Input RMSNorm** — computed in shared memory `s_norm[HIDDEN_SIZE]`. Uses two reductions: first a warp-level `cg::reduce` to sum squared values per warp, then a block-level sum. The inverse RMS is broadcast via shared memory, then applied element-wise with the `input_layernorm_weight` scale.

- **QKV projection** — `ldg_matvec_qkv_fp16`: computes the matrix-vector products `Q = q_weight × norm`, `K = k_weight × norm`, `V = v_weight × norm` in a single fused pass. The 4096 output rows (2048 Q + 1024 K + 1024 V) are split across blocks; each warp handles several rows. Loads use `ldg_load_weights_u4` (128-bit reads) which on sm_120 issue `ld.global.nc.v4.u32` — non-coherent (read-only) cache loads that hit the L1/L2 without polluting the coherency domain.

- **Attention** — `ldg_attention`:
  - Block 0 handles K-norm (per-head RMSNorm) + RoPE rotation + writing K and V into the KV cache.
  - Blocks 0–15 handle Q-norm (per-head RMSNorm) + RoPE rotation in parallel.
  - Remaining blocks prefetch the O-projection and MLP weights into L2 cache while attention runs.
  - After `grid.sync()`, blocks 0–15 each own one or more Q heads and compute the full causal attention score over all cached positions using online softmax (numerically stable incremental max/sum/acc update).
  - The warp-level partial results (max score, sum-of-exp, accumulated value vector) are stored in shared memory arrays; warp 0 reduces them to the final normalised output.

- **O-projection + Post-attention norm + SwiGLU MLP** — `ldg_o_proj_postnorm_mlp`:
  - O-projection: each block handles a slice of the 1024 output rows. Fuses the residual add (`g_activations[m] = dot(o_weight[m], attn_out) + hidden_residual[m]`).
  - Post-attention RMSNorm: block 0 computes the global inverse RMS and writes it to a single float in global memory (`g_norm_scratch`). All blocks read this scalar and apply it.
  - Gate projection and Up projection (SwiGLU): computed simultaneously per output row. Activation: `SiLU(gate) × up`. Written to `g_mlp_intermediate` (float32).
  - Down projection: reads `g_mlp_intermediate` with `float4` loads (16 bytes per load, fully coalesced), fuses the second residual add, writes `hidden_out` in float16.

**3. Final RMSNorm** — applied to `hidden_buffer` after all 28 layers; result written to `g_normalized` (float32) for the LM head.

#### Two-Phase LM Head

Projecting from `HIDDEN_SIZE=1024` to `VOCAB_SIZE=151,936` is the most compute-intensive single step. It is split across two kernels:

**Phase 1 — `ldg_lm_head_phase1`** (680 blocks × 256 threads):
Each block owns a shard of the vocabulary. Within each block, warps each handle `LDG_LM_ROWS_PER_WARP = 8` vocab rows simultaneously, unrolling the inner HIDDEN_SIZE loop in steps of 8 `half` elements (one `uint4` load). At the end each block writes its local argmax `(val, idx)` pair to `block_max_vals / block_max_idxs`.

**Phase 2 — `ldg_lm_head_phase2`** (1 block × 256 threads):
Reduces the 680 partial argmaxes to a single winner token ID written to `d_output_token` (device memory).

#### Weight Loads — `ldg_load_weights_u4`

```cpp
__device__ __forceinline__ uint4 ldg_load_weights_u4(const uint4 *ptr) {
    uint4 res;
    asm volatile(
        "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
        : "l"(ptr));
    return res;
}
```

`ld.global.nc` — non-coherent global load — tells the hardware this address will not be written by any other thread on this SM during the load's lifetime. This allows the hardware to use the read-only data cache path (separate from the L1 write cache) and avoids invalidation on sm_120's 64-byte cache lines. For streaming weight reads that are never reused, this keeps L2 pressure lower.

#### SiLU Activation

```cpp
__device__ __forceinline__ __half ldg_silu(__half x) {
    return __hmul(x, ptx_hrcp(__hadd(__float2half(1.0f), fast_exp(__hneg(x)))));
}
```

Computed in float16 using `fast_exp` = `exp2(x × log2(e))`. This avoids a float32 upcast for the activation, keeping the MLP intermediate computation in the lower-precision domain where throughput is higher.

---

## Three Kernel Entry Points

All three are C-linkage functions callable from the PyTorch C++ extension (`qwen_ops.cpp`).

### `launch_ldg_decode_direct`

Single-token decode. Launches `ldg_decode_kernel_direct` once, then the two LM head kernels. Used by `Decoder.step()` for testing and interactive use. Resets the barrier atomics before each call via `cudaMemsetAsync`.

### `launch_ldg_generate_nosync`

Batched no-sync generation — the primary production path. Takes `num_steps` as input and pre-queues all step launches into the CUDA stream without any per-step CPU synchronisation:

```
for step in 0..num_steps:
    cudaMemsetAsync(barrier_counter, 0, ...)   # reset sync state
    ldg_decode_kernel_persistent<<<...>>>      # forward pass
    ldg_lm_head_phase1<<<...>>>                # vocab projection
    ldg_lm_head_phase2<<<...>>>                # argmax
    ldg_update_step<<<1, 1>>>                  # write token, advance position
cudaStreamSynchronize(stream)                  # single sync after all steps
```

`ldg_update_step` is a tiny 1-thread kernel that:
- Reads the EOS flag; returns immediately if set (prevents writing garbage tokens after EOS)
- Writes the new token into `output_log[step]`
- Advances `d_mutable_position` and `d_mutable_token_id` in device memory
- Sets `d_eos_flag = 1` if the token equals `eos_token_id`

Subsequent decode kernels check `d_eos_flag` at the very start of `ldg_decode_body` and return immediately — the GPU pipeline stays active (blocks are scheduled) but execution is near-zero cost, matching the overhead of the remaining stream commands.

### `launch_ldg_prefill`

Processes all prompt tokens (except the last, which enters the decode path) in one C call:

```
cudaMemcpyAsync(d_prefill_token_ids, token_ids, num_tokens * sizeof(int), ...)

for t in 0..num_tokens:
    cudaMemsetAsync(barrier_counter/sense, ...)
    cudaMemcpyAsync(d_mutable_position, h_pinned_positions + t, ...)
    cudaMemcpyAsync(d_mutable_token_id, d_prefill_token_ids + t, ...)
    ldg_decode_kernel_persistent<<<...>>>   # builds KV cache for token t

cudaStreamSynchronize(stream)   # one sync after all tokens
```

Scratch buffers (`d_prefill_token_ids`, `h_pinned_positions`) are grow-only static allocations — allocated once and reused on every subsequent prefill call to avoid per-request malloc/free overhead.

---

## Memory Layout and Buffer Allocations

All scratch buffers are allocated once in `Decoder.__init__` and reused across every request. Sizes are derived from the compile-time model constants to ensure consistency:

| Buffer | Type | Shape / Size | Purpose |
|---|---|---|---|
| `_k_cache` | float16, CUDA | `[28, 8, 2048, 128]` | KV cache — K vectors |
| `_v_cache` | float16, CUDA | `[28, 8, 2048, 128]` | KV cache — V vectors |
| `_hidden` | float16, CUDA | `[1024]` | Hidden state (residual stream) |
| `_act` | float32, CUDA | `[1024]` | O-proj + residual accumulator |
| `_res` | float32, CUDA | `[1024]` | ABI compatibility (not read by kernel) |
| `_q` | float16, CUDA | `[2048]` | Query vectors |
| `_k` | float16, CUDA | `[1024]` | Key vector for current token |
| `_v` | float16, CUDA | `[1024]` | Value vector for current token |
| `_attn_out` | float16, CUDA | `[2048]` | Attention output |
| `_mlp_inter` | float32, CUDA | `[3072]` | SwiGLU intermediate |
| `_norm_out` | float32, CUDA | `[1024]` | Post-final-norm hidden (for LM head) |
| `_fmax_vals` | float32, CUDA | `[680]` | LM head Phase 1 partial maxima |
| `_fmax_idxs` | int32, CUDA | `[680]` | LM head Phase 1 partial argmax indices |
| `_output_log` | int32, CUDA | `[2048]` | Token IDs written on-device during generation |
| `_output_log_cpu` | int32, pinned | `[2048]` | Pinned mirror for fast DMA transfer |

The `_output_log_cpu` / `_output_log` pair enables a single DtoH copy at the end of generation (via `output_log.cpu()`) rather than one per token. Pinned memory allows the DMA engine to transfer directly without staging through pageable memory, reducing latency to approximately 10µs for 128 int32 tokens.

Total scratch buffer memory: approximately 110MB (dominated by the 235MB KV cache).

### `_pack_layer_weights` — Weight Pointer Table

The kernel accesses layer weights through an array of `LDGLayerWeight` structs, each holding 12 raw `half*` pointers (11 real + 1 padding). Python builds this table as a CUDA int64 tensor:

```python
def _pack_layer_weights(layer_weights: list) -> torch.Tensor:
    N = 11  # weights per layer
    for li in range(n_layers):
        for j in range(N):
            all_ptrs.append(layer_weights[li * N + j].data_ptr())
        all_ptrs.append(0)   # padding → 12 pointers × 8 bytes = 96 bytes
    
    # Align to 16-byte boundary by over-allocating 2 elements and offsetting
    t = torch.zeros(len(all_ptrs) + 2, dtype=torch.int64, device="cuda")
    offset_elems = (16 - (t.data_ptr() % 16)) % 16 // 8
    t_aligned = t[offset_elems : offset_elems + len(all_ptrs)]
    t_aligned.copy_(torch.tensor(all_ptrs, dtype=torch.int64))
    assert t_aligned.data_ptr() % 16 == 0
```

The 16-byte alignment is enforced by `qwen_ops.cpp` with a `TORCH_CHECK` before every kernel call. If violated, the kernel would dereference misaligned `LDGLayerWeight*` structs, producing `CUDA_ERROR_MISALIGNED_ADDRESS`.

### KV Cache — Partial Reset

The KV cache is allocated once as `[28, 8, 2048, 128]` float16 (approximately 235MB) and reused across requests. Instead of zeroing the full buffer each time, the `Decoder` tracks a high-water mark:

```python
def reset(self):
    self._position = 0
    if self._kv_high_water > 0:
        hw = self._kv_high_water
        self._k_cache[:, :, :hw, :].zero_()
        self._v_cache[:, :, :hw, :].zero_()
    self._kv_high_water = 0
```

For a 16-token prompt + 32-token output, this zeros only 48 positions × 2 caches × 28 layers × 8 heads × 128 dims × 2 bytes = ~3.4MB instead of the full 235MB — approximately 70x less data. The high-water mark is updated at the end of `generate()` to `n_prompt + len(output_tokens)`.

---

## Python Extension Interface — `qwen_ops.cpp`

The C++ file wraps the three CUDA entry points as PyTorch ops registered via `pybind11`. Key design points:

- **Stream acquisition**: `c10::cuda::getCurrentCUDAStream()` ensures the kernel runs on PyTorch's current CUDA stream, maintaining correct ordering with any prior torch operations.
- **Alignment guard**: All three ops check `layer_weights_packed.data_ptr() % 16 == 0` and raise `TORCH_CHECK` with a descriptive error if violated.
- **`generate_nosync` output**: Takes a caller-supplied `output_log` tensor (validated for dtype int32, CUDA, contiguous, and at least `num_steps` elements) and returns the same tensor. The caller pre-allocates once and reuses it.
- **`prefill` token IDs**: Requires CPU int32 contiguous tensor — the CUDA kernel copies it to device internally.
- **ABI version**: `abi_version()` returns `2`, validated by Python on import.

---

## Weight Loading — `Qwen06B_architecture.py`

### RoPE Table

Precomputed once at load time for all positions up to `MAX_SEQ_LEN=2048`:

```python
inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2) / HEAD_DIM))
positions = torch.arange(MAX_SEQ_LEN)
freqs     = torch.outer(positions, inv_freq)
cos_table = torch.cos(freqs).repeat(1, 2).to(torch.float16).cuda()
sin_table = torch.sin(freqs).repeat(1, 2).to(torch.float16).cuda()
```

`repeat(1, 2)` duplicates the frequency dimension so that both halves of the head vector use the same cos/sin value, matching the RoPE variant used by Qwen3. The tables have shape `[2048, 128]` and live permanently in GPU memory.

### Chain-of-Thought Suppression

The FastAPI server injects an empty `<think>` block before the assistant turn to skip Qwen3's chain-of-thought phase:

```python
prompt = re.sub(r'<think>.*?</think>\n*', '', prompt, flags=re.DOTALL)
prompt = prompt.rstrip() + '\n<think>\n\n</think>\n\n'
```

This reduces output tokens for factual queries from ~200+ (with reasoning trace) to ~10–40 (direct answers), directly improving measured latency. The benchmark prompts are all factual single-answer questions designed to benefit from this suppression.

---

## FastAPI Server — `megakernel.py`

An OpenAI-compatible HTTP server exposing `/v1/chat/completions`, `/v1/models`, and `/health`.

Key design decisions:

- **`ThreadPoolExecutor(max_workers=1)`**: All generation runs on a single background thread. This serialises GPU access without needing locks — Python's GIL is released during the `await loop.run_in_executor()` call, so the event loop remains responsive for health checks and model listing during inference.
- **`ORJSONResponse`**: Uses `orjson` for JSON serialisation if available (typically 3–10x faster than the standard library for small payloads), falling back to the standard `JSONResponse`.
- **Timestamp caching**: `_get_ts()` caches `int(time.time())` for 1 second to avoid repeated syscalls on high-throughput paths.
- **Request counter**: `itertools.count(1)` provides a lock-free monotonically increasing request ID for the `chatcmpl-N` response field.
- **Lifespan context manager**: Uses the FastAPI `@asynccontextmanager` lifespan pattern (required in FastAPI ≥ 0.93, replacing the deprecated `@app.on_event("startup")`).

---

## vLLM Baseline — `serve.sh`

Launches `vllm.entrypoints.openai.api_server` with flags that make it as comparable to the megakernel as possible:

| Flag | Value | Reason |
|---|---|---|
| `--enforce-eager` | — | Disables CUDA graph capture; uses standard one-kernel-per-op dispatch |
| `--max-num-seqs` | 1 | One request at a time; matches the megakernel's concurrency |
| `--dtype` | float16 | Matches megakernel precision |
| `--kv-cache-dtype` | float16 | Matches megakernel KV cache precision |
| `--scheduling-policy` | fcfs | No priority reordering |
| `--block-size` | 16 | PagedAttention block granularity |
| `--gpu-memory-utilization` | 0.90 | 90% VRAM for KV blocks |
| `--disable-log-requests` | — | Reduces terminal noise during the 100-request benchmark |

Without `--enforce-eager`, vLLM would capture CUDA graphs on warm-up and the comparison would measure graph replay vs. the megakernel's persistent kernel, which is a different trade-off.

---

## Benchmark Client — `client_benchmark.py`

Sends 100 requests from `prompt.py` to whichever server is pointed to by `HOST` in `.env`. Supports sequential (`CONCURRENCY=1`) and parallel (`CONCURRENCY=N`) modes. Latency is measured with `time.perf_counter()` around the blocking `client.chat.completions.create()` call, capturing full end-to-end wall time including HTTP overhead, server-side queueing, tokenisation, inference, and detokenisation.

### Metrics reported

- Average, median, standard deviation, min, max latency
- Throughput in requests/second
- All results saved to a JSON log file (configurable via `LOG_FILE` in `.env`)

---

## Key Design Decisions

### Single kernel launch per inference phase

`launch_ldg_generate_nosync` queues all `N` decode steps into the CUDA stream before calling `cudaStreamSynchronize` once at the end. Traditional frameworks (vLLM, HuggingFace) launch a separate kernel per operation per token. For a 32-token response with 28 layers, this is roughly:

- vLLM: `32 × (embed + 28 × ~8 ops + LM head) ≈ 32 × 225 = 7,200 kernel launches`
- Megakernel: `32 × (1 decode + 2 LM head + 1 step update) = 128 kernel launches`

Each launch has a fixed driver overhead of ~5–15µs. Eliminating 7,000+ launches saves 35–100ms per request.

### Partial KV cache reset

Only zeros positions `0..high_water_mark` of the KV cache rather than the full buffer. For a 16-token prompt + 32-token output this is ~70x less data zeroed per request. At float16 bandwidth of ~1.8 TB/s on the RTX 5090, zeroing 3.4MB takes ~2µs vs. ~130µs for the full 235MB.

### Single `cudaStreamSynchronize` per request

All token generation steps, LM head projections, and step-update kernels are queued into one stream. The GPU pipeline is kept full from prompt prefill through the last decode step. The only CPU sync is the single `cudaStreamSynchronize(stream)` at the end of `launch_ldg_generate_nosync`, plus the implicit sync from the subsequent `output_log.cpu()` (DtoH copy, already ordered by the stream sync).

### Pinned memory output buffer

The 2048-element `output_log` tensor is allocated in pinned (page-locked) CPU memory via PyTorch's `.pin_memory()`, then mirrored to a CUDA tensor via `.cuda()`. DtoH transfers from device memory to pinned memory bypass the OS page fault handler and are DMA-direct, taking approximately 10µs for 128 int32 values versus ~50–100µs for pageable memory.

### EOS early exit without CPU round-trip

When the kernel generates the EOS token, `ldg_update_step` sets `d_eos_flag = 1` on the device. Subsequent `ldg_decode_kernel_persistent` launches check this flag at the very start of `ldg_decode_body` and return immediately — no CPU intervention required. The stream continues with the remaining queued launches but they complete in nanoseconds. This avoids the CPU needing to inspect each token, which would require a per-step DtoH copy and sync.

### L2 prefetch during attention

While blocks 0–15 compute attention and block 0 writes the KV cache, the remaining 154 blocks are idle. The megakernel uses this slack to prefetch the O-projection and MLP weight matrices into L2 cache using non-coherent `ld.global.nc` loads with a deliberate `asm volatile("" ...)` fence to prevent the compiler from eliminating them. When these weights are needed immediately after the grid barrier, they are already resident in L2, reducing effective memory latency for the projection kernels.

### MaxL1 cache carveout

```cpp
cudaFuncSetAttribute(ldg_decode_kernel_persistent,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxL1);
```

On Blackwell, the L1/shared memory is unified and configurable. Setting `MaxL1` allocates as much of the on-chip SRAM as possible to the L1 data cache rather than shared memory. The decode kernel uses shared memory for RMSNorm scratch and the attention accumulator (`s_out_acc[LDG_NUM_WARPS][HEAD_DIM]`), but the dominant access pattern is weight streaming — maximising L1 improves hit rates for the per-layer weight pointer table and the embedding lookup.

---

## Setup by Platform

All platforms ultimately run the same server on Linux. The difference is only in how you connect and copy files.

---

### Windows (VS Code + WSL)

**Prerequisites:**
- VS Code with the Remote - SSH extension installed
- WSL2 with Ubuntu (run `wsl --install` in PowerShell if not already set up)
- NVIDIA Nsight Systems 2025.3.2 for viewing profiles (Windows native)

**Step 1 - Generate SSH key in WSL:**
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
```
Paste the public key into your Vast.ai instance under Manage SSH Keys.

**Step 2 - Connect VS Code to the remote server:**

1. Open VS Code and press Ctrl+Shift+P
2. Type Remote-SSH: Connect to Host, then Add New SSH Host
3. Enter: `ssh -p <PORT> root@<IP>`
4. Select `C:\Users\YourName\.ssh\config` to save
5. Click Connect - VS Code opens a new window on the server

**Step 3 - Copy files from WSL to server:**
```bash
export SERVER_IP=<your_instance_ip>
export SERVER_PORT=<your_instance_port>

ssh -i ~/.ssh/id_ed25519 -p $SERVER_PORT root@$SERVER_IP "mkdir -p ~/qwen_megakernel/Model"

scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  megakernel_5090.cu megakernel.py qwen_ops.cpp \
  rmsnorm.cu rmsnorm.cuh swiglu.cu swiglu.cuh swiglu_binding.cpp \
  client_benchmark.py serve.sh setup.py prompt.py \
  root@$SERVER_IP:~/qwen_megakernel/

scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  Model/Qwen06B_architecture.py \
  root@$SERVER_IP:~/qwen_megakernel/Model/
```

**Step 4 - SSH tunnel for accessing the server locally:**
```bash
ssh -i ~/.ssh/id_ed25519 -p $SERVER_PORT root@$SERVER_IP \
    -L 8000:localhost:8000 -N -o ServerAliveInterval=30 &

curl http://localhost:8000/health
```

**Viewing Nsight profiles on Windows:**

Copy .nsys-rep files from the server to WSL:
```bash
scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  root@$SERVER_IP:/tmp/megakernel_profile.nsys-rep \
  ~/your_project/megakernel_profile.nsys-rep
```

Open in Nsight Systems on Windows via File > Open and navigate to:
```
\\wsl.localhost\Ubuntu\home\<your_username>\your_project\megakernel_profile.nsys-rep
```

---

### macOS

**Prerequisites:**
- VS Code with the Remote - SSH extension
- Homebrew: `brew install openssh`

macOS has no CUDA support. All GPU work runs on the remote Linux server. Your Mac is purely the client and editor.

**Step 1 - Generate SSH key:**
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
```
Paste the public key into Vast.ai under Manage SSH Keys.

**Step 2 - Connect VS Code:**

1. Install the Remote - SSH extension in VS Code
2. Press Cmd+Shift+P and type Remote-SSH: Connect to Host
3. Enter: `ssh -p <PORT> root@<IP>`
4. Select `~/.ssh/config` to save
5. Click Connect

**Step 3 - Copy files from Mac to server:**
```bash
export SERVER_IP=<your_instance_ip>
export SERVER_PORT=<your_instance_port>

ssh -i ~/.ssh/id_ed25519 -p $SERVER_PORT root@$SERVER_IP "mkdir -p ~/qwen_megakernel/Model"

scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  megakernel_5090.cu megakernel.py qwen_ops.cpp \
  rmsnorm.cu rmsnorm.cuh swiglu.cu swiglu.cuh swiglu_binding.cpp \
  client_benchmark.py serve.sh setup.py prompt.py \
  root@$SERVER_IP:~/qwen_megakernel/

scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  Model/Qwen06B_architecture.py \
  root@$SERVER_IP:~/qwen_megakernel/Model/
```

**Step 4 - SSH tunnel:**
```bash
ssh -i ~/.ssh/id_ed25519 -p $SERVER_PORT root@$SERVER_IP \
    -L 8000:localhost:8000 -N -o ServerAliveInterval=30 &

curl http://localhost:8000/health
```

**Viewing Nsight profiles on Mac:**

Nsight Systems does not have a macOS GUI. Options:
- Copy .nsys-rep to a Windows or Linux machine with Nsight Systems installed
- Use the Nsight Systems CLI on the server: `nsys stats /tmp/megakernel_profile.nsys-rep`

---

### Local RTX 5090 - Windows + WSL2

If you have a physical RTX 5090 in your Windows machine, you can run everything locally via WSL2 with CUDA passthrough. No remote server needed.

**Prerequisites:**
- Windows 11 with WSL2 enabled
- NVIDIA Driver 580+ installed on Windows (WSL2 inherits this automatically)
- Ubuntu 24.04 in WSL2

**Step 1 - Verify CUDA is accessible in WSL2:**
```bash
nvidia-smi        # should show RTX 5090 and CUDA 13.0
nvcc --version    # should show CUDA 13.0
```

If nvcc is not found, install the CUDA toolkit from developer.nvidia.com/cuda-downloads and select WSL-Ubuntu as the target.

**Step 2 - Copy files into WSL:**
```bash
mkdir -p ~/qwen_megakernel/Model

# Access Windows files via /mnt/c/
cp /mnt/c/Users/YourName/qwen_project/megakernel_5090.cu ~/qwen_megakernel/
cp /mnt/c/Users/YourName/qwen_project/Model/Qwen06B_architecture.py ~/qwen_megakernel/Model/
# repeat for all files listed in Repository Structure above
```

**Step 3 - Install dependencies:**
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers fastapi "uvicorn[standard]" orjson pydantic openai ninja accelerate vllm
```

**Step 4 - Build the CUDA extension:**
```bash
cd ~/qwen_megakernel

CUDA_PATH=$(dirname $(which nvcc))/..
CUDA_HOME=$CUDA_PATH TORCH_CUDA_ARCH_LIST="12.0a" python setup.py build_ext --inplace
```

On local WSL2, the CUDA version mismatch patch is usually not needed. If you see CUDA_MISMATCH_MESSAGE, apply the same sed patch from the cloud setup section below.

**Step 5 - Full ncu profiling is available on local GPU:**

Unlike cloud containers, a local GPU gives full hardware counter access:
```bash
ncu --set full \
    --export ~/qwen_megakernel/megakernel_ncu_profile \
    python -c "
from Model.Qwen06B_architecture import Decoder
d = Decoder(verbose=False)
d.generate('What is the capital of France?', max_tokens=32)
"
```

Open `megakernel_ncu_profile.ncu-rep` in NVIDIA Nsight Compute on Windows:
```
\\wsl.localhost\Ubuntu\home\<username>\qwen_megakernel\megakernel_ncu_profile.ncu-rep
```

Everything from the cloud setup Steps 4 onward applies identically. The server runs in WSL and is accessible at localhost:8000 from both WSL and Windows.

---

### Local RTX 5090 - Native Linux

**Prerequisites:**
- Ubuntu 22.04 or 24.04
- NVIDIA Driver 580+: `sudo apt install nvidia-driver-580`
- CUDA 13.0 toolkit

**Step 1 - Verify:**
```bash
nvidia-smi        # should show RTX 5090
nvcc --version    # should show CUDA 13.0
```

**Step 2 - Install dependencies:**
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers fastapi "uvicorn[standard]" orjson pydantic openai ninja accelerate vllm
```

**Step 3 - Build:**
```bash
cd ~/qwen_megakernel

CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="12.0a" \
python setup.py build_ext --inplace
```

On native Linux the CUDA version mismatch patch is usually not needed. If you hit the error, apply the same sed patch from the cloud setup section.

Full ncu profiling works without restrictions on native Linux:
```bash
ncu --set full --export /tmp/megakernel_ncu \
    python -c "
from Model.Qwen06B_architecture import Decoder
d = Decoder(verbose=False)
d.generate('What is the capital of France?', max_tokens=32)
"
```

---

## Cloud Setup (Vast.ai) - Full Steps

### Step 1 - Provision Instance

Rent an RTX 5090 on Vast.ai (https://cloud.vast.ai):
- GPU: RTX 5090
- Image: pytorch/pytorch:latest or any Ubuntu 24 image
- Disk: 30GB minimum

### Step 2 - Install Dependencies

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers fastapi "uvicorn[standard]" orjson pydantic openai ninja accelerate vllm
```

### Step 3 - Build the CUDA Extension

```bash
cd ~/qwen_megakernel

# Patch PyTorch CUDA version check (driver 13.0 vs PyTorch built with 12.8)
LINE=$(grep -n "raise RuntimeError(CUDA_MISMATCH_MESSAGE" \
    /venv/main/lib/python3.12/site-packages/torch/utils/cpp_extension.py | cut -d: -f1)
sed -i "${LINE}s/raise RuntimeError/pass  # raise RuntimeError/" \
    /venv/main/lib/python3.12/site-packages/torch/utils/cpp_extension.py

CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="12.0a" \
python setup.py build_ext --inplace
```

Verify:
```bash
python -c "import qwen_megakernel_C; print('ABI version:', qwen_megakernel_C.abi_version())"
# Expected: ABI version: 2
```

### Step 4 - Create .env

```bash
cat > ~/qwen_megakernel/.env << 'EOF'
HOST=http://localhost
PORT=8000
MODEL=Qwen/Qwen3-0.6B
DTYPE=float16
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=1
BLOCK_SIZE=16
SWAP_SPACE=4
KV_CACHE_DTYPE=auto
NUM_REQUESTS=100
MAX_TOKENS=32
TEMPERATURE=0.0
LOG_FILE=latency_log_megakernel.json
CONCURRENCY=1
EOF
```

### Step 5 - Run Megakernel Server

```bash
cd ~/qwen_megakernel
python megakernel.py
```

Expected output:
```
[megakernel_server] Loading Qwen/Qwen3-0.6B with megakernel...
All weight pointers 16-byte aligned OK
[megakernel_server] Model loaded in ~8s
[megakernel_server] Serving on port 8000
```

Health check:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Step 6 - Run Megakernel Benchmark

In a second terminal on the server:
```bash
cd ~/qwen_megakernel
python client_benchmark.py
```

Raw kernel benchmark with no HTTP overhead:
```bash
python -c "
from prompt import PROMPTS
from Model.Qwen06B_architecture import Decoder
import time, statistics

d = Decoder(verbose=False)
d.generate(PROMPTS[0], max_tokens=32)

latencies = []
for p in PROMPTS:
    t0 = time.perf_counter()
    d.generate(p, max_tokens=32)
    latencies.append(time.perf_counter() - t0)

print(f'Avg:    {statistics.mean(latencies):.3f}s')
print(f'Median: {statistics.median(latencies):.3f}s')
print(f'Min:    {min(latencies):.3f}s')
print(f'Max:    {max(latencies):.3f}s')
print(f'Tok/s:  {32/statistics.mean(latencies):.1f}')
"
```

### Step 7 - Run vLLM Baseline

```bash
pkill -f megakernel.py
sed -i 's/LOG_FILE=.*/LOG_FILE=latency_log_vllm_baseline.json/' .env
bash serve.sh
```

Wait for `Application startup complete`, then in a second terminal:
```bash
python client_benchmark.py
```

### Step 8 - Profile with Nsight Systems

```bash
export PATH=/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64:$PATH

# Megakernel - 100 prompts
nsys profile --output /tmp/megakernel_profile_100 \
    python -c "
from prompt import PROMPTS
from Model.Qwen06B_architecture import Decoder
d = Decoder(verbose=False)
d.generate(PROMPTS[0], max_tokens=32)
for p in PROMPTS:
    d.generate(p, max_tokens=32)
"

# vLLM - profile the server process directly
nsys profile \
    --output /tmp/vllm_server_profile \
    --trace cuda,cudnn,cublas,osrt \
    --trace-fork-before-exec true \
    python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --dtype float16 \
    --port 8000 &

until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 2; done

python -c "
from prompt import PROMPTS
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='x')
for p in PROMPTS:
    client.chat.completions.create(
        model='Qwen/Qwen3-0.6B',
        messages=[{'role':'user','content':p}],
        max_tokens=32,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}}
    )
"
pkill -f vllm
```

Copy profiles to local machine:
```bash
scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  root@$SERVER_IP:/tmp/megakernel_profile_100.nsys-rep \
  root@$SERVER_IP:/tmp/vllm_server_profile.nsys-rep \
  ./profiles/
```

Open in NVIDIA Nsight Systems 2025.3.2.

---

## Key Design Decisions

**Single kernel launch per inference phase** — `launch_ldg_generate_nosync` runs all 32 decode steps inside one launch using GPU-side grid barriers between layers. Traditional frameworks launch a separate kernel per operation per token.

**Partial KV cache reset** — only zeros positions actually written (tracked via high-water mark) instead of clearing the full 235MB cache every request. For a 16-token prompt + 32-token output this is ~42x less data zeroed.

**Single cudaStreamSynchronize per request** — all N token generation steps run on-device with no CPU sync until the final DtoH transfer of the output log.

**Pinned memory output buffer** — the output token log is backed by pinned CPU memory for fast DMA transfer (~10us for 128 int32 tokens).

---

## Troubleshooting

| Error | Fix |
|---|---|
| `CUDA_MISMATCH_MESSAGE` during build | Apply the sed patch in the build step |
| `ABI version mismatch` | Rebuild: `python setup.py build_ext --inplace --force` |
| `qwen_megakernel_C op decode unavailable` | PyTorch was upgraded - rebuild the extension |
| `Address already in use` on port 8000 | `pkill -f megakernel.py` or `pkill -f vllm` |
| `ERR_NVGPUCTRPERM` for ncu | Container restriction — use nsys instead, or use a local or bare-metal GPU for full ncu access |
| `nvidia-smi not found` in WSL2 | Install NVIDIA driver 580+ on the Windows host — WSL2 inherits it automatically |
