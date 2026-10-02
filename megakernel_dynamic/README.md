# Qwen3-0.6B megakernel with a dynamic persistent tile scheduler

A rebuild of `Tools/megakernel` around the scheduling idea in CUTLASS's
`python/CuTeDSL/cutlass/utils/dynamic_persistent_tile_scheduler.py`
(`ClcDynamicPersistentTileScheduler`), targeting **B200 (sm_100a)**. The rebuild
exists twice: as CUDA C++ (`cuda/`) and as CuTeDSL (`cutedsl/`).

Reference material from the CUTLASS checkout (`~/cutlass`):

| File | What was taken |
|---|---|
| `python/CuTeDSL/cutlass/utils/dynamic_persistent_tile_scheduler.py` | scheduler API: initial work = own launch, `advance_to_next_work` issues a CLC query, `get_current_work` decodes it |
| `python/CuTeDSL/cutlass/pipeline/sm100.py` (`PipelineClcFetchAsync`) | full/empty mbarrier ring that hands work from the scheduler warp to the other warps |
| `examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic.py` | warp specialisation: scheduler warp + TMA load warp + compute warps |
| `media/docs/cpp/blackwell_cluster_launch_control.md` | CLC rules (grid = work tiles, `try_cancel` steals CTAs that have not launched) |
| `examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py` | the block-scaled formats: MXF8 (e4m3 + e8m0 per 32) and NVF4 (e2m1 + e4m3 per 16) |

## What changed compared with `Tools/megakernel`

| `Tools/megakernel/megakernel_5090.cu` | this folder |
|---|---|
| 170 CTAs (= 5090 SM count), static row split per phase | work cut into ~33k tiles per token, handed out dynamically |
| ~8 software grid barriers per layer; 154 CTAs idle during attention | no grid barriers; each tile waits only on the counter of the phase it reads |
| weights loaded after each barrier | a TMA warp streams the next tiles' weights into a 6 x 32 KB smem ring *while* the compute warps wait on dependencies |
| 4 launches per token (decode, 2 x LM head, step update) + memsets | **one launch per request**: prefill, every decode step, LM-head argmax and token feedback all on-device |
| mixed fp16/fp32 intermediates | fp16 rounding at the same points as HF transformers, ties in argmax resolved like `torch.argmax` |
| fp16 weights only | fp16, **fp8 (MXF8)** or **fp4 (NVF4)** weights, dequantised inside the GEMV |

## How the CUTLASS idea maps onto a megakernel

```
grid = one CTA per tile   (ClcDynamicPersistentTileSchedulerParams.get_grid_shape)

scheduler warp      : own launch → ticket;  loop { try_cancel → if granted: ticket → publish to ring }
load warp  (TMA)    : read ticket → cp.async.bulk the tile's weights into the smem ring (no activation dependency)
8 compute warps     : read ticket → wait on dependency counter → GEMV / attention from smem → release counter
```

**Tile graph, per layer** (`T_LAYER = 1088` tiles, 28 layers, plus 2374 LM-head tiles per generated token):

| phase | tiles | per tile | waits for |
|---|---|---|---|
| QKV | 256 | 16 rows of q/k/v (32 KB), grouped per KV head | previous layer's DOWN (layer 0: previous token's argmax) |
| ATTN | 64 | one KV head x one 256-position split (both query heads) | the 32 QKV tiles of *its own* KV head only |
| O-proj | 128 | 8 rows of `o_proj` + residual | all 8 KV heads combined |
| gate/up | 384 | 8 gate + 8 up rows, SiLU*up | all O-proj tiles |
| down | 256 | 4 rows of `down_proj` + residual | all gate/up tiles |
| LM head | 2374 | 64 vocab rows (4 x 32 KB chunks), local argmax | last layer's DOWN |

RMSNorms are recomputed by each tile that needs them (cached per CTA). The
last attention split for a head merges the splits, and the last LM tile does
the global argmax. Both use the split-K "last block finishes" pattern rather
than another phase.

**One deliberate difference from CUTLASS.** GEMM tiles are independent, so CUTLASS
uses the cancelled CTA's `blockIdx` as the tile. Megakernel tiles depend on
earlier tiles, which is only deadlock-free if tiles are handed out in
dependency order, and the hardware promises no order for `try_cancel`.
So here a granted launch/cancel is a *permission* to run one tile, and the tile
itself is an ordered ticket from a global counter. Permissions = grid size = tiles,
so every ticket is used exactly once, and the lowest unfinished ticket is always
some running CTA's current tile with all its inputs ready. Co-residency is
therefore not required. The `oneshot` test mode (below) checks exactly this
on any GPU.

## Weight formats (fp16 / fp8 / fp4)

Weight-only quantization: activations, norms, the KV cache and the embedding
lookup stay fp16. The formats are the block-scaled ones Blackwell's tensor cores
use (same definitions as CUTLASS's `blockscaled_gemm` examples):

| format | values | block scale | per-tensor scale | bytes / weight | ring on B200 |
|---|---|---|---|---|---|
| `fp16` | fp16 | — | — | 2 | 6 x 32 KB |
| `fp8` = MXF8 | e4m3 | e8m0 (power of two) per 32 along K | — | 1.03 | 11 x 16.5 KB |
| `fp4` = NVF4 | e2m1, 2 per byte, low nibble first | e4m3 per 16 along K | fp32 | 0.56 | 16 x 9 KB |

Decode at batch 1 is bound by weight bandwidth, so these cut per-token weight traffic
from ~1.15 GB to ~0.6 GB (fp8) or ~0.33 GB (fp4). Tiles keep the same rows in every
format, so the task graph, dependencies and scheduling are unchanged; a stage holds a
chunk's weight rows followed by their scale rows, and the ring and claim-ahead get
deeper because stages are smaller. Tensor cores are not used: at batch 1 a GEMV has
nothing to tile over N. Instead each lane unpacks two weights at a time to fp16
(`cvt.rn.f16x2.e4m3x2` / `cvt.rn.f16x2.e2m1x2`, SASS `F2FP...UNPACK_B`) and feeds them
to `fma.rn.f32.f16` (SASS `FHFMA`, fp32 += fp16 x fp16, new on sm_100). That is ~1.5
instructions per weight, so even fp4 stays memory-bound. The block scale is applied
once per 16 weights and the per-tensor scale once per row.

`qwen_dps.quantize_weights` does round-to-nearest quantization on the GPU (the LM head,
tied to the embedding, gets its own quantized copy). Quality on 512 tokens of
text, teacher-forced, HF transformers with the dequantized weights:

| weights | perplexity | next-token agreement with fp16 | KL(fp16 ‖ q) | weight error |
|---|---|---|---|---|
| fp16 | 9.24 | — | — | — |
| fp8 (MXF8) | 9.32 (+0.9 %) | 94.3 % | 0.011 | 2.7 % |
| fp4 (NVF4) | 10.70 (+16 %) | 84.1 % | 0.147 | 9.5 % |

fp8 is close to lossless. fp4 with plain round-to-nearest costs real quality on a model
this small; the kernel only reads the packed values and scales, so a better quantizer
(GPTQ/AWQ-style, or keeping the most sensitive matrices in fp8) can be dropped in.

## Scheduler modes

| mode | grid | how work is fetched | notes |
|---|---|---|---|
| `clc` (default on sm_100+) | one CTA per tile | `clusterlaunchcontrol.try_cancel` + ordered ticket | elastic if other work occupies SMs; after EOS the remaining grid is cancelled in batches of 16 queries |
| `atomic` | one CTA per SM (148 on B200) | ordered ticket only (software dynamic persistent scheduler) | fallback for pre-Blackwell GPUs; worth benchmarking against `clc` |
| `oneshot` (CUDA, test only) | one CTA per tile | one ticket per CTA, no stealing | reproduces CLC's launch pattern without CLC hardware |

## Layout

```
megakernel_dynamic/
  qwen_dps.py               DpsDecoder: weight loading, MXF8/NVF4 quantization, layer table, generate()
  test_dps.py               greedy-token comparison against HF transformers + latency, per weight format
  cuda/
    dps_arch.cuh            PTX wrappers: mbarrier, cp.async.bulk, CLC; mbarrier/TMA emulation below sm_90
    dps_scheduler.cuh       DynamicPersistentTileScheduler (port of the CuTeDSL class + fetch ring)
    qwen_dps_megakernel.cu  task graph, tiles, warp roles, host launcher
    qwen_dps.h, qwen_dps_ops.cpp, setup.py
  cutedsl/
    qwen_dps_cutedsl.py     same kernel in CuTeDSL, using cutlass.utils.ClcDynamicPersistentTileScheduler
```

## Build and run on the B200 server

Requires CUDA 12.8+ (13.x tested for compilation), PyTorch with CUDA, `transformers`,
and `nvidia-cutlass-dsl>=4.4` for the CuTeDSL backend.

```bash
cd megakernel_dynamic/cuda && python setup.py build_ext --inplace
```

```bash
python megakernel_dynamic/test_dps.py
```

```bash
python megakernel_dynamic/test_dps.py --backend cutedsl
```

`test_dps.py` runs every weight format (`--formats fp16,fp8,fp4`) with every scheduler
mode the GPU supports. Each format is checked against HF running the same weights:
the fp16 model for fp16, and the model with dequantized weights for fp8/fp4. That
isolates kernel bugs from quantization error. It also prints how many leading tokens
still agree with the fp16 model, and times a 128-token decode with EOS disabled
(`--bench-tokens N` to change it). `DPS_ARCH=120a` builds for an RTX 5090. On any
other GPU `DPS_ARCH=<cc>` builds the atomic scheduler with emulated TMA.

Using it from Python (same `generate()` contract as `Model/Qwen06B_architecture.Decoder`):

```python
from qwen_dps import DpsDecoder, load_weights
weights, tokenizer, _ = load_weights()
dec = DpsDecoder(weights, tokenizer, backend="cuda", sched="auto",   # or backend="cutedsl"
                 weight_format="fp8")                                 # "fp16" | "fp8" | "fp4"
text, n_prompt, n_out = dec.generate(prompt, max_tokens=128)
```

Tuning knobs (CUDA): `DPS_RING_BYTES` (shared memory for the weight ring, default
192 KB on B200; each format cuts it into as many stages as fit, up to 16) and
`DPS_SSTAGES` (tiles a CTA may claim ahead at fp16, default 6; scaled up to 12 for
the smaller fp8/fp4 tiles), plus the tile sizes at the top of `qwen_dps_megakernel.cu`.
`cutedsl/qwen_dps_cutedsl.py` mirrors those constants.

## Verification status

Done locally on a GTX 1650 (sm_75, so no CLC or TMA hardware):

- CUDA backend, `atomic` and `oneshot` modes, emulated mbarrier/TMA path:
  greedy tokens identical to HF transformers on all test prompts, including a
  371-token prompt (multi-split attention) and early EOS.
- fp8 and fp4 (CUDA, `atomic`): greedy tokens identical to HF running the dequantized
  weights on every test prompt. The local GPU has no fp8/fp4 conversion or mixed FMA,
  so it runs a software fallback (~24 ms/token vs 9.6 ms for fp16 there). That timing
  says nothing about B200, where both are single instructions.
- Prompt split across two launches (`start_pos > 0`) produces the same tokens as one launch.
- Scheduler ring depths 1, 2 and 6 pass.
- sm_100a builds of both backends compile cleanly for all three formats (CUDA: 104-116
  registers, CuTeDSL: 74-103, no spills or local memory). The SASS contains `UGETNEXTWORKID`
  (CLC), `UBLKCP` (TMA bulk copy), `SYNCS.*` (mbarrier tx), and for fp8/fp4
  `F2FP.F16.E4M3/E2M1.UNPACK_B` + `FHFMA` in the GEMV.

**Not yet run on real hardware:** the `clc` mode, the real TMA/mbarrier path, the
hardware fp8/fp4 conversion path, and the CuTeDSL backend (CuTeDSL needs sm_80+).
Run `test_dps.py` for both backends on the B200 first.

## Limitations

- Batch 1. Prompt tokens are processed one at a time, as in the original kernel.
- `max_seq` up to 2048 (`ATTN_SPLITS x ATTN_CHUNK`).
- In `clc` mode an early EOS leaves the rest of the grid to be cancelled.
  The running CTAs drain it in batches, but `atomic` mode stops instantly.
  If generations often end long before `max_tokens`, compare the two modes.
- The CuTeDSL attention loop is not unrolled (the CUDA one processes 4 positions per iteration).
- The CuTeDSL fp8/fp4 paths need sm_100+ (`fma.rn.f32.f16`, `cvt.rn.f16x2.e2m1x2`). The CUDA
  build falls back to software conversion below that, but is only fast on Blackwell.
- Weight-only quantization. fp8/fp4 activations (W8A8/W4A4) would need a different
  kernel, and at batch 1 they would not reduce memory traffic much further.
