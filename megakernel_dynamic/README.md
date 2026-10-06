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
| 170 CTAs (= 5090 SM count), static row split per phase | work cut into ~17k tiles per token (at most 128 per phase), handed out dynamically or statically |
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

**Tile graph, per layer** (`T_LAYER = 576` tiles, 28 layers, plus 1187 LM-head tiles per generated token).
A chunk is one ring stage (32 KB at fp16); a tile streams its chunks one after another:

| phase | tiles | per tile | waits for |
|---|---|---|---|
| QKV | 128 | 32 rows of q/k/v (2 chunks), grouped per KV head | previous layer's DOWN (layer 0: previous token's argmax) |
| ATTN | 64 | one KV head x one 256-position split (both query heads) | the 16 QKV tiles of *its own* KV head only |
| O-proj | 128 | 8 rows of `o_proj` + residual (1 chunk) | all 8 KV heads combined |
| gate/up | 128 | 24 gate + 24 up rows (3 chunks), SiLU*up | all O-proj tiles |
| down | 128 | 8 rows of `down_proj` + residual (2 chunks) | all gate/up tiles |
| LM head | 1187 | 128 vocab rows (8 chunks), local argmax | last layer's DOWN |

No phase has more tiles than a B200 has SMs, so each phase is one round: with an even
hand-out every SM runs at most one tile of it. The first B200 traces showed why this
matters: a tile costs ~1.5 us of fixed latency (dependency check, input prep, barriers,
done-signal) but streams 32 KB in ~0.6 us, and the original tile sizes (256/384/256
tiles for QKV/gate-up/down) took 2-3 rounds per phase.

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
chunk's weight rows followed by their scale rows, and the ring gets deeper because
stages are smaller. Tensor cores are not used: at batch 1 a GEMV has
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
| `static` (CUDA) | one CTA per SM, cooperative launch | CTA c runs tickets c, c + G, c + 2G, ... (G = grid), no atomics | the static persistent baseline; with phases of at most G tiles, every CTA gets at most one tile per phase |
| `oneshot` (CUDA, test only) | one CTA per tile | one ticket per CTA, no stealing | reproduces CLC's launch pattern without CLC hardware |

In every mode a CTA claims its next ticket only once its work ring has a free slot, so it
holds at most `DPS_SSTAGES` (default 1) tickets it has not started. Claiming further ahead
lets the TMA warp prefetch more weights, but in the dynamic modes it hands tiles out before
anyone knows which will be ready first; see [B200 benchmarks](#b200-benchmarks).

## Layout

```
megakernel_dynamic/
  qwen_dps.py               DpsDecoder: weight loading, MXF8/NVF4 quantization, layer table, generate()
  test_dps.py               greedy-token comparison against HF transformers + latency, per weight format
  trace_dps.py              per-tile timeline (DPS_TRACE build): where a token step's time goes
  cuda/
    dps_arch.cuh            PTX wrappers: mbarrier, cp.async.bulk, CLC; mbarrier/TMA emulation below sm_90
    dps_scheduler.cuh       DynamicPersistentTileScheduler (port of the CuTeDSL class + fetch ring)
    qwen_dps_megakernel.cu  task graph, tiles, warp roles, host launcher
    qwen_dps.h, qwen_dps_ops.cpp, setup.py
  cutedsl/
    qwen_dps_cutedsl.py     same kernel in CuTeDSL, using cutlass.utils.ClcDynamicPersistentTileScheduler
```

## Build and run on the B200 server

Requires CUDA 12.8+ (the B200 run used 13.2), PyTorch with CUDA, `transformers`, and
`nvidia-cutlass-dsl==4.4.0` for the CuTeDSL backend. 4.4.0 is the version the backend
was written against; keep it pinned until the hang seen with 4.8.0 is understood.
PyTorch must be built for the same CUDA major version as `nvcc`, or the extension
build refuses to run, so on a CUDA 13 machine install the `cu130` wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

```bash
pip install numpy transformers nvidia-cutlass-dsl==4.4.0
```

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
(`--bench-tokens N` to change it). Each kernel call runs under a 60 s watchdog
(`--timeout S`, 0 turns it off): a hung kernel prints the Python stacks and exits
instead of holding the GPU. `DPS_ARCH=120a` builds for an RTX 5090. On any
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
`DPS_SSTAGES` (claim-ahead: tiles a CTA may claim before starting them, default 1, same for
every format), plus the tile sizes at the top of `qwen_dps_megakernel.cu`.
`cutedsl/qwen_dps_cutedsl.py` mirrors the claim-ahead, but still uses the original tile
sizes and has no `static` mode.

### Tracing where the time goes

`DPS_TRACE=1` builds a second module, `qwen_dps_trace_C`, next to the normal one. In it,
compute thread 0 writes one 48-byte record per tile for a chosen range of token steps:
`%globaltimer` stamps for when the scheduler claimed the ticket, when the compute warps got
it, when its dependencies were satisfied and when it finished, plus the time spent building
the input vector and waiting for weight stages. The normal build is unchanged.

```bash
cd megakernel_dynamic/cuda && DPS_TRACE=1 python setup.py build_ext --inplace && cd ../..
```

```bash
python megakernel_dynamic/trace_dps.py --formats fp16,fp8,fp4 --out-dir traces
```

`trace_dps.py` traces 8 decode steps per format and scheduler, then prints three views:
the mean cost of a tile in each phase split into queue / dependency wait / input prep /
weight wait / the rest; the critical path of a step (28 x QKV -> attention -> O -> gate/up
-> down, then the LM head), with each link's share of the step time and the hand-off delay
between phases; and how SM time divides between running tiles, waiting on dependencies and
sitting between tiles. It also times the same decode on the normal build, so the tracing
overhead is visible. `--load traces/*.npz` re-runs the analysis on saved traces without a GPU.

## B200 benchmarks

One vast.ai B200 (148 SMs), CUDA 13.2, PyTorch 2.14.1+cu130, batch 1, greedy decoding, the whole
request in one launch. CUDA backend only (the CuTeDSL backend did not run yet, see
[Verification status](#verification-status)). Raw logs and traces are in `results/`.

These runs used the original design: 256 / 64 / 128 / 384 / 256 tiles per layer for QKV /
attention / O-proj / gate-up / down, and a default claim-ahead of 6 (scaled to 11-12 for
fp8/fp4). The code has changed since, based on what they showed; see
[Changes since the B200 run](#changes-since-the-b200-run).

### Latency per token step

`test_dps.py`: 23-token prompt + 128 new tokens in one launch, total time divided by the 150
token steps (the 22 prefill steps skip the LM head). 1000 us per step is about 1000 tokens/s.

| weights | `atomic` | `clc` | `clc`, claim-ahead 1 | weight-bandwidth floor at 8 TB/s |
|---|---|---|---|---|
| fp16 | 1096 us | 1101 us | **825 us** (-25%) | ~144 us |
| fp8 | 976 us | 941 us | not measured | ~75 us |
| fp4 | 1419 us | 1448 us | **947 us** (-35%, claim-ahead 3) | ~41 us |

Claim-ahead is how many tiles each CTA claims before it needs them (`DPS_SSTAGES`, default 6,
scaled up to 12 for the smaller fp8/fp4 tiles). The last column was built with
`DPS_SSTAGES=1 python setup.py build_ext --inplace` and gives the same tokens. It is not the
default yet: fp8 and the `atomic` scheduler have not been measured with it.

### Where a token step goes

`trace_dps.py` with the default claim-ahead: every tile of 8 decode steps, taken after the
first 16 decode steps of the launch. "Untraced" is the same decode on the normal build.

| weights, scheduler | decode step, untraced | traced | SM time running tiles | waiting on dependencies |
|---|---|---|---|---|
| fp16, `atomic` | 1064 us | 1110 us | 37% | 60% |
| fp16, `clc` | 1076 us | 1128 us | 37% | 61% |
| fp8, `atomic` | 952 us | 986 us | 40% | 56% |
| fp8, `clc` | 939 us | 983 us | 40% | 56% |
| fp4, `atomic` | 1278 us | 1302 us | 29% | 68% |
| fp4, `clc` | 1236 us | 1349 us | 28% | 69% |

Per tile, the parts that look expensive are cheap: the weights are already in shared memory
when a tile needs them (mean wait ~0.04 us), a tile takes ~1.5-1.9 us from inputs ready to
done, and the hand-off between phases (last tile of one phase done -> first tile of the next
ready) is ~0.25 us. Yet each phase lasts 6-12 us, because its tiles are spread unevenly:

| phase (fp16, `clc`) | tiles | even share per SM | tiles on the busiest SM | SMs used | phase duration |
|---|---|---|---|---|---|
| QKV | 256 | 1.7 | 4.0 | 148 | 7.9 us |
| attention | 64 | 0.4 | 1.0 | 64 | 10.5 us |
| O-proj | 128 | 0.9 | 3.0 | 96 | 6.2 us |
| gate/up | 384 | 2.6 | 4.8 | 148 | 9.8 us |
| down | 256 | 1.7 | 4.1 | 148 | 7.6 us |

(Phase duration = first tile ready to last tile done, mean over 28 layers x 8 steps. Attention
has one tile per SM; its duration follows the per-head QKV groups it waits on.)

Each phase lasts about as long as its busiest SM needs: 4 tiles x ~1.7 us. The cause is
claim-ahead. Every CTA claims its next 6-12 tickets in advance so the TMA warp can prefetch
their weights. Across 148 CTAs that is one to two layers' worth of tiles (1088 per layer)
handed out before it is known which will become ready first. When a phase opens, some SMs
hold 2-3x their share and others none. With fp4 (claim-ahead 12) it is worst: the busiest SM averages 5.6 QKV tiles against a share
of 1.7. Claiming one tile ahead should keep the hand-out close to even, which is the likely
source of the 25-35% above; a trace with claim-ahead 1 has not been taken yet. The CLC and
atomic schedulers behave the same here, because both claim ahead the same way.

## Changes since the B200 run

Not yet measured on a B200. Each change follows from the traces above:

1. **Claim-ahead 1 by default, for every format.** A CTA also claims a ticket only once its
   work ring has a free slot for it. Before, the scheduler warp claimed its next ticket and
   then waited for a slot, so "claim-ahead 1" in the table above really held 2 tickets.
2. **One round per phase.** QKV, gate/up and down tiles are 2-3x larger (2, 3 and 2 ring
   stages each), so every phase has at most 128 tiles and no SM needs more than one tile of
   a phase. The LM head uses 128-row tiles (1187 instead of 2374).
3. **`static` scheduler** (CUDA only): CTA c runs tickets c, c + G, c + 2G, ... with no
   atomics, launched cooperatively so all CTAs are resident. With phases of at most G tiles,
   every CTA gets at most one tile per phase by construction. It is also the static baseline
   to compare CLC against.
4. **CuTeDSL deadlock fixed** (compiles for sm_100a with CuTeDSL 4.4.0 and 4.8.0; not run yet).
   See [Verification status](#verification-status).

How much the first two buy is open. From the traces: with the original tile sizes and a
perfectly even hand-out, a token step would still take ~0.52 ms (5 phases x 28 layers of
1-3 rounds of ~1.7 us tiles, plus ~0.25 us hand-offs and ~70 us of LM head). One round per
phase is meant to go below that, since the extra weight per tile streams in during the
dependency wait. The next B200 session measures it:

```bash
python megakernel_dynamic/test_dps.py
```

```bash
python megakernel_dynamic/test_dps.py --backend cutedsl
```

```bash
python megakernel_dynamic/trace_dps.py --formats fp16,fp8,fp4 --out-dir traces
```

`trace_dps.py` now also prints the hand-out balance per phase (tiles on the busiest SM
against the even share), which is the number the changes above are meant to fix.

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

After the [changes since the B200 run](#changes-since-the-b200-run), locally (GTX 1650):

- CUDA backend: greedy tokens identical to HF for fp16, fp8 and fp4 with the `atomic` and
  `static` schedulers, and for fp16 with `oneshot` (28 of 28 prompt checks).
- `trace_dps.py` on the new tiles: every phase's busiest SM runs the minimum possible
  (ceil(tiles / 14 SMs)); in `static` mode attention too (5 tiles against 6.5 in `atomic`).
  The GTX 1650 is bandwidth-bound, so its timings say nothing about B200.
- sm_100a: CUDA 100-114 registers (trace build 110-122), no spills or local memory;
  CuTeDSL compiles all six variants with 4.4.0 and 4.8.0 in ~1-2 s each.

On a B200 (2026-10-05, CUDA 13.2, PyTorch 2.14.1+cu130; raw logs in `results/`):

- CUDA backend: greedy tokens identical to HF transformers (fp16) and to HF running the
  dequantized weights (fp8/fp4) on every test prompt, with both the `atomic` and `clc`
  schedulers, and again with claim-ahead 1. This was the first run of the CLC path, the real
  TMA/mbarrier path and the hardware fp8/fp4 instructions. Timings: [B200 benchmarks](#b200-benchmarks).
- CuTeDSL backend: **deadlocks** on its first variant (`atomic`, fp16), with CuTeDSL 4.4.0 and
  4.8.0 alike. It compiles and launches in ~1.5 s, then the GPU spins at 100%.
  `hang_probe.py` read the progress counters during the hang (`results/hang_probe.txt`):
  layer 0's QKV finished, attention finished 45 of 64 tiles although all their inputs were
  ready, and O-proj onward never started, while some QKV tile counters went past their full
  count. **Root cause, fixed since:** the ring position (`cons`) of the load and compute loops
  was advanced inside the helper `fetch_work`. CuTeDSL carries a value into the next iteration
  of a dynamic loop only if the loop body assigns it or calls a method on it, so `cons` was
  never carried: every iteration re-read the same ring slot. Tickets were duplicated (the
  extra QKV counts) or dropped (the missing attention tiles), and the load warp and compute
  warps disagreed on tiles until the compute warps waited for weights that never came. The
  loops now advance `cons` themselves; the preprocessed code confirms both loops carry it.

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
