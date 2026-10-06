"""CuTeDSL port of the dynamic-persistent Qwen3-0.6B megakernel (target: B200, sm_100a).

Same task graph, tiling, dependency counters and fp16 rounding points as
../cuda/qwen_dps_megakernel.cu — read that file's header for the design. What
this version takes directly from CUTLASS's CuTeDSL library:

  * cutlass.utils.ClcDynamicPersistentTileSchedulerParams — grid shape for the
    CLC launch (one CTA per tile);
  * cutlass.utils.ClcDynamicPersistentTileScheduler — decodes the 16-byte
    clusterlaunchcontrol.try_cancel response (work_tile_info_from_clc_response);
  * cute.arch.issue_clc_query / clc_response — the CLC instructions;
  * cutlass.pipeline.PipelineAsync — the ring that hands work tickets from the
    scheduler warp to the load and compute warps (the role PipelineClcFetchAsync
    plays in dense_gemm_persistent_dynamic.py; here the scheduler warp is the only
    reader of raw CLC responses, so those use a single tx-count mbarrier).

Weights can be fp16, MXF8 (e4m3 + e8m0 per 32) or NVF4 (e2m1 + e4m3 per 16 + fp32
per tensor), with the same stage geometry as the CUDA build.

Inline PTX is used where CuTeDSL 4.4 has no op: the 1-D cp.async.bulk (it exposes
TMA tensor copies only) and the dequantising dot products, which issue the same
F2FP unpack + fma.rn.f32.f16 (FHFMA) sequence as the CUDA kernel (sm_100+).

    CUTE_DSL_ARCH=sm_100a python qwen_dps_cutedsl.py     # compile-check every scheduler x weight format
"""

import os
import sys

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Boolean, Float16, Float32, Int32, Int64, Uint8, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import make_ptr

# model
H, I, NQ, NKV, HD, NL, VOCAB = 1024, 3072, 16, 8, 128, 28, 151936
QS = NQ * HD
EPS = 1e-6

# CTA layout (matches the CUDA build for B200)
RING_BYTES = 196608         # DPS_RING_BYTES
SSTAGES_BASE = 1            # DPS_SSTAGES: tiles a CTA may claim before starting them
COMPUTE_WARPS = 8
COMPUTE_THREADS = COMPUTE_WARPS * 32
LOAD_WARP, SCHED_WARP = 8, 9
BLOCK_THREADS = 320
GROUP_BAR_ID = 1
DRAIN_BATCH = 16

# work decomposition
QKV_ROWS = 16
QKV_TILES_GROUP = (4 * HD) // QKV_ROWS          # 32 tiles per KV group (2 q heads + k + v)
T_QKV = NKV * QKV_TILES_GROUP                   # 256
ATTN_CHUNK, ATTN_SPLITS = 256, 8
T_ATTN = NKV * ATTN_SPLITS                      # 64
O_ROWS, GU_ROWS, D_ROWS = 8, 8, 4
T_O, T_GU, T_D = H // O_ROWS, I // GU_ROWS, H // D_ROWS
T_LAYER = T_QKV + T_ATTN + T_O + T_GU + T_D
T_STEP = NL * T_LAYER
LM_ROWS, LM_CHUNK_ROWS = 64, 16
LM_CHUNKS = LM_ROWS // LM_CHUNK_ROWS
T_LM = VOCAB // LM_ROWS
MAX_SEQ_SUPPORTED = ATTN_CHUNK * ATTN_SPLITS
PH_QKV, PH_ATTN, PH_OPROJ, PH_GATEUP, PH_DOWN, PH_LM = range(6)

# Layer table: 25 int64 slots per layer, same as QwenDpsLayerWeights in cuda/qwen_dps.h:
# 4 fp16 norm pointers, then (data, scale, gscale bits) for each of the 7 matrices.
S_IN_NORM, S_Q_NORM, S_K_NORM, S_POST_NORM = range(4)
M_Q, M_K, M_V, M_O, M_GATE, M_UP, M_DOWN = range(7)
SLOTS = 25

# weight formats: bits per value, values per lane group, values per scale (0 = none)
FP16, FP8, FP4 = 0, 1, 2
FORMATS = {"fp16": FP16, "fp8": FP8, "fp4": FP4}
BITS = {FP16: 16, FP8: 8, FP4: 4}
GROUP = {FP16: 8, FP8: 16, FP4: 16}
SBLOCK = {FP16: 0, FP8: 32, FP4: 16}


def row_bytes(f, k):
    return k * BITS[f] // 8


def scale_bytes(f, k):
    return k // SBLOCK[f] if SBLOCK[f] else 0


def lane_groups(f, k):
    return k // (32 * GROUP[f])


def stage_bytes(f):
    tile = lambda rows, k: rows * (row_bytes(f, k) + scale_bytes(f, k))  # noqa: E731
    b = max(tile(QKV_ROWS, H), tile(O_ROWS, QS), tile(2 * GU_ROWS, H), tile(D_ROWS, I), tile(LM_CHUNK_ROWS, H))
    return (b + 127) // 128 * 128


def weight_stages(f):
    return min(max(RING_BYTES // stage_bytes(f), 1), 16)


def sched_stages(f):
    # Same claim-ahead for every format: on B200, claiming deeper spread tiles unevenly.
    return min(max(SSTAGES_BASE, 1), 12)

# global counters (int32, 32 bytes apart)
C_TICKET = 0
C_QKV = 1
C_ASPLIT = C_QKV + NKV
C_ATTN = C_ASPLIT + NKV
C_OPROJ, C_GATEUP, C_DOWN, C_LM_ARRIVE, C_LM_DONE, C_EOS = range(C_ATTN + 1, C_ATTN + 7)
C_NUM = C_EOS + 1
CSTRIDE = 8


def _align256(x):
    return (x + 255) & ~255


def _workspace_layout():
    sizes = [
        ("sync", 4 * C_NUM * CSTRIDE),
        ("hidden", 2 * H),
        ("h1", 2 * H),
        ("qkv", 2 * (QS + 2 * NKV * HD)),      # q | k | v
        ("attn_out", 2 * QS),
        ("act", 2 * I),
        ("attn_part", 4 * NKV * ATTN_SPLITS * 2 * (HD + 2)),
        ("lm_val", 4 * T_LM),
        ("lm_idx", 4 * T_LM),
    ]
    off, layout = 0, {}
    for name, size in sizes:
        layout[name] = off
        off += _align256(size)
    return layout, off


WS, WS_BYTES = _workspace_layout()
SYNC_BYTES = 4 * C_NUM * CSTRIDE


# inline PTX
def _asm(ret, operands, text, constraints, loc, ip):
    return llvm.inline_asm(ret, operands, text, constraints, has_side_effects=True,
                           is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):
    _asm(None, [Int32(ns).ir_value(loc=loc, ip=ip)], "nanosleep.u32 $0;", "r", loc, ip)


@dsl_user_op
def l2_evict_first_policy(*, loc=None, ip=None):
    return Int64(_asm(T.i64(), [], "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l", loc, ip))


@dsl_user_op
def bulk_g2s(dst_smem, src_gmem, nbytes, mbar_smem, policy, *, loc=None, ip=None):
    """1-D TMA copy global -> shared that completes `nbytes` of tx on `mbar_smem`."""
    _asm(None,
         [Int32(dst_smem).ir_value(loc=loc, ip=ip), Int64(src_gmem).ir_value(loc=loc, ip=ip),
          Int32(nbytes).ir_value(loc=loc, ip=ip), Int32(mbar_smem).ir_value(loc=loc, ip=ip),
          Int64(policy).ir_value(loc=loc, ip=ip)],
         "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
         " [$0], [$1], $2, [$3], $4;",
         "r,l,r,r,l", loc, ip)


def _asm_pure(ret, operands, text, constraints, loc, ip):
    return llvm.inline_asm(ret, operands, text, constraints, has_side_effects=False,
                           is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def fp8x4_dot(acc, w, x0, x1, *, loc=None, ip=None):
    """acc + sum of 4 e4m3 weights (one word) times 4 fp16 activations (two words)."""
    return Float32(_asm_pure(
        T.f32(),
        [Float32(acc).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip),
         Int32(x0).ir_value(loc=loc, ip=ip), Int32(x1).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 a0, a1, w0, w1, w2, w3, xa, xb, xc, xd;\n.reg .b32 h0, h1;\n.reg .f32 t;\n"
        "mov.b32 {a0, a1}, $2;\n"
        "cvt.rn.f16x2.e4m3x2 h0, a0;\ncvt.rn.f16x2.e4m3x2 h1, a1;\n"
        "mov.b32 {w0, w1}, h0;\nmov.b32 {w2, w3}, h1;\n"
        "mov.b32 {xa, xb}, $3;\nmov.b32 {xc, xd}, $4;\n"
        "fma.rn.f32.f16 t, w0, xa, $1;\nfma.rn.f32.f16 t, w1, xb, t;\n"
        "fma.rn.f32.f16 t, w2, xc, t;\nfma.rn.f32.f16 $0, w3, xd, t;\n}",
        "=f,f,r,r,r", loc, ip))


@dsl_user_op
def fp4x8_dot(acc, w, x0, x1, x2, x3, *, loc=None, ip=None):
    """acc + sum of 8 e2m1 weights (one word, low nibble first) times 8 fp16 activations."""
    return Float32(_asm_pure(
        T.f32(),
        [Float32(acc).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip)]
        + [Int32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1, x2, x3)],
        "{\n.reg .b8 c0, c1, c2, c3;\n.reg .b32 h0, h1, h2, h3;\n"
        ".reg .b16 e0, e1, e2, e3, e4, e5, e6, e7, xa, xb, xc, xd, xe, xf, xg, xh;\n.reg .f32 t;\n"
        "mov.b32 {c0, c1, c2, c3}, $2;\n"
        "cvt.rn.f16x2.e2m1x2 h0, c0;\ncvt.rn.f16x2.e2m1x2 h1, c1;\n"
        "cvt.rn.f16x2.e2m1x2 h2, c2;\ncvt.rn.f16x2.e2m1x2 h3, c3;\n"
        "mov.b32 {e0, e1}, h0;\nmov.b32 {e2, e3}, h1;\nmov.b32 {e4, e5}, h2;\nmov.b32 {e6, e7}, h3;\n"
        "mov.b32 {xa, xb}, $3;\nmov.b32 {xc, xd}, $4;\nmov.b32 {xe, xf}, $5;\nmov.b32 {xg, xh}, $6;\n"
        "fma.rn.f32.f16 t, e0, xa, $1;\nfma.rn.f32.f16 t, e1, xb, t;\n"
        "fma.rn.f32.f16 t, e2, xc, t;\nfma.rn.f32.f16 t, e3, xd, t;\n"
        "fma.rn.f32.f16 t, e4, xe, t;\nfma.rn.f32.f16 t, e5, xf, t;\n"
        "fma.rn.f32.f16 t, e6, xg, t;\nfma.rn.f32.f16 $0, e7, xh, t;\n}",
        "=f,f,r,r,r,r,r", loc, ip))


@dsl_user_op
def f32_from_bits(bits, *, loc=None, ip=None):
    return Float32(llvm.bitcast(T.f32(), Int32(bits).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def e8m0_to_f32(u8, *, loc=None, ip=None):
    return f32_from_bits(Int32(u8) << 23, loc=loc, ip=ip)


@dsl_user_op
def e4m3_to_f32(u8, *, loc=None, ip=None):
    return Float32(_asm_pure(
        T.f32(), [Int32(u8).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 in16, z, lo, hi;\n.reg .b32 h2;\nmov.b32 {in16, z}, $1;\n"
        "cvt.rn.f16x2.e4m3x2 h2, in16;\nmov.b32 {lo, hi}, h2;\ncvt.f32.f16 $0, lo;\n}",
        "=f,r", loc, ip))


# trace-time helpers (no dynamic control flow)
def warp_sum(v):
    for off in (16, 8, 4, 2, 1):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=off)
    return v


def group_bar():
    cute.arch.barrier(barrier_id=GROUP_BAR_ID, number_of_threads=COMPUTE_THREADS)


def f16r(x):
    """Round through fp16, as HF does between ops."""
    return Float32(Float16(x))


def ring_next(idx, phase, n, stages):
    s = idx + n
    return s % stages, phase ^ ((s // stages) % 2)


def gptr(dtype, addr, align=16):
    return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


def norm_ptr(c, layer, slot):
    return gptr(Float16, c.layer_tab[layer * SLOTS + slot])


def mat_slot(layer, m, which):
    """which: 0 data pointer, 1 scale pointer, 2 gscale bits."""
    return layer * SLOTS + 4 + 3 * m + which


def mat_gscale(c, layer, m):
    return f32_from_bits(Int32(c.layer_tab[mat_slot(layer, m, 2)]))


def fetch_work(c, work_pipe, cons):
    """consumer_wait + read ticket + consumer_release (cf. get_current_work).

    The caller advances `cons` itself, in the loop body: the DSL carries a value into
    the next iteration of a dynamic loop only if the body assigns it or calls a method
    on it, and an object mutated inside a helper is neither. With the advance in here,
    every iteration re-read the same ring slot, which duplicated and dropped tickets
    and let the load and compute warps disagree on tiles until the kernel deadlocked."""
    work_pipe.consumer_wait(cons)
    t = c.work[cons.index]
    work_pipe.consumer_release(cons)
    return t


class KernelContext:
    """Per-CTA handles shared by the role loops.

    The DSL would carry a SimpleNamespace's fields through every control-flow
    region and rebuild them, leaking values between sibling branches. Fields here
    are set once at kernel entry and only read afterwards (stores go through the
    tensors they hold), so the object exposes no IR values and rebuilds as itself.
    """

    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        return self


def signal(c, counter):
    cute.arch.atomic_add(c.sync + counter * CSTRIDE, Int32(1), sem="release", scope="gpu")


class QwenDpsKernel:
    """One launch = prompt prefill + max_new greedy decode steps."""

    def __init__(self, use_clc: bool, weight_format: int = FP16):
        self.use_clc = use_clc
        self.fmt = weight_format
        self.SB = stage_bytes(weight_format)
        self.W = weight_stages(weight_format)
        self.SST = sched_stages(weight_format)

    # host entry
    @cute.jit
    def __call__(self, layer_tab: cute.Pointer, embed: cute.Pointer, final_norm: cute.Pointer,
                 lm_data: cute.Pointer, lm_scale: cute.Pointer, lm_gscale: Float32,
                 cos_t: cute.Pointer, sin_t: cute.Pointer,
                 k_cache: cute.Pointer, v_cache: cute.Pointer, ws: cute.Pointer,
                 tokens: cute.Pointer, out_log: cute.Pointer,
                 n_pre: Int32, total_tiles: Int32, start_pos: Int32, max_seq: Int32,
                 eos_token: Int32, attn_scale: Float32, grid_ctas: Int32, stream: cuda.CUstream):
        # CLC: the grid is the tile space, exactly like the CUTLASS GEMM example.
        sched_params = utils.ClcDynamicPersistentTileSchedulerParams((total_tiles, 1, 1), (1, 1, 1))
        grid = (grid_ctas, 1, 1)
        if const_expr(self.use_clc):
            grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(sched_params)
        self.kernel(layer_tab, embed, final_norm, lm_data, lm_scale, lm_gscale, cos_t, sin_t, k_cache,
                    v_cache, ws, tokens, out_log, n_pre, total_tiles, start_pos, max_seq, eos_token,
                    attn_scale, sched_params).launch(grid=grid, block=[BLOCK_THREADS, 1, 1], stream=stream)

    # kernel
    @cute.kernel
    def kernel(self, layer_tab: cute.Pointer, embed: cute.Pointer, final_norm: cute.Pointer,
               lm_data: cute.Pointer, lm_scale: cute.Pointer, lm_gscale: Float32,
               cos_t: cute.Pointer, sin_t: cute.Pointer,
               k_cache: cute.Pointer, v_cache: cute.Pointer, ws: cute.Pointer,
               tokens: cute.Pointer, out_log: cute.Pointer,
               n_pre: Int32, total_tiles: Int32, start_pos: Int32, max_seq: Int32,
               eos_token: Int32, attn_scale: Float32,
               sched_params: utils.ClcDynamicPersistentTileSchedulerParams):
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        W, SST = self.W, self.SST

        @cute.struct
        class Storage:
            clc_resp: cute.struct.Align[cute.struct.MemRange[Int32, 4 * DRAIN_BATCH], 16]
            work_bars: cute.struct.MemRange[cutlass.Int64, 2 * SST]
            clc_bar: cute.struct.MemRange[cutlass.Int64, 1]
            w_full: cute.struct.MemRange[cutlass.Int64, W]
            w_empty: cute.struct.MemRange[cutlass.Int64, W]
            work: cute.struct.MemRange[Int32, SST]
            dep_seen: cute.struct.MemRange[Int32, C_NUM]
            misc: cute.struct.MemRange[Int32, 4]          # [0] input tag, [1] hand-off flag
            red: cute.struct.MemRange[Float32, COMPUTE_WARPS]
            best_v: cute.struct.MemRange[Float32, COMPUTE_WARPS]
            best_i: cute.struct.MemRange[Int32, COMPUTE_WARPS]
            wm: cute.struct.MemRange[Float32, 2 * COMPUTE_WARPS]
            wl: cute.struct.MemRange[Float32, 2 * COMPUTE_WARPS]

        smem = utils.SmemAllocator()
        st = smem.allocate(Storage)
        wbuf = smem.allocate_tensor(Uint8, cute.make_layout(self.SB * W), byte_alignment=128)
        xs = smem.allocate_tensor(Float16, cute.make_layout(I), byte_alignment=16)
        qs = smem.allocate_tensor(Float32, cute.make_layout(2 * HD), byte_alignment=16)
        wacc = smem.allocate_tensor(Float32, cute.make_layout(COMPUTE_WARPS * 2 * HD), byte_alignment=16)

        # Everything the role loops need, gathered once.
        c = KernelContext()
        c.tid, c.warp, c.lane = tidx, warp, lane
        c.layer_tab = cute.make_tensor(layer_tab, cute.make_layout(NL * SLOTS))
        c.embed, c.final_norm, c.cos_t, c.sin_t = embed, final_norm, cos_t, sin_t
        c.lm_data, c.lm_scale, c.lm_gscale = lm_data, lm_scale, lm_gscale
        c.k_cache, c.v_cache = k_cache, v_cache
        c.tokens = cute.make_tensor(tokens, cute.make_layout(1 << 20))
        c.out_log = cute.make_tensor(out_log, cute.make_layout(1 << 20))
        c.sync = cute.recast_ptr(ws + WS["sync"], dtype=Int32)
        c.hidden = cute.recast_ptr(ws + WS["hidden"], dtype=Float16)
        c.h1 = cute.recast_ptr(ws + WS["h1"], dtype=Float16)
        c.qkv = cute.recast_ptr(ws + WS["qkv"], dtype=Float16)
        c.attn_out = cute.recast_ptr(ws + WS["attn_out"], dtype=Float16)
        c.act = cute.recast_ptr(ws + WS["act"], dtype=Float16)
        c.attn_part = cute.recast_ptr(ws + WS["attn_part"], dtype=Float32)
        c.lm_val = cute.recast_ptr(ws + WS["lm_val"], dtype=Float32)
        c.lm_idx = cute.recast_ptr(ws + WS["lm_idx"], dtype=Int32)
        c.n_pre, c.total_tiles, c.start_pos, c.max_seq = n_pre, total_tiles, start_pos, max_seq
        c.eos_token, c.attn_scale = eos_token, attn_scale
        c.clc_resp = st.clc_resp.data_ptr()
        c.clc_bar = st.clc_bar.data_ptr()
        c.w_full, c.w_empty = st.w_full.data_ptr(), st.w_empty.data_ptr()
        c.work = st.work.get_tensor(cute.make_layout(SST))
        c.dep_seen = st.dep_seen.get_tensor(cute.make_layout(C_NUM))
        c.misc = st.misc.get_tensor(cute.make_layout(4))
        c.red = st.red.get_tensor(cute.make_layout(COMPUTE_WARPS))
        c.best_v = st.best_v.get_tensor(cute.make_layout(COMPUTE_WARPS))
        c.best_i = st.best_i.get_tensor(cute.make_layout(COMPUTE_WARPS))
        c.wm = st.wm.get_tensor(cute.make_layout(2 * COMPUTE_WARPS))
        c.wl = st.wl.get_tensor(cute.make_layout(2 * COMPUTE_WARPS))
        c.wbuf, c.xs, c.qs, c.wacc = wbuf, xs, qs, wacc

        if tidx == 0:
            for i in cutlass.range_constexpr(W):
                cute.arch.mbarrier_init(c.w_full + i, 1)
                cute.arch.mbarrier_init(c.w_empty + i, COMPUTE_WARPS)
            cute.arch.mbarrier_init(c.clc_bar, 1)
            c.misc[0] = Int32(-1)
            for i in cutlass.range_constexpr(C_NUM):
                c.dep_seen[i] = Int32(0)
        # Scheduler warp (32 producers) -> load warp + compute warps (288 consumers).
        work_pipe = pipeline.PipelineAsync.create(
            num_stages=SST,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 + COMPUTE_THREADS),
            barrier_storage=st.work_bars.data_ptr(),
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if warp == SCHED_WARP:
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                sched_params, cute.arch.block_idx(), cute.arch.grid_dim(), c.clc_resp)
            self.sched_loop(c, work_pipe, tile_sched)
        elif warp == LOAD_WARP:
            self.load_loop(c, work_pipe)
        else:
            self.compute_loop(c, work_pipe)

    # scheduler warp
    @cute.jit
    def take_ticket(self, c):
        stop = Int32(0)
        v = Int32(0)
        if c.lane == 0:
            stop = cute.arch.load(c.sync + C_EOS * CSTRIDE, Int32, sem="relaxed", scope="gpu")
            if stop == 0:
                v = cute.arch.atomic_add(c.sync + C_TICKET * CSTRIDE, Int32(1), sem="relaxed", scope="gpu")
        stop = cute.arch.shuffle_sync(stop, 0)
        v = cute.arch.shuffle_sync(v, 0)
        t = Int32(-1)
        if stop == 0 and v < c.total_tiles:
            t = v
        return t

    @cute.jit
    def clc_fetch(self, c, tile_sched, phase):
        """advance_to_next_work + get_current_work for a single CLC query."""
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(c.clc_bar, 16)
            cute.arch.issue_clc_query(c.clc_bar, c.clc_resp)
        cute.arch.mbarrier_wait(c.clc_bar, phase)
        work = tile_sched.work_tile_info_from_clc_response(c.clc_resp)
        return Int32(work.is_valid_tile)

    @cute.jit
    def clc_drain(self, c, phase):
        """After EOS, cancel the rest of the grid in batches instead of letting
        every remaining CTA launch (one ~200 KB CTA per SM at a time)."""
        more = Boolean(True)
        ph = phase
        while more:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(c.clc_bar, 16 * DRAIN_BATCH)
                for i in cutlass.range_constexpr(DRAIN_BATCH):
                    cute.arch.issue_clc_query(c.clc_bar, c.clc_resp + 4 * i)
            cute.arch.mbarrier_wait(c.clc_bar, ph)
            ph = ph ^ 1
            ok = Int32(1)
            for i in cutlass.range_constexpr(DRAIN_BATCH):
                _, _, _, valid = cute.arch.clc_response(c.clc_resp + 4 * i)
                ok = ok & valid
            cute.arch.fence_proxy("async.shared", space="cta")
            if ok == 0:
                more = Boolean(False)
        return ph

    @cute.jit
    def sched_loop(self, c, work_pipe, tile_sched):
        prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.SST)
        # A ticket is claimed only once a ring slot is free for it (producer_acquire
        # first), so a CTA holds at most SST tickets it has not started.
        work_pipe.producer_acquire(prod)
        # initial_work_tile_info: the CTA's own launch is its first permission.
        t = self.take_ticket(c)
        clc_phase = Int32(0)
        keep = Boolean(True)
        while keep:
            if c.lane == 0:
                c.work[prod.index] = t
            work_pipe.producer_commit(prod)
            prod.advance()
            if t < 0:
                keep = Boolean(False)
            else:
                work_pipe.producer_acquire(prod)
                t = Int32(-1)
                stop = cute.arch.shuffle_sync(
                    cute.arch.load(c.sync + C_EOS * CSTRIDE, Int32, sem="relaxed", scope="gpu"), 0)
                if stop != 0:
                    if const_expr(self.use_clc):
                        clc_phase = self.clc_drain(c, clc_phase)
                else:
                    valid = Int32(1)
                    if const_expr(self.use_clc):
                        # advance_to_next_work: steal a CTA that has not launched yet.
                        valid = self.clc_fetch(c, tile_sched, clc_phase)
                        clc_phase = clc_phase ^ 1
                    if valid != 0:
                        t = self.take_ticket(c)

    # tile decode
    @cute.jit
    def decode_tile(self, c, t):
        pre = c.n_pre * T_STEP
        step = Int32(0)
        rem = Int32(0)
        if t < pre:
            step = t // T_STEP
            rem = t - step * T_STEP
        else:
            u = t - pre
            q = u // (T_STEP + T_LM)
            step = c.n_pre + q
            rem = u - q * (T_STEP + T_LM)
        layer = Int32(NL)
        phase = Int32(PH_LM)
        idx = rem - T_STEP
        if rem < T_STEP:
            layer = rem // T_LAYER
            x = rem - layer * T_LAYER
            if x < T_QKV:
                phase = Int32(PH_QKV)
                idx = x
            elif x < T_QKV + T_ATTN:
                phase = Int32(PH_ATTN)
                idx = x - T_QKV
            elif x < T_QKV + T_ATTN + T_O:
                phase = Int32(PH_OPROJ)
                idx = x - (T_QKV + T_ATTN)
            elif x < T_QKV + T_ATTN + T_O + T_GU:
                phase = Int32(PH_GATEUP)
                idx = x - (T_QKV + T_ATTN + T_O)
            else:
                phase = Int32(PH_DOWN)
                idx = x - (T_QKV + T_ATTN + T_O + T_GU)
        return step, layer, phase, idx

    @cute.jit
    def qkv_rows(self, idx):
        """-> (offset of the tile's first output in the q|k|v buffer, matrix index, first row)."""
        g = idx // QKV_TILES_GROUP
        j = idx % QKV_TILES_GROUP
        m = Int32(M_Q)
        row0 = g * 2 * HD + j * QKV_ROWS
        base = Int32(0)
        if j >= 24:
            m = Int32(M_V)
            row0 = g * HD + (j - 24) * QKV_ROWS
            base = Int32(QS + NKV * HD)
        elif j >= 16:
            m = Int32(M_K)
            row0 = g * HD + (j - 16) * QKV_ROWS
            base = Int32(QS)
        return base + row0, m, row0

    # load warp (TMA producer)
    @cute.jit
    def chunk_segs(self, c, layer, phase, idx, ch):
        """-> (src, dst, bytes) x 4 for chunk `ch` of a tile; unused segments have 0 bytes.
        A stage holds the chunk's weight rows followed by their scale rows."""
        f = self.fmt
        has_scales = SBLOCK[f] > 0
        rbH, sbH = row_bytes(f, H), scale_bytes(f, H)
        rbQ, sbQ = row_bytes(f, QS), scale_bytes(f, QS)
        rbI, sbI = row_bytes(f, I), scale_bytes(f, I)
        s0, s1, s2, s3 = Int64(0), Int64(0), Int64(0), Int64(0)
        d0, d1, d2, d3 = Int32(0), Int32(0), Int32(0), Int32(0)
        n0, n1, n2, n3 = Int32(0), Int32(0), Int32(0), Int32(0)
        if phase == PH_LM:
            row0 = idx * LM_ROWS + ch * LM_CHUNK_ROWS
            s0 = c.lm_data.toint() + Int64(row0) * rbH
            n0 = Int32(LM_CHUNK_ROWS * rbH)
            if const_expr(has_scales):
                s1 = c.lm_scale.toint() + Int64(row0) * sbH
                d1 = Int32(LM_CHUNK_ROWS * rbH)
                n1 = Int32(LM_CHUNK_ROWS * sbH)
        elif phase == PH_QKV:
            _, m, row0 = self.qkv_rows(idx)
            s0 = c.layer_tab[mat_slot(layer, m, 0)] + Int64(row0) * rbH
            n0 = Int32(QKV_ROWS * rbH)
            if const_expr(has_scales):
                s1 = c.layer_tab[mat_slot(layer, m, 1)] + Int64(row0) * sbH
                d1 = Int32(QKV_ROWS * rbH)
                n1 = Int32(QKV_ROWS * sbH)
        elif phase == PH_OPROJ:
            row0 = idx * O_ROWS
            s0 = c.layer_tab[mat_slot(layer, M_O, 0)] + Int64(row0) * rbQ
            n0 = Int32(O_ROWS * rbQ)
            if const_expr(has_scales):
                s1 = c.layer_tab[mat_slot(layer, M_O, 1)] + Int64(row0) * sbQ
                d1 = Int32(O_ROWS * rbQ)
                n1 = Int32(O_ROWS * sbQ)
        elif phase == PH_GATEUP:
            # [gate w | up w | gate scales | up scales]: rows 0..7 gate, 8..15 up
            row0 = idx * GU_ROWS
            s0 = c.layer_tab[mat_slot(layer, M_GATE, 0)] + Int64(row0) * rbH
            n0 = Int32(GU_ROWS * rbH)
            s1 = c.layer_tab[mat_slot(layer, M_UP, 0)] + Int64(row0) * rbH
            d1 = Int32(GU_ROWS * rbH)
            n1 = Int32(GU_ROWS * rbH)
            if const_expr(has_scales):
                s2 = c.layer_tab[mat_slot(layer, M_GATE, 1)] + Int64(row0) * sbH
                d2 = Int32(2 * GU_ROWS * rbH)
                n2 = Int32(GU_ROWS * sbH)
                s3 = c.layer_tab[mat_slot(layer, M_UP, 1)] + Int64(row0) * sbH
                d3 = Int32(2 * GU_ROWS * rbH + GU_ROWS * sbH)
                n3 = Int32(GU_ROWS * sbH)
        else:
            row0 = idx * D_ROWS
            s0 = c.layer_tab[mat_slot(layer, M_DOWN, 0)] + Int64(row0) * rbI
            n0 = Int32(D_ROWS * rbI)
            if const_expr(has_scales):
                s1 = c.layer_tab[mat_slot(layer, M_DOWN, 1)] + Int64(row0) * sbI
                d1 = Int32(D_ROWS * rbI)
                n1 = Int32(D_ROWS * sbI)
        return s0, d0, n0, s1, d1, n1, s2, d2, n2, s3, d3, n3

    @cute.jit
    def load_loop(self, c, work_pipe):
        cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.SST)
        ws_idx = Int32(0)
        ws_ph = Int32(1)   # producer starts on the "already empty" phase
        policy = l2_evict_first_policy()
        wbuf_addr = c.wbuf.iterator.toint()
        t = fetch_work(c, work_pipe, cons)
        cons.advance()
        while t >= 0:
            step, layer, phase, idx = self.decode_tile(c, t)
            nch = Int32(1)
            if phase == PH_ATTN:
                nch = Int32(0)
            if phase == PH_LM:
                nch = Int32(LM_CHUNKS)
            for ch in range(nch):
                s0, d0, n0, s1, d1, n1, s2, d2, n2, s3, d3, n3 = self.chunk_segs(c, layer, phase, idx, ch)
                cute.arch.mbarrier_wait(c.w_empty + ws_idx, ws_ph)
                full = c.w_full + ws_idx
                dst = wbuf_addr + ws_idx * self.SB
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full, n0 + n1 + n2 + n3)
                    bulk_g2s(dst + d0, s0, n0, full.toint(), policy)
                    if n1 > 0:
                        bulk_g2s(dst + d1, s1, n1, full.toint(), policy)
                    if n2 > 0:
                        bulk_g2s(dst + d2, s2, n2, full.toint(), policy)
                    if n3 > 0:
                        bulk_g2s(dst + d3, s3, n3, full.toint(), policy)
                ws_idx, ws_ph = ring_next(ws_idx, ws_ph, 1, self.W)
            t = fetch_work(c, work_pipe, cons)
            cons.advance()

    # compute warps
    @cute.jit
    def wait_counter(self, c, cid, target):
        if c.dep_seen[cid] < target:
            ptr = c.sync + cid * CSTRIDE
            v = cute.arch.load(ptr, Int32, sem="acquire", scope="gpu")
            while v < target:
                nanosleep(32)
                v = cute.arch.load(ptr, Int32, sem="acquire", scope="gpu")
            c.dep_seen[cid] = v

    @cute.jit
    def wait_deps(self, c, step, layer, phase, idx):
        inst = step * NL + layer
        if phase == PH_QKV:
            if layer > 0:
                self.wait_counter(c, C_DOWN, inst * T_D)
            else:
                if step > 0:
                    self.wait_counter(c, C_DOWN, inst * T_D)          # previous token done
                if step > c.n_pre:
                    self.wait_counter(c, C_LM_DONE, step - c.n_pre)   # its argmax is known
        elif phase == PH_ATTN:
            self.wait_counter(c, C_QKV + idx // ATTN_SPLITS, (inst + 1) * QKV_TILES_GROUP)
        elif phase == PH_OPROJ:
            self.wait_counter(c, C_ATTN, (inst + 1) * NKV)
        elif phase == PH_GATEUP:
            self.wait_counter(c, C_OPROJ, (inst + 1) * T_O)
        elif phase == PH_DOWN:
            self.wait_counter(c, C_GATEUP, (inst + 1) * T_GU)
        else:
            self.wait_counter(c, C_DOWN, (step * NL + NL) * T_D)

    @cute.jit
    def group_sum(self, c, v):
        v = warp_sum(v)
        if c.lane == 0:
            c.red[c.warp] = v
        group_bar()
        s = Float32(0.0)
        for w in cutlass.range_constexpr(COMPUTE_WARPS):
            s = s + c.red[w]
        group_bar()
        return s

    @cute.jit
    def prep_rmsnorm(self, c, src, weight):
        """xs = half(w * half(x * rsqrt(mean(x^2) + eps)))  (HF Qwen3RMSNorm in fp16)."""
        vec4 = cute.make_layout((4, COMPUTE_THREADS), stride=(1, 4))
        x = cute.make_tensor(src, vec4)[None, c.tid].load().to(Float32)
        ss = self.group_sum(c, (x * x).reduce(cute.ReductionOp.ADD, Float32(0.0), 0))
        rstd = Float32(cute.math.rsqrt(ss / Float32(H) + Float32(EPS)))
        w = cute.make_tensor(weight, vec4)[None, c.tid].load().to(Float32)
        y = (x * rstd).to(Float16).to(Float32)
        cute.make_tensor(c.xs.iterator, vec4)[None, c.tid].store((w * y).to(Float16))
        group_bar()

    @cute.jit
    def prep_copy(self, c, src, n: cutlass.Constexpr):
        vec8 = cute.make_layout((8, n // 8), stride=(1, 8))
        s = cute.make_tensor(src, vec8)
        d = cute.make_tensor(c.xs.iterator, vec8)
        for i in range(c.tid, n // 8, COMPUTE_THREADS):
            d[None, i].store(s[None, i].load())
        group_bar()

    @cute.jit
    def stage_release(self, c, s):
        cute.arch.sync_warp()
        if c.lane == 0:
            cute.arch.mbarrier_arrive(c.w_empty + s)

    def x_regs(self, c, k, khalf=None):
        """This lane's slice of xs, matching row_dot's groups (j*32 + lane): fp16 ->
        fp32 vectors of 8 values; fp8/fp4 -> 8 packed fp16x2 words (16 values)."""
        nj = lane_groups(self.fmt, k)
        if self.fmt == FP16:
            ptr, half_off = c.xs.iterator, I // 2
        else:
            ptr, half_off = cute.recast_ptr(c.xs.iterator, dtype=Int32), I // 4
        if khalf is None:
            xv = cute.make_tensor(ptr, cute.make_layout((8, nj, 32), stride=(1, 256, 8)))
            vals = [xv[None, j, c.lane].load() for j in range(nj)]
        else:
            xv = cute.make_tensor(ptr, cute.make_layout((8, nj, 32, 2), stride=(1, 256, 8, half_off)))
            vals = [xv[None, j, c.lane, khalf].load() for j in range(nj)]
        return [v.to(Float32) for v in vals] if self.fmt == FP16 else vals

    def wview(self, c, k, rows, split=False):
        """Weight-ring view (e, j, lane, [khalf,] row, stage) for rows of length k:
        fp16 in halves (8 per group), fp8/fp4 in 32-bit words (4 or 2 per group)."""
        f = self.fmt
        if f == FP16:
            ptr, e, unit = cute.recast_ptr(c.wbuf.iterator, dtype=Float16), 8, 2
        else:
            ptr, e, unit = cute.recast_ptr(c.wbuf.iterator, dtype=Int32), GROUP[f] * BITS[f] // 32, 4
        row, stage = row_bytes(f, k) // unit, self.SB // unit
        if split:
            nj = lane_groups(f, k // 2)
            return cute.make_tensor(ptr, cute.make_layout(
                (e, nj, 32, 2, rows, self.W), stride=(1, 32 * e, e, row // 2, row, stage)))
        nj = lane_groups(f, k)
        return cute.make_tensor(ptr, cute.make_layout((e, nj, 32, rows, self.W), stride=(1, 32 * e, e, row, stage)))

    def row_dot(self, c, wv, xr, r, s, scale_off, g0=0, khalf=None):
        """Warp-wide dot product of weight row r (stage s) with x, before the per-tensor
        scale. scale_off: byte offset of the row's block scales inside the stage."""
        acc = Float32(0.0)
        for j in range(len(xr)):
            w = (wv[None, j, c.lane, r, s] if khalf is None else wv[None, j, c.lane, khalf, r, s]).load()
            if self.fmt == FP16:
                acc = acc + (w.to(Float32) * xr[j]).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                continue
            g = g0 + j * 32 + c.lane
            part = Float32(0.0)
            if self.fmt == FP8:
                for t in range(4):
                    part = fp8x4_dot(part, w[t], xr[j][2 * t], xr[j][2 * t + 1])
                sc = e8m0_to_f32(c.wbuf[s * self.SB + scale_off + g // 2])   # 2 groups per 32-block
            else:
                for t in range(2):
                    part = fp4x8_dot(part, w[t], xr[j][4 * t], xr[j][4 * t + 1], xr[j][4 * t + 2], xr[j][4 * t + 3])
                sc = e4m3_to_f32(c.wbuf[s * self.SB + scale_off + g])        # 1 group per 16-block
            acc = acc + part * sc
        return warp_sum(acc)

    @cute.jit
    def tile_qkv(self, c, step, layer, idx, s, ph):
        rb, sb = row_bytes(self.fmt, H), scale_bytes(self.fmt, H)
        tag = (step * (NL + 1) + layer) * 8 + PH_QKV
        if c.misc[0] != tag:
            addr = c.hidden.toint()
            if layer == 0:
                addr = c.embed.toint() + Int64(c.tokens[step]) * (H * 2)
            self.prep_rmsnorm(c, gptr(Float16, addr), norm_ptr(c, layer, S_IN_NORM))
            if c.tid == 0:
                c.misc[0] = tag
        out_off, m, _ = self.qkv_rows(idx)
        gs = mat_gscale(c, layer, m)
        out = cute.make_tensor(c.qkv, cute.make_layout(QS + 2 * NKV * HD))
        xr = self.x_regs(c, H)
        wv = self.wview(c, H, QKV_ROWS)
        cute.arch.mbarrier_wait(c.w_full + s, ph)
        for rr in cutlass.range_constexpr(QKV_ROWS // COMPUTE_WARPS):
            r = c.warp + rr * COMPUTE_WARPS
            d = self.row_dot(c, wv, xr, r, s, QKV_ROWS * rb + r * sb)
            if const_expr(self.fmt != FP16):
                d = d * gs
            if c.lane == 0:
                out[out_off + r] = Float16(d)
        self.stage_release(c, s)
        group_bar()
        if c.tid == 0:
            signal(c, C_QKV + idx // QKV_TILES_GROUP)

    def head_norm_rope(self, c, src, norm_w, pos):
        """RMSNorm over one head + RoPE (HF fp16 rounding). Lane owns dims lane + 32*i."""
        x = cute.make_tensor(src, cute.make_layout(HD))
        w = cute.make_tensor(norm_w, cute.make_layout(HD))
        cs = cute.make_tensor(c.cos_t, cute.make_layout(MAX_SEQ_SUPPORTED * HD))
        sn = cute.make_tensor(c.sin_t, cute.make_layout(MAX_SEQ_SUPPORTED * HD))
        x0, x1, x2, x3 = (Float32(x[c.lane + 32 * i]) for i in range(4))
        ss = warp_sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3)
        rstd = Float32(cute.math.rsqrt(ss / Float32(HD) + Float32(EPS)))
        n = [f16r(Float32(w[c.lane + 32 * i]) * f16r(xi * rstd)) for i, xi in enumerate((x0, x1, x2, x3))]
        rot = [-n[2], -n[3], n[0], n[1]]   # rotate_half partner of dim lane + 32*i
        out = []
        for i in range(4):
            d = pos * HD + c.lane + 32 * i
            out.append(f16r(f16r(n[i] * Float32(cs[d])) + f16r(rot[i] * Float32(sn[d]))))
        return out[0], out[1], out[2], out[3]

    @cute.jit
    def tile_attention(self, c, step, layer, idx):
        h = idx // ATTN_SPLITS
        sp = idx % ATTN_SPLITS
        pos = c.start_pos + step
        length = pos + 1
        t0 = sp * ATTN_CHUNK
        t1 = min(t0 + ATTN_CHUNK, length)
        head_off = cute.assume((layer * NKV + h) * c.max_seq * HD, divby=HD)
        kc = c.k_cache + head_off
        vc = c.v_cache + head_off
        qkv = c.qkv
        part_t = cute.make_tensor(c.attn_part, cute.make_layout(NKV * ATTN_SPLITS * 2 * (HD + 2)))

        if t0 < length:
            # q-norm + RoPE for the two query heads of KV head h; the split owning
            # the new position also appends k (normed + roped) and v to the cache.
            if c.warp < 2:
                o0, o1, o2, o3 = self.head_norm_rope(c, qkv + (2 * h + c.warp) * HD,
                                                     norm_ptr(c, layer, S_Q_NORM), pos)
                for i, o in enumerate((o0, o1, o2, o3)):
                    c.qs[c.warp * HD + c.lane + 32 * i] = o
            elif c.warp == 2 and pos < t1:
                o0, o1, o2, o3 = self.head_norm_rope(c, qkv + QS + h * HD,
                                                     norm_ptr(c, layer, S_K_NORM), pos)
                kt = cute.make_tensor(kc, cute.make_layout(MAX_SEQ_SUPPORTED * HD))
                for i, o in enumerate((o0, o1, o2, o3)):
                    kt[pos * HD + c.lane + 32 * i] = Float16(o)
            elif c.warp == 3 and pos < t1:
                vec4 = cute.make_layout((4, 32), stride=(1, 4))
                vsrc = cute.make_tensor(qkv + QS + NKV * HD + h * HD, vec4)
                vdst = cute.make_tensor(vc + cute.assume(pos * HD, divby=HD), vec4)
                vdst[None, c.lane].store(vsrc[None, c.lane].load())
            group_bar()

            # Per-warp online softmax; lane owns dims 4*lane .. 4*lane+3.
            qv = cute.make_tensor(c.qs.iterator, cute.make_layout((4, 32, 2), stride=(1, 4, HD)))
            qa = qv[None, c.lane, 0].load()
            qb = qv[None, c.lane, 1].load()
            kv_layout = cute.make_layout((4, 32, MAX_SEQ_SUPPORTED), stride=(1, 4, HD))
            kt = cute.make_tensor(kc, kv_layout)
            vt = cute.make_tensor(vc, kv_layout)
            neg_inf = Float32(float("-inf"))
            m0, m1, l0, l1 = neg_inf, neg_inf, Float32(0.0), Float32(0.0)
            a00, a01, a02, a03 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            a10, a11, a12, a13 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            t = t0 + c.warp
            while t < t1:
                k4 = kt[None, c.lane, t].load().to(Float32)
                v4 = vt[None, c.lane, t].load().to(Float32)
                s0 = warp_sum((qa * k4).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)) * c.attn_scale
                s1 = warp_sum((qb * k4).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)) * c.attn_scale
                n0 = max(m0, s0)
                eo0 = Float32(cute.math.exp(m0 - n0, fastmath=True))
                e0 = Float32(cute.math.exp(s0 - n0, fastmath=True))
                n1 = max(m1, s1)
                eo1 = Float32(cute.math.exp(m1 - n1, fastmath=True))
                e1 = Float32(cute.math.exp(s1 - n1, fastmath=True))
                l0 = l0 * eo0 + e0
                l1 = l1 * eo1 + e1
                a00, a01, a02, a03 = (a00 * eo0 + e0 * v4[0], a01 * eo0 + e0 * v4[1],
                                      a02 * eo0 + e0 * v4[2], a03 * eo0 + e0 * v4[3])
                a10, a11, a12, a13 = (a10 * eo1 + e1 * v4[0], a11 * eo1 + e1 * v4[1],
                                      a12 * eo1 + e1 * v4[2], a13 * eo1 + e1 * v4[3])
                m0, m1 = n0, n1
                t = t + COMPUTE_WARPS
            base0 = (c.warp * 2) * HD + 4 * c.lane
            base1 = (c.warp * 2 + 1) * HD + 4 * c.lane
            for i, a in enumerate((a00, a01, a02, a03)):
                c.wacc[base0 + i] = a
            for i, a in enumerate((a10, a11, a12, a13)):
                c.wacc[base1 + i] = a
            if c.lane == 0:
                c.wm[c.warp * 2] = m0
                c.wm[c.warp * 2 + 1] = m1
                c.wl[c.warp * 2] = l0
                c.wl[c.warp * 2 + 1] = l1
            group_bar()

            # Merge the 8 warps; write this split's (max, sum, acc[128]) per head.
            j = c.tid // HD
            d = c.tid % HD
            mx = neg_inf
            for w in cutlass.range_constexpr(COMPUTE_WARPS):
                mx = max(mx, c.wm[w * 2 + j])
            lsum = Float32(0.0)
            acc = Float32(0.0)
            for w in cutlass.range_constexpr(COMPUTE_WARPS):
                e = Float32(0.0)
                if c.wm[w * 2 + j] != neg_inf:
                    e = Float32(cute.math.exp(c.wm[w * 2 + j] - mx, fastmath=True))
                lsum = lsum + c.wl[w * 2 + j] * e
                acc = acc + c.wacc[(w * 2 + j) * HD + d] * e
            pbase = ((h * ATTN_SPLITS + sp) * 2 + j) * (HD + 2)
            part_t[pbase + 2 + d] = acc
            if d == 0:
                part_t[pbase] = mx
                part_t[pbase + 1] = lsum

        # The last split to finish for this KV head combines all splits.
        group_bar()
        if c.tid == 0:
            cute.arch.fence_acq_rel_gpu()
            old = cute.arch.atomic_add(c.sync + (C_ASPLIT + h) * CSTRIDE, Int32(1), sem="relaxed", scope="gpu")
            last = Int32(0)
            if (old + 1) % ATTN_SPLITS == 0:
                last = Int32(1)
                cute.arch.fence_acq_rel_gpu()
            c.misc[1] = last
        group_bar()
        if c.misc[1] != 0:
            n_act = (length + ATTN_CHUNK - 1) // ATTN_CHUNK
            j = c.tid // HD
            d = c.tid % HD
            mx = Float32(float("-inf"))
            for s in range(n_act):
                pb = ((h * ATTN_SPLITS + s) * 2 + j) * (HD + 2)
                mx = max(mx, cute.arch.load(c.attn_part + pb, Float32, cop="cg"))
            lsum = Float32(0.0)
            acc = Float32(0.0)
            for s in range(n_act):
                pb = ((h * ATTN_SPLITS + s) * 2 + j) * (HD + 2)
                e = Float32(cute.math.exp(cute.arch.load(c.attn_part + pb, Float32, cop="cg") - mx, fastmath=True))
                lsum = lsum + cute.arch.load(c.attn_part + pb + 1, Float32, cop="cg") * e
                acc = acc + cute.arch.load(c.attn_part + pb + 2 + d, Float32, cop="cg") * e
            out = cute.make_tensor(c.attn_out, cute.make_layout(QS))
            out[(2 * h + j) * HD + d] = Float16(acc / lsum)
            group_bar()
            if c.tid == 0:
                signal(c, C_ATTN)

    @cute.jit
    def tile_oproj(self, c, step, layer, idx, s, ph):
        rb, sb = row_bytes(self.fmt, QS), scale_bytes(self.fmt, QS)
        tag = (step * (NL + 1) + layer) * 8 + PH_OPROJ
        if c.misc[0] != tag:
            self.prep_copy(c, c.attn_out, QS)
            if c.tid == 0:
                c.misc[0] = tag
        gs = mat_gscale(c, layer, M_O)
        xr = self.x_regs(c, QS)
        wv = self.wview(c, QS, O_ROWS)
        cute.arch.mbarrier_wait(c.w_full + s, ph)
        d = self.row_dot(c, wv, xr, c.warp, s, O_ROWS * rb + c.warp * sb)
        if const_expr(self.fmt != FP16):
            d = d * gs
        if c.lane == 0:
            row = idx * O_ROWS + c.warp
            res_addr = c.hidden.toint()
            if layer == 0:
                res_addr = c.embed.toint() + Int64(c.tokens[step]) * (H * 2)
            res = cute.make_tensor(gptr(Float16, res_addr), cute.make_layout(H))
            h1 = cute.make_tensor(c.h1, cute.make_layout(H))
            h1[row] = Float16(Float32(res[row]) + f16r(d))
        self.stage_release(c, s)
        group_bar()
        if c.tid == 0:
            signal(c, C_OPROJ)

    @cute.jit
    def tile_gateup(self, c, step, layer, idx, s, ph):
        rb, sb = row_bytes(self.fmt, H), scale_bytes(self.fmt, H)
        tag = (step * (NL + 1) + layer) * 8 + PH_GATEUP
        if c.misc[0] != tag:
            self.prep_rmsnorm(c, c.h1, norm_ptr(c, layer, S_POST_NORM))
            if c.tid == 0:
                c.misc[0] = tag
        gs_g = mat_gscale(c, layer, M_GATE)
        gs_u = mat_gscale(c, layer, M_UP)
        xr = self.x_regs(c, H)
        wv = self.wview(c, H, 2 * GU_ROWS)   # rows 0..7 gate, 8..15 up
        sc_off = 2 * GU_ROWS * rb
        cute.arch.mbarrier_wait(c.w_full + s, ph)
        g = self.row_dot(c, wv, xr, c.warp, s, sc_off + c.warp * sb)
        u = self.row_dot(c, wv, xr, GU_ROWS + c.warp, s, sc_off + (GU_ROWS + c.warp) * sb)
        if const_expr(self.fmt != FP16):
            g = g * gs_g
            u = u * gs_u
        g = f16r(g)
        u = f16r(u)
        if c.lane == 0:
            silu = f16r(g / (Float32(1.0) + Float32(cute.math.exp(-g))))
            act = cute.make_tensor(c.act, cute.make_layout(I))
            act[idx * GU_ROWS + c.warp] = Float16(silu * u)
        self.stage_release(c, s)
        group_bar()
        if c.tid == 0:
            signal(c, C_GATEUP)

    @cute.jit
    def tile_down(self, c, step, layer, idx, s, ph):
        rb, sb = row_bytes(self.fmt, I), scale_bytes(self.fmt, I)
        tag = (step * (NL + 1) + layer) * 8 + PH_DOWN
        if c.misc[0] != tag:
            self.prep_copy(c, c.act, I)
            if c.tid == 0:
                c.misc[0] = tag
        gs = mat_gscale(c, layer, M_DOWN)
        # Two warps per row, each covering half of K = 3072.
        r = c.warp // 2
        khalf = c.warp % 2
        xr = self.x_regs(c, I // 2, khalf)
        wv = self.wview(c, I, D_ROWS, split=True)
        cute.arch.mbarrier_wait(c.w_full + s, ph)
        acc = self.row_dot(c, wv, xr, r, s, D_ROWS * rb + r * sb, khalf * lane_groups(self.fmt, I // 2) * 32, khalf)
        self.stage_release(c, s)
        if c.lane == 0:
            c.red[c.warp] = acc
        group_bar()
        if c.tid < D_ROWS:
            row = idx * D_ROWS + c.tid
            d = c.red[2 * c.tid] + c.red[2 * c.tid + 1]
            if const_expr(self.fmt != FP16):
                d = d * gs
            d = f16r(d)
            h1 = cute.make_tensor(c.h1, cute.make_layout(H))
            hid = cute.make_tensor(c.hidden, cute.make_layout(H))
            hid[row] = Float16(Float32(h1[row]) + d)
        group_bar()
        if c.tid == 0:
            signal(c, C_DOWN)

    @cute.jit
    def tile_lm(self, c, step, idx, s0, ph0):
        rb, sb = row_bytes(self.fmt, H), scale_bytes(self.fmt, H)
        tag = (step * (NL + 1) + NL) * 8 + PH_LM
        if c.misc[0] != tag:
            self.prep_rmsnorm(c, c.hidden, c.final_norm)
            if c.tid == 0:
                c.misc[0] = tag
        xr = self.x_regs(c, H)
        wv = self.wview(c, H, LM_CHUNK_ROWS)
        bv = Float32(float("-inf"))
        bi = Int32(0x7FFFFFFF)
        for ch in cutlass.range_constexpr(LM_CHUNKS):
            s, ph = ring_next(s0, ph0, ch, self.W)
            cute.arch.mbarrier_wait(c.w_full + s, ph)
            for rr in cutlass.range_constexpr(LM_CHUNK_ROWS // COMPUTE_WARPS):
                r = c.warp + rr * COMPUTE_WARPS
                logit = self.row_dot(c, wv, xr, r, s, LM_CHUNK_ROWS * rb + r * sb)
                if const_expr(self.fmt != FP16):
                    logit = logit * c.lm_gscale
                logit = f16r(logit)   # HF logits are fp16
                row = idx * LM_ROWS + ch * LM_CHUNK_ROWS + r
                if logit > bv or (logit == bv and row < bi):
                    bv = logit
                    bi = row
            self.stage_release(c, s)
        if c.lane == 0:
            c.best_v[c.warp] = bv
            c.best_i[c.warp] = bi
        group_bar()
        lm_val = cute.make_tensor(c.lm_val, cute.make_layout(T_LM))
        lm_idx = cute.make_tensor(c.lm_idx, cute.make_layout(T_LM))
        if c.tid == 0:
            for w in cutlass.range_constexpr(1, COMPUTE_WARPS):
                if c.best_v[w] > bv or (c.best_v[w] == bv and c.best_i[w] < bi):
                    bv = c.best_v[w]
                    bi = c.best_i[w]
            lm_val[idx] = bv
            lm_idx[idx] = bi
            cute.arch.fence_acq_rel_gpu()
            old = cute.arch.atomic_add(c.sync + C_LM_ARRIVE * CSTRIDE, Int32(1), sem="relaxed", scope="gpu")
            last = Int32(0)
            if (old + 1) % T_LM == 0:
                last = Int32(1)
                cute.arch.fence_acq_rel_gpu()
            c.misc[1] = last
        group_bar()
        if c.misc[1] != 0:
            # Last LM tile of the step: global argmax, feed the token back, maybe stop.
            bv = Float32(float("-inf"))
            bi = Int32(0x7FFFFFFF)
            for i in range(c.tid, T_LM, COMPUTE_THREADS):
                v = cute.arch.load(c.lm_val + i, Float32, cop="cg")
                k = cute.arch.load(c.lm_idx + i, Int32, cop="cg")
                if v > bv or (v == bv and k < bi):
                    bv = v
                    bi = k
            for off in (16, 8, 4, 2, 1):
                ov = cute.arch.shuffle_sync_bfly(bv, offset=off)
                oi = cute.arch.shuffle_sync_bfly(bi, offset=off)
                if ov > bv or (ov == bv and oi < bi):
                    bv = ov
                    bi = oi
            group_bar()
            if c.lane == 0:
                c.best_v[c.warp] = bv
                c.best_i[c.warp] = bi
            group_bar()
            if c.tid == 0:
                for w in cutlass.range_constexpr(COMPUTE_WARPS):
                    if c.best_v[w] > bv or (c.best_v[w] == bv and c.best_i[w] < bi):
                        bv = c.best_v[w]
                        bi = c.best_i[w]
                eos = c.sync + C_EOS * CSTRIDE
                if cute.arch.load(eos, Int32, sem="relaxed", scope="gpu") == 0:
                    c.tokens[step + 1] = bi
                    c.out_log[step - c.n_pre] = bi
                    if bi == c.eos_token:
                        cute.arch.atomic_exch(eos, Int32(1), sem="relaxed", scope="gpu")
                cute.arch.fence_acq_rel_gpu()
                cute.arch.atomic_add(c.sync + C_LM_DONE * CSTRIDE, Int32(1), sem="relaxed", scope="gpu")

    @cute.jit
    def compute_loop(self, c, work_pipe):
        cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.SST)
        ws_idx = Int32(0)
        ws_ph = Int32(0)
        t = fetch_work(c, work_pipe, cons)
        cons.advance()
        while t >= 0:
            step, layer, phase, idx = self.decode_tile(c, t)
            if c.tid == 0:
                self.wait_deps(c, step, layer, phase, idx)
            group_bar()
            nch = Int32(1)
            if phase == PH_QKV:
                self.tile_qkv(c, step, layer, idx, ws_idx, ws_ph)
            elif phase == PH_ATTN:
                self.tile_attention(c, step, layer, idx)
                nch = Int32(0)
            elif phase == PH_OPROJ:
                self.tile_oproj(c, step, layer, idx, ws_idx, ws_ph)
            elif phase == PH_GATEUP:
                self.tile_gateup(c, step, layer, idx, ws_idx, ws_ph)
            elif phase == PH_DOWN:
                self.tile_down(c, step, layer, idx, ws_idx, ws_ph)
            else:
                self.tile_lm(c, step, idx, ws_idx, ws_ph)
                nch = Int32(LM_CHUNKS)
            ws_idx, ws_ph = ring_next(ws_idx, ws_ph, nch, self.W)
            t = fetch_work(c, work_pipe, cons)
            cons.advance()


# host wrapper
def _gmem(dtype, t, align=16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


class CuteDslMegakernel:
    """Drop-in backend for qwen_dps.DpsDecoder (backend="cutedsl")."""

    def __init__(self, weights, layer_table, k_cache, v_cache, weight_format="fp16"):
        self.w = weights
        self.layer_table = layer_table
        self.k_cache, self.v_cache = k_cache, v_cache
        self.fmt = FORMATS[weight_format]
        self.ws = torch.zeros(WS_BYTES, dtype=torch.uint8, device="cuda")
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        self.num_sms = props.multi_processor_count
        self.clc_supported = props.major >= 10
        self._compiled = {}

    def info(self):
        return dict(num_sms=self.num_sms, clc_supported=self.clc_supported, real_mbarrier=True,
                    block_threads=BLOCK_THREADS, stage_bytes=stage_bytes(self.fmt),
                    weight_stages=weight_stages(self.fmt), sched_stages=sched_stages(self.fmt), ctas_per_sm=1,
                    tiles_per_step=T_STEP, lm_tiles=T_LM, max_seq_supported=MAX_SEQ_SUPPORTED,
                    trace_build=False, phase_tiles=[T_QKV, T_ATTN, T_O, T_GU, T_D])

    def _args(self, tokens, out_log, n_pre, total, start_pos, eos, grid, stream):
        w, lm = self.w, self.w["lm_head"]
        lm_scale = lm.data if lm.scale is None else lm.scale   # never read for fp16
        return (_gmem(Int64, self.layer_table), _gmem(Float16, w["embed"]), _gmem(Float16, w["final_norm"]),
                _gmem(Uint8, lm.data), _gmem(Uint8, lm_scale), Float32(lm.gscale),
                _gmem(Float16, w["cos"]), _gmem(Float16, w["sin"]),
                _gmem(Float16, self.k_cache), _gmem(Float16, self.v_cache), _gmem(Uint8, self.ws, 256),
                _gmem(Int32, tokens, 4), _gmem(Int32, out_log, 4),
                Int32(n_pre), Int32(total), Int32(start_pos), Int32(self.k_cache.shape[2]), Int32(eos),
                Float32(1.0 / HD ** 0.5), Int32(grid), stream)

    def _kernel(self, use_clc, args):
        if use_clc not in self._compiled:
            self._compiled[use_clc] = cute.compile(QwenDpsKernel(use_clc, self.fmt), *args)
        return self._compiled[use_clc]

    def generate(self, tokens, n_prompt, max_new, start_pos, eos, sched_mode, out_log):
        if sched_mode in (3, 4):
            raise ValueError("the oneshot and static schedulers are only implemented in the CUDA backend")
        use_clc = self.clc_supported if sched_mode == 0 else sched_mode == 2
        if use_clc and not self.clc_supported:
            raise RuntimeError("CLC needs an sm_100+ GPU")
        last_pos = start_pos + n_prompt + max_new - 2
        if last_pos >= min(MAX_SEQ_SUPPORTED, self.k_cache.shape[2]):
            raise ValueError("sequence exceeds the KV cache")
        n_pre = n_prompt - 1
        total = n_pre * T_STEP + max_new * (T_STEP + T_LM)
        grid = total if use_clc else self.num_sms
        self.ws[:SYNC_BYTES].zero_()
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = self._args(tokens, out_log, n_pre, total, start_pos, eos, grid, stream)
        self._kernel(use_clc, args)(*args)
        return (2 if use_clc else 1), grid


def compile_check(formats=("fp16", "fp8", "fp4")):
    """Compile every scheduler x weight-format variant for CUTE_DSL_ARCH (default sm_100a),
    without needing a GPU of that architecture."""
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
    from cuda.bindings.driver import CUstream

    def fake(dtype, align=16):
        return make_ptr(dtype, 1 << 20, cute.AddressSpace.gmem, assumed_align=align)

    for name in formats:
        for use_clc in (True, False):
            args = (fake(Int64), fake(Float16), fake(Float16), fake(Uint8), fake(Uint8), Float32(1.0),
                    fake(Float16), fake(Float16), fake(Float16), fake(Float16), fake(Uint8, 256),
                    fake(Int32, 4), fake(Int32, 4), Int32(1), Int32(T_STEP), Int32(0), Int32(2048), Int32(-1),
                    Float32(0.088), Int32(148), CUstream(0))
            cute.compile(QwenDpsKernel(use_clc, FORMATS[name]), *args)
            print(f"compiled {name} {'CLC' if use_clc else 'atomic'} variant for {os.environ['CUTE_DSL_ARCH']}")


if __name__ == "__main__":
    compile_check(sys.argv[1:] or ("fp16", "fp8", "fp4"))
