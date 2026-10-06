// qwen_dps_megakernel.cu — Qwen3-0.6B decode megakernel driven by a dynamic
// persistent tile scheduler (CLC on Blackwell / B200, atomic tickets elsewhere).
//
// Differences from Tools/megakernel/megakernel_5090.cu:
//   * No grid-wide barriers. Each phase of each layer is cut into tiles; a tile
//     waits only on the global counter of the phase it reads from (dataflow).
//   * Tiles are handed out dynamically (dps_scheduler.cuh), so no CTA idles while
//     16 of them do attention, and the grid no longer has to equal the SM count.
//   * Warp specialisation as in CUTLASS's dense_gemm_persistent_dynamic example:
//     a scheduler warp fetches work, a load warp streams the next tiles' weights
//     into a shared-memory ring with TMA bulk copies (they do not depend on
//     activations, so they overlap the dependency wait), and 8 compute warps
//     consume them.
//   * The whole request — prompt prefill, every decode step, LM head and argmax —
//     is ONE launch. Generated tokens are fed back on device.
//   * Activations are rounded to fp16 at the same points as HF transformers.
//   * Weights can be fp16, MXF8 (e4m3 + e8m0 per 32) or NVF4 (e2m1 + e4m3 per 16 +
//     fp32 per tensor). Tiles keep the same rows in every format, so the task graph
//     is unchanged; quantized tiles are just fewer bytes and the ring gets deeper.
//     Dequantisation happens in the GEMV: on sm_100 one F2FP unpacks two weights to
//     fp16 and FHFMA (fp32 += fp16 * fp16) consumes them, ~1.5 instructions/weight.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "dps_scheduler.cuh"
#include "qwen_dps.h"

namespace qwen {

// model
constexpr int   H     = 1024;
constexpr int   I     = 3072;
constexpr int   NQ    = 16;
constexpr int   NKV   = 8;
constexpr int   HD    = 128;
constexpr int   QS    = NQ * HD;
constexpr int   NL    = 28;
constexpr int   VOCAB = 151936;
constexpr float EPS   = 1e-6f;

// CTA layout
#ifndef DPS_RING_BYTES
#define DPS_RING_BYTES 196608  // weight ring: 192 KB of the 227 KB a B200 CTA may use
#endif
#ifndef DPS_SSTAGES
#define DPS_SSTAGES 1          // tiles a CTA may claim before starting them (see dps_scheduler.cuh)
#endif
constexpr int COMPUTE_WARPS   = 8;
constexpr int COMPUTE_THREADS = COMPUTE_WARPS * 32;
constexpr int LOAD_WARP       = 8;
constexpr int SCHED_WARP      = 9;
constexpr int BLOCK_THREADS   = 320;
constexpr int GROUP_BAR_ID    = 1;   // named barrier shared by the compute warps only

// work decomposition (tiles per layer)
// Every phase has at most 128 tiles, so on B200 (148 SMs) a phase is one round: with
// an even hand-out each SM runs at most one tile of it. A tile is a few ring stages
// ("chunks") of weights: a tile costs ~1.5 us of fixed latency (dependency check,
// input prep, barriers, done-signal) but streams 32 KB in ~0.6 us, so fewer, bigger
// tiles beat several rounds of small ones.
// QKV rows are grouped per KV head (2 q heads + k + v = 512 rows) so attention
// for head h can start as soon as its own group is done.
constexpr int QKV_CHUNK_ROWS  = 16;
constexpr int QKV_ROWS        = 32;                        // 2 chunks
constexpr int GROUP_ROWS      = 2 * HD + HD + HD;
constexpr int QKV_TILES_GROUP = GROUP_ROWS / QKV_ROWS;     // 16
constexpr int T_QKV           = NKV * QKV_TILES_GROUP;     // 128
constexpr int ATTN_CHUNK      = 256;                       // KV positions per split
constexpr int ATTN_SPLITS     = 8;                         // => max_seq <= 2048
constexpr int T_ATTN          = NKV * ATTN_SPLITS;         // 64
constexpr int O_ROWS          = 8;                         // 1 chunk
constexpr int T_O             = H / O_ROWS;                // 128
constexpr int GU_CHUNK_ROWS   = 8;                         // 8 gate + 8 up rows per chunk
constexpr int GU_ROWS         = 24;                        // 3 chunks
constexpr int T_GU            = I / GU_ROWS;               // 128
constexpr int D_CHUNK_ROWS    = 4;
constexpr int D_ROWS          = 8;                         // 2 chunks
constexpr int T_D             = H / D_ROWS;                // 128
constexpr int T_LAYER         = T_QKV + T_ATTN + T_O + T_GU + T_D;
constexpr int T_STEP          = NL * T_LAYER;
constexpr int LM_ROWS         = 128;
constexpr int LM_CHUNK_ROWS   = 16;
constexpr int LM_CHUNKS       = LM_ROWS / LM_CHUNK_ROWS;   // 8
constexpr int T_LM            = VOCAB / LM_ROWS;           // 1187
constexpr int QKV_CHUNKS      = QKV_ROWS / QKV_CHUNK_ROWS;
constexpr int GU_CHUNKS       = GU_ROWS / GU_CHUNK_ROWS;
constexpr int D_CHUNKS        = D_ROWS / D_CHUNK_ROWS;
static_assert(VOCAB % LM_ROWS == 0, "vocab must split evenly into LM tiles");
static_assert(HD % QKV_ROWS == 0, "a QKV tile must not straddle q/k/v heads");
static_assert(QKV_ROWS % QKV_CHUNK_ROWS == 0 && QKV_CHUNK_ROWS % COMPUTE_WARPS == 0, "QKV chunking");
static_assert(GU_CHUNK_ROWS == COMPUTE_WARPS && GU_ROWS % GU_CHUNK_ROWS == 0, "gate/up: one row pair per warp");
static_assert(D_CHUNK_ROWS * 2 == COMPUTE_WARPS && D_ROWS % D_CHUNK_ROWS == 0, "down: two warps per row");
static_assert(O_ROWS == COMPUTE_WARPS, "O-proj: one row per warp");

// weight formats
// GROUP = weights one lane handles per step (one 16-byte or 8-byte load);
// SBLOCK = weights sharing one scale along K (0 = unscaled).
template <int F> struct Fmt;
template <> struct Fmt<QWEN_DPS_FP16> { static constexpr int BITS = 16, GROUP = 8,  SBLOCK = 0;  };
template <> struct Fmt<QWEN_DPS_FP8>  { static constexpr int BITS = 8,  GROUP = 16, SBLOCK = 32; };
template <> struct Fmt<QWEN_DPS_FP4>  { static constexpr int BITS = 4,  GROUP = 16, SBLOCK = 16; };

template <int F> __host__ __device__ constexpr int row_bytes(int k) { return k * Fmt<F>::BITS / 8; }
template <int F> __host__ __device__ constexpr int scale_bytes(int k) {
    return Fmt<F>::SBLOCK ? k / Fmt<F>::SBLOCK : 0;
}
template <int F> __host__ __device__ constexpr int lane_groups(int k) { return k / (32 * Fmt<F>::GROUP); }
template <int F> constexpr int XU = Fmt<F>::GROUP / 8;   // uint4s of x per lane group
template <int F> constexpr int tile_bytes(int rows, int k) { return rows * (row_bytes<F>(k) + scale_bytes<F>(k)); }

constexpr int cmax(int a, int b) { return a > b ? a : b; }
constexpr int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// One ring stage holds the largest weight chunk (weights followed by their scales).
template <int F> constexpr int stage_bytes() {
    const int b = cmax(cmax(tile_bytes<F>(QKV_CHUNK_ROWS, H), tile_bytes<F>(O_ROWS, QS)),
                       cmax(cmax(tile_bytes<F>(2 * GU_CHUNK_ROWS, H), tile_bytes<F>(D_CHUNK_ROWS, I)),
                            tile_bytes<F>(LM_CHUNK_ROWS, H)));
    return (b + 127) / 128 * 128;
}
template <int F> constexpr int weight_stages() { return clampi(DPS_RING_BYTES / stage_bytes<F>(), 1, 16); }
// Same claim-ahead for every format: claiming deeper only spreads tiles unevenly.
template <int F> constexpr int sched_stages() { return clampi(DPS_SSTAGES, 1, 12); }

enum Phase { PH_QKV = 0, PH_ATTN, PH_OPROJ, PH_GATEUP, PH_DOWN, PH_LM };

// global counters (monotonic, zeroed per launch, 32 B apart)
enum Counter {
    C_TICKET    = 0,
    C_QKV       = 1,            // + kv group, counts finished QKV tiles
    C_ASPLIT    = C_QKV + NKV,  // + kv head, counts finished attention splits
    C_ATTN      = C_ASPLIT + NKV,   // combined kv heads
    C_OPROJ,
    C_GATEUP,
    C_DOWN,
    C_LM_ARRIVE,
    C_LM_DONE,                  // finalized LM steps (next token known)
    C_EOS,
    C_NUM
};
constexpr int CSTRIDE = 8;

struct LmPart { float v; int i; };

struct Params {
    const half *embed, *final_norm, *cos_t, *sin_t;
    QwenDpsMatrix lm;
    const QwenDpsLayerWeights *layers;
    half     *k_cache, *v_cache;
    unsigned *sync;
    int      *tokens, *output_log;
    half     *hidden, *h1, *q, *k, *v, *attn_out, *act;
    float    *attn_part;   // [NKV][ATTN_SPLITS][2][HD + 2]: max, sum, acc[HD]
    LmPart   *lm_part;     // [T_LM]
    int   n_pre;           // leading steps that only fill the KV cache
    int   total_tiles;
    int   start_pos, max_seq, eos_token, sched_mode;
    float attn_scale;
    QwenDpsTraceRecord *trace;   // DPS_TRACE builds only
    int   trace_first, trace_count;
};

// Shared memory that does not depend on the weight format.
struct alignas(16) SmemCommon {
    half     xs[I];                       // matvec input vector (fp16 values)
    float    qs[2][HD];                   // roped queries of the 2 heads sharing a KV head
    float    wm[COMPUTE_WARPS][2], wl[COMPUTE_WARPS][2];
    float    wacc[COMPUTE_WARPS][2][HD];
    float    red[D_CHUNKS * COMPUTE_WARPS];   // block sums; per-chunk partials of a down tile
    float    best_v[COMPUTE_WARPS];
    int      best_i[COMPUTE_WARPS];
    int      x_tag;                       // which input currently sits in xs
    int      flag;
    unsigned dep_seen[C_NUM];             // last counter values observed (thread 0)
    unsigned tr_prep, tr_wwait;           // DPS_TRACE: ns spent in input prep / weight waits this tile
};

template <int F>
struct alignas(128) Smem {
    static constexpr int W = weight_stages<F>(), SST = sched_stages<F>(), SB = stage_bytes<F>();
    uint8_t wbuf[W][SB];
    dps::Barrier w_full[W], w_empty[W];
    dps::SchedulerStorage<SST> sched;
    SmemCommon c;
};

template <int F> using Scheduler = dps::DynamicPersistentTileScheduler<Smem<F>::SST>;
template <int F> using RingState = dps::PipelineState<Smem<F>::W>;

struct Tile { int step, layer, phase, idx; };

// small helpers
__device__ __forceinline__ void group_bar() { dps::named_bar_sync(GROUP_BAR_ID, COMPUTE_THREADS); }

// DPS_TRACE: adds the scope's duration (as seen by compute thread 0) to a per-tile counter.
struct TraceSpan {
#if DPS_TRACE
    unsigned *acc;
    uint64_t  t0;
    __device__ explicit TraceSpan(unsigned &a) : acc(&a), t0(dps::globaltimer_ns()) {}
    __device__ ~TraceSpan() {
        if (threadIdx.x == 0) *acc += static_cast<unsigned>(dps::globaltimer_ns() - t0);
    }
#else
    __device__ explicit TraceSpan(unsigned &) {}
#endif
};

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ bool better(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);   // torch.argmax: first max wins
}

__device__ __forceinline__ float group_sum(SmemCommon &S, float v) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) S.red[threadIdx.x >> 5] = v;
    group_bar();
    float s = 0.f;
    #pragma unroll
    for (int w = 0; w < COMPUTE_WARPS; ++w) s += S.red[w];
    group_bar();
    return s;
}

__device__ __forceinline__ float h2f_round(float x) { return __half2float(__float2half(x)); }

__device__ __forceinline__ float ldcg_h(const half *p) {
    return __half2float(__ushort_as_half(__ldcg(reinterpret_cast<const unsigned short *>(p))));
}

// dequantising dot products
// acc + w.lo*x.lo + w.hi*x.hi on packed fp16 pairs. fp16 x fp16 is exact in fp32,
// so sm_100's mixed-precision FMA and the convert+FFMA fallback agree bit for bit.
__device__ __forceinline__ float fma2(float acc, uint32_t w, uint32_t x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{\n"
        ".reg .b16 wl, wh, xl, xh;\n"
        "mov.b32 {wl, wh}, %1;\n"
        "mov.b32 {xl, xh}, %2;\n"
        "fma.rn.f32.f16 %0, wl, xl, %0;\n"
        "fma.rn.f32.f16 %0, wh, xh, %0;\n"
        "}" : "+f"(acc) : "r"(w), "r"(x));
    return acc;
#else
    const float2 a = __half22float2(*reinterpret_cast<const half2 *>(&w));
    const float2 b = __half22float2(*reinterpret_cast<const half2 *>(&x));
    acc = fmaf(a.x, b.x, acc);
    return fmaf(a.y, b.y, acc);
#endif
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t v) {
    const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
    return static_cast<uint32_t>(r.x) | (static_cast<uint32_t>(r.y) << 16);
}

__device__ __forceinline__ uint32_t e2m1x2_to_f16x2(uint32_t v) {
    const __half2_raw r = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(v), __NV_E2M1);
    return static_cast<uint32_t>(r.x) | (static_cast<uint32_t>(r.y) << 16);
}

__device__ __forceinline__ float e8m0_to_f32(uint32_t s) { return __uint_as_float(s << 23); }

__device__ __forceinline__ float e4m3_to_f32(uint32_t s) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(s), __NV_E4M3)));
}

// Unscaled partial dot product of lane group g of a weight row with x (GROUP values).
template <int F> __device__ __forceinline__ float group_dot(const uint8_t *wrow, int g, const uint4 *x);

template <> __device__ __forceinline__ float group_dot<QWEN_DPS_FP16>(const uint8_t *wrow, int g, const uint4 *x) {
    const uint4 w = reinterpret_cast<const uint4 *>(wrow)[g];
    float p = 0.f;
    p = fma2(p, w.x, x[0].x);
    p = fma2(p, w.y, x[0].y);
    p = fma2(p, w.z, x[0].z);
    return fma2(p, w.w, x[0].w);
}

template <> __device__ __forceinline__ float group_dot<QWEN_DPS_FP8>(const uint8_t *wrow, int g, const uint4 *x) {
    const uint4 w = reinterpret_cast<const uint4 *>(wrow)[g];
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
    const uint32_t xs[8] = {x[0].x, x[0].y, x[0].z, x[0].w, x[1].x, x[1].y, x[1].z, x[1].w};
    float p = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        p = fma2(p, e4m3x2_to_f16x2(ws[i] & 0xffffu), xs[2 * i]);
        p = fma2(p, e4m3x2_to_f16x2(ws[i] >> 16), xs[2 * i + 1]);
    }
    return p;
}

template <> __device__ __forceinline__ float group_dot<QWEN_DPS_FP4>(const uint8_t *wrow, int g, const uint4 *x) {
    const uint2 w = reinterpret_cast<const uint2 *>(wrow)[g];
    const uint32_t ws[2] = {w.x, w.y};
    const uint32_t xs[8] = {x[0].x, x[0].y, x[0].z, x[0].w, x[1].x, x[1].y, x[1].z, x[1].w};
    float p = 0.f;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) p = fma2(p, e2m1x2_to_f16x2((ws[i] >> (8 * b)) & 0xffu), xs[4 * i + b]);
    }
    return p;
}

template <int F> __device__ __forceinline__ float group_scale(const uint8_t *srow, int g) {
    if constexpr (F == QWEN_DPS_FP8) return e8m0_to_f32(srow[g >> 1]);   // 2 groups per 32-block
    else return e4m3_to_f32(srow[g]);                                    // 1 group per 16-block
}

// Warp-wide dot product of one weight row slice with x; lane handles groups
// g0 + j*32 + lane. Excludes the per-tensor scale.
template <int F, int NJ>
__device__ __forceinline__ float dot_row(const uint8_t *wrow, const uint8_t *srow, int g0,
                                         const uint4 (&x)[NJ * XU<F>]) {
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int g = g0 + j * 32 + lane;
        const float part = group_dot<F>(wrow, g, &x[j * XU<F>]);
        if constexpr (F == QWEN_DPS_FP16) acc += part;
        else acc = fmaf(part, group_scale<F>(srow, g), acc);
    }
    return warp_sum(acc);
}

// Lane slice of xs matching dot_row's groups.
template <int F, int NJ>
__device__ __forceinline__ void load_x(const SmemCommon &S, int k_offset, uint4 (&x)[NJ * XU<F>]) {
    const uint4 *xs = reinterpret_cast<const uint4 *>(S.xs + k_offset);
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int j = 0; j < NJ; ++j) {
        #pragma unroll
        for (int u = 0; u < XU<F>; ++u) x[j * XU<F> + u] = xs[(j * 32 + lane) * XU<F> + u];
    }
}

// tile rasterization (the _swizzle_and_rasterize analogue)
// Tickets are laid out in topological order:
//   [n_pre prefill steps x T_STEP] [decode steps x (T_STEP + T_LM)]
// and within a layer: QKV (group-major) | ATTN (head-major) | O | GATEUP | DOWN.
__device__ __forceinline__ Tile decode_tile(int t, const Params &p) {
    Tile r;
    const int pre = p.n_pre * T_STEP;
    int rem;
    if (t < pre) {
        r.step = t / T_STEP;
        rem    = t - r.step * T_STEP;
    } else {
        const int u = t - pre, q = u / (T_STEP + T_LM);
        r.step = p.n_pre + q;
        rem    = u - q * (T_STEP + T_LM);
    }
    if (rem >= T_STEP) { r.layer = NL; r.phase = PH_LM; r.idx = rem - T_STEP; return r; }
    r.layer = rem / T_LAYER;
    int x = rem - r.layer * T_LAYER;
    if (x < T_QKV)  { r.phase = PH_QKV;    r.idx = x; return r; }
    x -= T_QKV;
    if (x < T_ATTN) { r.phase = PH_ATTN;   r.idx = x; return r; }
    x -= T_ATTN;
    if (x < T_O)    { r.phase = PH_OPROJ;  r.idx = x; return r; }
    x -= T_O;
    if (x < T_GU)   { r.phase = PH_GATEUP; r.idx = x; return r; }
    r.phase = PH_DOWN; r.idx = x - T_GU;
    return r;
}

// QKV tile -> (q/k/v, first row). Rows of one tile never straddle q/k/v.
// A group's tiles cover its 2 q heads, then its k head, then its v head.
__device__ __forceinline__ void qkv_rows(const Tile &T, int &which, int &row0) {
    constexpr int TQ = 2 * HD / QKV_ROWS, TK = HD / QKV_ROWS;
    const int g = T.idx / QKV_TILES_GROUP, j = T.idx % QKV_TILES_GROUP;
    if (j < TQ)           { which = 0; row0 = g * 2 * HD + j * QKV_ROWS; }
    else if (j < TQ + TK) { which = 1; row0 = g * HD + (j - TQ) * QKV_ROWS; }
    else                  { which = 2; row0 = g * HD + (j - TQ - TK) * QKV_ROWS; }
}

__device__ __forceinline__ const QwenDpsMatrix &qkv_matrix(const QwenDpsLayerWeights &lw, int which) {
    return which == 0 ? lw.q : (which == 1 ? lw.k : lw.v);
}

// A chunk is one ring stage: up to 4 bulk copies (weights, then their scales).
struct Seg { const void *src; uint32_t dst, bytes; };
struct Chunk {
    Seg seg[4];
    int n = 0;
    uint32_t total = 0;
    __device__ void add(const void *src, uint32_t dst, uint32_t bytes) {
        seg[n++] = {src, dst, bytes};
        total += bytes;
    }
};

__device__ __forceinline__ int num_chunks(const Tile &T) {
    switch (T.phase) {
    case PH_QKV:    return QKV_CHUNKS;
    case PH_ATTN:   return 0;
    case PH_OPROJ:  return 1;
    case PH_GATEUP: return GU_CHUNKS;
    case PH_DOWN:   return D_CHUNKS;
    default:        return LM_CHUNKS;
    }
}

// rows [row0, row0 + rows) of m: weights at stage offset dst_w, scales at dst_s.
template <int F>
__device__ __forceinline__ void add_rows(Chunk &ch, const QwenDpsMatrix &m, int k, int row0, int rows,
                                         uint32_t dst_w, uint32_t dst_s) {
    ch.add(static_cast<const uint8_t *>(m.data) + (size_t)row0 * row_bytes<F>(k), dst_w, rows * row_bytes<F>(k));
    if constexpr (Fmt<F>::SBLOCK != 0)
        ch.add(static_cast<const uint8_t *>(m.scale) + (size_t)row0 * scale_bytes<F>(k), dst_s,
               rows * scale_bytes<F>(k));
}

template <int F>
__device__ Chunk tile_chunk(const Tile &T, int c, const Params &p) {
    Chunk ch;
    if (T.phase == PH_LM) {
        constexpr int rb = row_bytes<F>(H);
        add_rows<F>(ch, p.lm, H, T.idx * LM_ROWS + c * LM_CHUNK_ROWS, LM_CHUNK_ROWS, 0, LM_CHUNK_ROWS * rb);
        return ch;
    }
    const QwenDpsLayerWeights &lw = p.layers[T.layer];
    switch (T.phase) {
    case PH_QKV: {
        int which, row0;
        qkv_rows(T, which, row0);
        add_rows<F>(ch, qkv_matrix(lw, which), H, row0 + c * QKV_CHUNK_ROWS, QKV_CHUNK_ROWS, 0,
                    QKV_CHUNK_ROWS * row_bytes<F>(H));
        break;
    }
    case PH_OPROJ:
        add_rows<F>(ch, lw.o, QS, T.idx * O_ROWS, O_ROWS, 0, O_ROWS * row_bytes<F>(QS));
        break;
    case PH_GATEUP: {
        // [gate w | up w | gate scales | up scales]: rows 0..7 gate, 8..15 up
        constexpr int rb = row_bytes<F>(H), sb = scale_bytes<F>(H);
        const int row0 = T.idx * GU_ROWS + c * GU_CHUNK_ROWS;
        add_rows<F>(ch, lw.gate, H, row0, GU_CHUNK_ROWS, 0, 2 * GU_CHUNK_ROWS * rb);
        add_rows<F>(ch, lw.up, H, row0, GU_CHUNK_ROWS, GU_CHUNK_ROWS * rb,
                    2 * GU_CHUNK_ROWS * rb + GU_CHUNK_ROWS * sb);
        break;
    }
    case PH_DOWN:
        add_rows<F>(ch, lw.down, I, T.idx * D_ROWS + c * D_CHUNK_ROWS, D_CHUNK_ROWS, 0,
                    D_CHUNK_ROWS * row_bytes<F>(I));
        break;
    }
    return ch;
}

// dependency handling (compute thread 0 only)
__device__ __forceinline__ void wait_counter(SmemCommon &S, const Params &p, int c, unsigned target) {
    if (S.dep_seen[c] >= target) return;
    const unsigned *ptr = p.sync + c * CSTRIDE;
    unsigned v = dps::ld_acquire(ptr);
    while (v < target) {
        dps::nanosleep_ns(32);
        v = dps::ld_acquire(ptr);
    }
    S.dep_seen[c] = v;
}

__device__ __forceinline__ void signal_done(const Params &p, int c) {
    dps::signal_release(p.sync + c * CSTRIDE);
}

__device__ void wait_dependencies(SmemCommon &S, const Params &p, const Tile &T) {
    const unsigned inst = (unsigned)(T.step * NL + T.layer);
    switch (T.phase) {
    case PH_QKV:
        if (T.layer > 0) {
            wait_counter(S, p, C_DOWN, inst * T_D);
        } else {
            if (T.step > 0) wait_counter(S, p, C_DOWN, inst * T_D);                 // previous token done
            if (T.step > p.n_pre) wait_counter(S, p, C_LM_DONE, T.step - p.n_pre);  // its argmax is known
        }
        break;
    case PH_ATTN:   wait_counter(S, p, C_QKV + T.idx / ATTN_SPLITS, (inst + 1) * QKV_TILES_GROUP); break;
    case PH_OPROJ:  wait_counter(S, p, C_ATTN, (inst + 1) * NKV); break;
    case PH_GATEUP: wait_counter(S, p, C_OPROJ, (inst + 1) * T_O); break;
    case PH_DOWN:   wait_counter(S, p, C_GATEUP, (inst + 1) * T_GU); break;
    case PH_LM:     wait_counter(S, p, C_DOWN, (unsigned)(T.step * NL + NL) * T_D); break;
    }
}

// input preparation (all compute threads, called right after a group_bar)
// HF Qwen3RMSNorm in fp16: y = half(x * rsqrt(mean(x^2) + eps)); out = half(w * y).
__device__ void prep_rmsnorm(SmemCommon &S, const half *src, const half *w) {
    TraceSpan span(S.tr_prep);
    const int tid = threadIdx.x;   // 256 threads x 4 elements
    const uint2 raw = __ldcg(reinterpret_cast<const uint2 *>(src) + tid);
    const half2 *xh = reinterpret_cast<const half2 *>(&raw);
    const float2 a = __half22float2(xh[0]), b = __half22float2(xh[1]);
    const float ss = group_sum(S, a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y);
    const float rstd = rsqrtf(ss / (float)H + EPS);
    const uint2 wraw = __ldg(reinterpret_cast<const uint2 *>(w) + tid);
    const half2 *wh = reinterpret_cast<const half2 *>(&wraw);
    const float2 wa = __half22float2(wh[0]), wb = __half22float2(wh[1]);
    uint2 out;
    half2 *oh = reinterpret_cast<half2 *>(&out);
    oh[0] = __floats2half2_rn(wa.x * h2f_round(a.x * rstd), wa.y * h2f_round(a.y * rstd));
    oh[1] = __floats2half2_rn(wb.x * h2f_round(b.x * rstd), wb.y * h2f_round(b.y * rstd));
    reinterpret_cast<uint2 *>(S.xs)[tid] = out;
    group_bar();
}

__device__ void prep_copy(SmemCommon &S, const half *src, int n) {
    TraceSpan span(S.tr_prep);
    const uint4 *s = reinterpret_cast<const uint4 *>(src);
    uint4 *d = reinterpret_cast<uint4 *>(S.xs);
    for (int i = threadIdx.x; i < n / 8; i += COMPUTE_THREADS) d[i] = __ldcg(s + i);
    group_bar();
}

__device__ __forceinline__ int x_tag(const Tile &T) { return ((T.step * (NL + 1)) + T.layer) * 8 + T.phase; }

__device__ __forceinline__ const half *layer_input(const Params &p, const Tile &T) {
    if (T.layer > 0) return p.hidden;
    return p.embed + (size_t)__ldcg(p.tokens + T.step) * H;
}

// weight ring (consumer side)
template <int F>
__device__ __forceinline__ const uint8_t *stage_wait(Smem<F> &S, RingState<F> &ws) {
    TraceSpan span(S.c.tr_wwait);
    dps::bar_wait(&S.w_full[ws.index], ws.phase);
    return S.wbuf[ws.index];
}
template <int F>
__device__ __forceinline__ void stage_release(Smem<F> &S, RingState<F> &ws) {
    __syncwarp();
    if ((threadIdx.x & 31) == 0) dps::bar_arrive(&S.w_empty[ws.index]);
    ws.advance();
}

// tiles
template <int F>
__device__ void tile_qkv(Smem<F> &S, const Params &p, const Tile &T, RingState<F> &ws) {
    constexpr int rb = row_bytes<F>(H), sb = scale_bytes<F>(H), NJ = lane_groups<F>(H);
    const QwenDpsLayerWeights &lw = p.layers[T.layer];
    const int tag = x_tag(T);
    if (S.c.x_tag != tag) {
        prep_rmsnorm(S.c, layer_input(p, T), static_cast<const half *>(lw.input_layernorm));
        if (threadIdx.x == 0) S.c.x_tag = tag;
    }
    int which, row0;
    qkv_rows(T, which, row0);
    half *out = (which == 0 ? p.q : which == 1 ? p.k : p.v) + row0;
    const float gs = qkv_matrix(lw, which).gscale;

    uint4 x[NJ * XU<F>];
    load_x<F, NJ>(S.c, 0, x);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll 1
    for (int c = 0; c < QKV_CHUNKS; ++c) {
        const uint8_t *st = stage_wait(S, ws);
        #pragma unroll
        for (int r = warp; r < QKV_CHUNK_ROWS; r += COMPUTE_WARPS) {
            const float d = dot_row<F, NJ>(st + r * rb, st + QKV_CHUNK_ROWS * rb + r * sb, 0, x) * gs;
            if (lane == 0) out[c * QKV_CHUNK_ROWS + r] = __float2half(d);
        }
        stage_release(S, ws);
    }
    group_bar();
    if (threadIdx.x == 0) signal_done(p, C_QKV + T.idx / QKV_TILES_GROUP);
}

// RMSNorm over one 128-dim head + RoPE, HF rounding. Lane owns dims lane + 32*i,
// so the rotate_half partners (d, d+64) sit in the same lane.
__device__ __forceinline__ void head_norm_rope(const half *src, const half *nw, const half *cos_row,
                                               const half *sin_row, float (&out)[4]) {
    const int lane = threadIdx.x & 31;
    float x[4], n[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = ldcg_h(src + lane + 32 * i);
    const float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
    const float rstd = rsqrtf(ss / (float)HD + EPS);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        n[i] = h2f_round(__half2float(__ldg(nw + lane + 32 * i)) * h2f_round(x[i] * rstd));
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane + 32 * i;
        const float c = __half2float(__ldg(cos_row + d)), s = __half2float(__ldg(sin_row + d));
        const float rot = (i < 2) ? -n[i + 2] : n[i - 2];
        out[i] = h2f_round(h2f_round(n[i] * c) + h2f_round(rot * s));
    }
}

__device__ void tile_attention(SmemCommon &S, const Params &p, const Tile &T) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h = T.idx / ATTN_SPLITS, c = T.idx % ATTN_SPLITS;
    const int pos = p.start_pos + T.step, len = pos + 1;
    const int t0 = c * ATTN_CHUNK, t1 = min(t0 + ATTN_CHUNK, len);
    const QwenDpsLayerWeights &lw = p.layers[T.layer];
    half *kc = p.k_cache + ((size_t)T.layer * NKV + h) * p.max_seq * HD;
    half *vc = p.v_cache + ((size_t)T.layer * NKV + h) * p.max_seq * HD;
    const half *cos_row = p.cos_t + (size_t)pos * HD, *sin_row = p.sin_t + (size_t)pos * HD;

    if (t0 < len) {
        // q-norm + RoPE for the two query heads of this KV head; the split that owns
        // the new position also normalises, ropes and appends k, and appends v.
        if (warp < 2) {
            float o[4];
            head_norm_rope(p.q + (2 * h + warp) * HD, static_cast<const half *>(lw.q_norm), cos_row, sin_row, o);
            #pragma unroll
            for (int i = 0; i < 4; ++i) S.qs[warp][lane + 32 * i] = o[i];
        } else if (warp == 2 && pos < t1) {
            float o[4];
            head_norm_rope(p.k + h * HD, static_cast<const half *>(lw.k_norm), cos_row, sin_row, o);
            #pragma unroll
            for (int i = 0; i < 4; ++i) kc[(size_t)pos * HD + lane + 32 * i] = __float2half(o[i]);
        } else if (warp == 3 && pos < t1) {
            reinterpret_cast<uint2 *>(vc + (size_t)pos * HD)[lane] =
                __ldcg(reinterpret_cast<const uint2 *>(p.v + h * HD) + lane);
        }
        group_bar();

        // Per-warp online softmax over positions t0+warp, t0+warp+8, ...
        // Lane owns dims 4*lane .. 4*lane+3 so one 8-byte load covers a row.
        const float4 qa = reinterpret_cast<const float4 *>(S.qs[0])[lane];
        const float4 qb = reinterpret_cast<const float4 *>(S.qs[1])[lane];
        float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        constexpr int U = 4;
        for (int tb = t0 + warp; tb < t1; tb += COMPUTE_WARPS * U) {
            uint2 kr[U], vr[U];
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                const int t = tb + u * COMPUTE_WARPS;
                if (t < t1) {
                    kr[u] = __ldcg(reinterpret_cast<const uint2 *>(kc + (size_t)t * HD) + lane);
                    vr[u] = __ldcg(reinterpret_cast<const uint2 *>(vc + (size_t)t * HD) + lane);
                }
            }
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                if (tb + u * COMPUTE_WARPS >= t1) break;
                const half2 *kh = reinterpret_cast<const half2 *>(&kr[u]);
                const half2 *vh = reinterpret_cast<const half2 *>(&vr[u]);
                const float2 k01 = __half22float2(kh[0]), k23 = __half22float2(kh[1]);
                const float2 v01 = __half22float2(vh[0]), v23 = __half22float2(vh[1]);
                const float vv[4] = {v01.x, v01.y, v23.x, v23.y};
                float s[2];
                s[0] = warp_sum(qa.x * k01.x + qa.y * k01.y + qa.z * k23.x + qa.w * k23.y) * p.attn_scale;
                s[1] = warp_sum(qb.x * k01.x + qb.y * k01.y + qb.z * k23.x + qb.w * k23.y) * p.attn_scale;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float mn = fmaxf(m[j], s[j]);
                    const float eo = __expf(m[j] - mn), e = __expf(s[j] - mn);
                    l[j] = l[j] * eo + e;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) acc[j][i] = acc[j][i] * eo + e * vv[i];
                    m[j] = mn;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            reinterpret_cast<float4 *>(S.wacc[warp][j])[lane] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
            if (lane == 0) { S.wm[warp][j] = m[j]; S.wl[warp][j] = l[j]; }
        }
        group_bar();

        // Merge the 8 warps, write this split's partial (max, sum, acc) to global.
        const int j = tid >> 7, d = tid & (HD - 1);
        float M = -INFINITY;
        #pragma unroll
        for (int w = 0; w < COMPUTE_WARPS; ++w) M = fmaxf(M, S.wm[w][j]);
        float L = 0.f, A = 0.f;
        #pragma unroll
        for (int w = 0; w < COMPUTE_WARPS; ++w) {
            const float e = (S.wm[w][j] == -INFINITY) ? 0.f : __expf(S.wm[w][j] - M);
            L += S.wl[w][j] * e;
            A += S.wacc[w][j][d] * e;
        }
        float *part = p.attn_part + (((size_t)h * ATTN_SPLITS + c) * 2 + j) * (HD + 2);
        part[2 + d] = A;
        if (d == 0) { part[0] = M; part[1] = L; }
    }

    // Split-K style hand-off: the last split to finish for this KV head combines.
    group_bar();
    if (tid == 0) {
        __threadfence();
        const unsigned old = atomicAdd(p.sync + (C_ASPLIT + h) * CSTRIDE, 1u);
        S.flag = ((old + 1) % ATTN_SPLITS) == 0;
        if (S.flag) __threadfence();
    }
    group_bar();
    if (S.flag) {
        const int n_act = (len + ATTN_CHUNK - 1) / ATTN_CHUNK;
        const int j = tid >> 7, d = tid & (HD - 1);
        float M = -INFINITY;
        for (int s = 0; s < n_act; ++s)
            M = fmaxf(M, __ldcg(p.attn_part + (((size_t)h * ATTN_SPLITS + s) * 2 + j) * (HD + 2)));
        float L = 0.f, A = 0.f;
        for (int s = 0; s < n_act; ++s) {
            const float *part = p.attn_part + (((size_t)h * ATTN_SPLITS + s) * 2 + j) * (HD + 2);
            const float e = __expf(__ldcg(part) - M);
            L += __ldcg(part + 1) * e;
            A += __ldcg(part + 2 + d) * e;
        }
        p.attn_out[(2 * h + j) * HD + d] = __float2half(A / L);
        group_bar();
        if (tid == 0) signal_done(p, C_ATTN);
    }
}

template <int F>
__device__ void tile_oproj(Smem<F> &S, const Params &p, const Tile &T, RingState<F> &ws) {
    constexpr int rb = row_bytes<F>(QS), sb = scale_bytes<F>(QS), NJ = lane_groups<F>(QS);
    const int tag = x_tag(T);
    if (S.c.x_tag != tag) {
        prep_copy(S.c, p.attn_out, QS);
        if (threadIdx.x == 0) S.c.x_tag = tag;
    }
    uint4 x[NJ * XU<F>];
    load_x<F, NJ>(S.c, 0, x);
    const uint8_t *st = stage_wait(S, ws);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float d = dot_row<F, NJ>(st + warp * rb, st + O_ROWS * rb + warp * sb, 0, x) * p.layers[T.layer].o.gscale;
    if (lane == 0) {
        const int row = T.idx * O_ROWS + warp;
        const float res = ldcg_h(layer_input(p, T) + row);
        p.h1[row] = __float2half(res + h2f_round(d));
    }
    stage_release(S, ws);
    group_bar();
    if (threadIdx.x == 0) signal_done(p, C_OPROJ);
}

template <int F>
__device__ void tile_gateup(Smem<F> &S, const Params &p, const Tile &T, RingState<F> &ws) {
    constexpr int rb = row_bytes<F>(H), sb = scale_bytes<F>(H), NJ = lane_groups<F>(H);
    const QwenDpsLayerWeights &lw = p.layers[T.layer];
    const int tag = x_tag(T);
    if (S.c.x_tag != tag) {
        prep_rmsnorm(S.c, p.h1, static_cast<const half *>(lw.post_attn_layernorm));
        if (threadIdx.x == 0) S.c.x_tag = tag;
    }
    uint4 x[NJ * XU<F>];
    load_x<F, NJ>(S.c, 0, x);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll 1
    for (int c = 0; c < GU_CHUNKS; ++c) {
        const uint8_t *st = stage_wait(S, ws);
        const uint8_t *sc = st + 2 * GU_CHUNK_ROWS * rb;
        const float g = h2f_round(dot_row<F, NJ>(st + warp * rb, sc + warp * sb, 0, x) * lw.gate.gscale);
        const float u = h2f_round(dot_row<F, NJ>(st + (GU_CHUNK_ROWS + warp) * rb,
                                                 sc + (GU_CHUNK_ROWS + warp) * sb, 0, x) * lw.up.gscale);
        stage_release(S, ws);
        if (lane == 0) {
            const float silu = h2f_round(g / (1.f + expf(-g)));
            p.act[T.idx * GU_ROWS + c * GU_CHUNK_ROWS + warp] = __float2half(silu * u);
        }
    }
    group_bar();
    if (threadIdx.x == 0) signal_done(p, C_GATEUP);
}

template <int F>
__device__ void tile_down(Smem<F> &S, const Params &p, const Tile &T, RingState<F> &ws) {
    constexpr int rb = row_bytes<F>(I), sb = scale_bytes<F>(I), NJ = lane_groups<F>(I / 2);
    const int tag = x_tag(T);
    if (S.c.x_tag != tag) {
        prep_copy(S.c, p.act, I);
        if (threadIdx.x == 0) S.c.x_tag = tag;
    }
    // Two warps per row, each covering half of K = 3072.
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = warp >> 1, khalf = warp & 1;
    uint4 x[NJ * XU<F>];
    load_x<F, NJ>(S.c, khalf * (I / 2), x);
    #pragma unroll 1
    for (int c = 0; c < D_CHUNKS; ++c) {
        const uint8_t *st = stage_wait(S, ws);
        const float part = dot_row<F, NJ>(st + r * rb, st + D_CHUNK_ROWS * rb + r * sb, khalf * NJ * 32, x);
        stage_release(S, ws);
        if (lane == 0) S.c.red[c * COMPUTE_WARPS + warp] = part;
    }
    group_bar();
    if (threadIdx.x < D_ROWS) {   // thread t finishes row t of the tile: chunk t / 4, row t % 4 in it
        const float *red = S.c.red + (threadIdx.x / D_CHUNK_ROWS) * COMPUTE_WARPS + 2 * (threadIdx.x % D_CHUNK_ROWS);
        const int row = T.idx * D_ROWS + threadIdx.x;
        const float d = h2f_round((red[0] + red[1]) * p.layers[T.layer].down.gscale);
        p.hidden[row] = __float2half(ldcg_h(p.h1 + row) + d);
    }
    group_bar();
    if (threadIdx.x == 0) signal_done(p, C_DOWN);
}

template <int F>
__device__ void tile_lm(Smem<F> &S, const Params &p, const Tile &T, RingState<F> &ws) {
    constexpr int rb = row_bytes<F>(H), sb = scale_bytes<F>(H), NJ = lane_groups<F>(H);
    SmemCommon &C = S.c;
    const int tag = x_tag(T);
    if (C.x_tag != tag) {
        prep_rmsnorm(C, p.hidden, p.final_norm);
        if (threadIdx.x == 0) C.x_tag = tag;
    }
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    uint4 x[NJ * XU<F>];
    load_x<F, NJ>(C, 0, x);
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    #pragma unroll 1
    for (int c = 0; c < LM_CHUNKS; ++c) {
        const uint8_t *st = stage_wait(S, ws);
        #pragma unroll
        for (int r = warp; r < LM_CHUNK_ROWS; r += COMPUTE_WARPS) {
            // HF logits are fp16
            const float logit = h2f_round(dot_row<F, NJ>(st + r * rb, st + LM_CHUNK_ROWS * rb + r * sb, 0, x) *
                                          p.lm.gscale);
            const int row = T.idx * LM_ROWS + c * LM_CHUNK_ROWS + r;
            if (better(logit, row, bv, bi)) { bv = logit; bi = row; }
        }
        stage_release(S, ws);
    }
    if (lane == 0) { C.best_v[warp] = bv; C.best_i[warp] = bi; }
    group_bar();
    if (tid == 0) {
        for (int w = 1; w < COMPUTE_WARPS; ++w)
            if (better(C.best_v[w], C.best_i[w], bv, bi)) { bv = C.best_v[w]; bi = C.best_i[w]; }
        p.lm_part[T.idx] = {bv, bi};
        __threadfence();
        const unsigned old = atomicAdd(p.sync + C_LM_ARRIVE * CSTRIDE, 1u);
        C.flag = ((old + 1) % T_LM) == 0;
        if (C.flag) __threadfence();
    }
    group_bar();
    if (!C.flag) return;

    // Last LM tile of this step: global argmax, feed the token back, maybe stop.
    bv = -INFINITY;
    bi = 0x7fffffff;
    for (int i = tid; i < T_LM; i += COMPUTE_THREADS) {
        const int2 e = __ldcg(reinterpret_cast<const int2 *>(p.lm_part) + i);
        const float v = __int_as_float(e.x);
        if (better(v, e.y, bv, bi)) { bv = v; bi = e.y; }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int   oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    group_bar();   // best_v/best_i are reused
    if (lane == 0) { C.best_v[warp] = bv; C.best_i[warp] = bi; }
    group_bar();
    if (tid == 0) {
        for (int w = 0; w < COMPUTE_WARPS; ++w)
            if (better(C.best_v[w], C.best_i[w], bv, bi)) { bv = C.best_v[w]; bi = C.best_i[w]; }
        unsigned *eos = p.sync + C_EOS * CSTRIDE;
        if (dps::ld_relaxed(eos) == 0u) {
            p.tokens[T.step + 1] = bi;
            p.output_log[T.step - p.n_pre] = bi;
            if (bi == p.eos_token) atomicExch(eos, 1u);
        }
        __threadfence();
        atomicAdd(p.sync + C_LM_DONE * CSTRIDE, 1u);
    }
}

// DPS_TRACE: compute thread 0 writes one record per tile in the traced ticket range.
__device__ void trace_tile(const Params &p, const SmemCommon &S, const dps::WorkTileInfo &w,
                           uint64_t start, uint64_t ready) {
#if DPS_TRACE
    const unsigned i = static_cast<unsigned>(w.tile_idx - p.trace_first);
    if (p.trace == nullptr || i >= static_cast<unsigned>(p.trace_count)) return;
    QwenDpsTraceRecord r;
    r.claim    = w.claim_ns;
    r.start    = start;
    r.ready    = ready;
    r.end      = dps::globaltimer_ns();
    r.prep_ns  = S.tr_prep;
    r.wwait_ns = S.tr_wwait;
    r.sm       = dps::smid();
    r.cta      = blockIdx.x;
    p.trace[i] = r;
#endif
}

// warp roles
template <int F>
__device__ void compute_loop(Smem<F> &S, const Params &p, Scheduler<F> &sched) {
    auto cons = dps::make_consumer_state<Smem<F>::SST>();
    auto ws   = dps::make_consumer_state<Smem<F>::W>();
    for (;;) {
        const dps::WorkTileInfo work = sched.get_current_work(cons);
        if (!work.is_valid_tile) break;
        uint64_t t_start = 0, t_ready = 0;
#if DPS_TRACE
        t_start = dps::globaltimer_ns();
        if (threadIdx.x == 0) S.c.tr_prep = S.c.tr_wwait = 0u;
#endif
        const Tile T = decode_tile(work.tile_idx, p);
        if (threadIdx.x == 0) wait_dependencies(S.c, p, T);
        group_bar();
#if DPS_TRACE
        t_ready = dps::globaltimer_ns();
#endif
        switch (T.phase) {
        case PH_QKV:    tile_qkv<F>(S, p, T, ws); break;
        case PH_ATTN:   tile_attention(S.c, p, T); break;
        case PH_OPROJ:  tile_oproj<F>(S, p, T, ws); break;
        case PH_GATEUP: tile_gateup<F>(S, p, T, ws); break;
        case PH_DOWN:   tile_down<F>(S, p, T, ws); break;
        default:        tile_lm<F>(S, p, T, ws); break;
        }
        if (threadIdx.x == 0) trace_tile(p, S.c, work, t_start, t_ready);
    }
}

// TMA producer: streams weights for the claimed tiles as far ahead as the ring allows.
template <int F>
__device__ void load_loop(Smem<F> &S, const Params &p, Scheduler<F> &sched) {
    auto cons = dps::make_consumer_state<Smem<F>::SST>();
    auto ws   = dps::make_producer_state<Smem<F>::W>();
    const uint64_t policy = dps::l2_evict_first_policy();
    for (;;) {
        const dps::WorkTileInfo work = sched.get_current_work(cons);
        if (!work.is_valid_tile) break;
        const Tile T = decode_tile(work.tile_idx, p);
        const int nc = num_chunks(T);
        for (int c = 0; c < nc; ++c) {
            const Chunk ch = tile_chunk<F>(T, c, p);
            dps::bar_wait(&S.w_empty[ws.index], ws.phase);
            if ((threadIdx.x & 31) == 0) dps::bar_arrive_expect_tx(&S.w_full[ws.index], ch.total);
            __syncwarp();
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i < ch.n)
                    dps::bulk_g2s_warp(S.wbuf[ws.index] + ch.seg[i].dst, ch.seg[i].src, ch.seg[i].bytes,
                                       &S.w_full[ws.index], policy);
            }
            ws.advance();
        }
    }
}

template <int F>
__global__ void __launch_bounds__(BLOCK_THREADS, 1) qwen_dps_kernel(const __grid_constant__ Params p) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem<F> &S = *reinterpret_cast<Smem<F> *>(smem_raw);
    const int warp = threadIdx.x >> 5;

    if (threadIdx.x == 0) {
        for (int i = 0; i < Smem<F>::W; ++i) {
            dps::bar_init(&S.w_full[i], 1);
            dps::bar_init(&S.w_empty[i], COMPUTE_WARPS);
        }
        Scheduler<F>::init_storage(&S.sched, COMPUTE_WARPS + 1);   // compute warps + load warp
        dps::bar_init_fence();
        S.c.x_tag = -1;
        for (int c = 0; c < C_NUM; ++c) S.c.dep_seen[c] = 0u;
    }
    __syncthreads();

    const dps::SchedulerParams sp{p.sync + C_TICKET * CSTRIDE, p.sync + C_EOS * CSTRIDE,
                                  p.total_tiles, p.sched_mode};
    Scheduler<F> sched(sp, &S.sched);

    if (warp == SCHED_WARP) {
        if ((threadIdx.x & 31) == 0) sched.run_producer();
    } else if (warp == LOAD_WARP) {
        load_loop<F>(S, p, sched);
    } else {
        compute_loop<F>(S, p, sched);
    }
}

__global__ void qwen_dps_probe(int *out) {
    out[0] = DPS_HAS_CLC;
    out[1] = DPS_REAL_MBARRIER;
}

// host
constexpr size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

struct WorkspaceLayout {
    size_t sync, hidden, h1, q, k, v, attn_out, act, attn_part, lm_part, total;
};

constexpr WorkspaceLayout workspace_layout() {
    WorkspaceLayout L{};
    size_t o = 0;
    L.sync      = o; o += align256(sizeof(unsigned) * C_NUM * CSTRIDE);
    L.hidden    = o; o += align256(sizeof(half) * H);
    L.h1        = o; o += align256(sizeof(half) * H);
    L.q         = o; o += align256(sizeof(half) * QS);
    L.k         = o; o += align256(sizeof(half) * NKV * HD);
    L.v         = o; o += align256(sizeof(half) * NKV * HD);
    L.attn_out  = o; o += align256(sizeof(half) * QS);
    L.act       = o; o += align256(sizeof(half) * I);
    L.attn_part = o; o += align256(sizeof(float) * NKV * ATTN_SPLITS * 2 * (HD + 2));
    L.lm_part   = o; o += align256(sizeof(LmPart) * T_LM);
    L.total     = o;
    return L;
}

struct DeviceCaps { int device = -1, num_sms = 0, cc_major = 0, clc = 0, real_mbar = 0, ctas_per_sm = 0, coop = 0; };

template <int F>
static cudaError_t query_caps(DeviceCaps &caps) {
    int dev;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    static DeviceCaps cache[64];
    if (dev < 64 && cache[dev].device == dev) { caps = cache[dev]; return cudaSuccess; }

    caps.device = dev;
    cudaDeviceGetAttribute(&caps.num_sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&caps.cc_major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&caps.coop, cudaDevAttrCooperativeLaunch, dev);
    err = cudaFuncSetAttribute(qwen_dps_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem<F>));
    if (err != cudaSuccess) return err;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&caps.ctas_per_sm, qwen_dps_kernel<F>, BLOCK_THREADS,
                                                        sizeof(Smem<F>));
    if (err != cudaSuccess) return err;

    int *d_probe = nullptr, h_probe[2] = {0, 0};
    if ((err = cudaMalloc(&d_probe, sizeof(h_probe))) != cudaSuccess) return err;
    qwen_dps_probe<<<1, 1>>>(d_probe);
    err = cudaMemcpy(h_probe, d_probe, sizeof(h_probe), cudaMemcpyDeviceToHost);
    cudaFree(d_probe);
    if (err != cudaSuccess) return err;
    caps.clc = (caps.cc_major >= 10) && h_probe[0];
    caps.real_mbar = h_probe[1];
    if (dev < 64) cache[dev] = caps;
    return cudaSuccess;
}

template <int F>
static cudaError_t info_impl(QwenDpsInfo *info) {
    DeviceCaps caps;
    cudaError_t err = query_caps<F>(caps);
    if (err != cudaSuccess) return err;
    info->num_sms           = caps.num_sms;
    info->clc_supported     = caps.clc;
    info->real_mbarrier     = caps.real_mbar;
    info->block_threads     = BLOCK_THREADS;
    info->smem_bytes        = (int)sizeof(Smem<F>);
    info->stage_bytes       = Smem<F>::SB;
    info->weight_stages     = Smem<F>::W;
    info->sched_stages      = Smem<F>::SST;
    info->ctas_per_sm       = caps.ctas_per_sm;
    info->tiles_per_step    = T_STEP;
    info->lm_tiles          = T_LM;
    info->max_seq_supported = ATTN_CHUNK * ATTN_SPLITS;
    info->trace_build       = DPS_TRACE;
    const int phase_tiles[5] = {T_QKV, T_ATTN, T_O, T_GU, T_D};
    for (int i = 0; i < 5; ++i) info->phase_tiles[i] = phase_tiles[i];
    return cudaSuccess;
}

template <int F>
static cudaError_t launch_impl(const QwenDpsLaunch *a, cudaStream_t stream, int *used_mode, long long *grid_ctas) {
    if (a->n_prompt < 1 || a->max_new < 1) return cudaErrorInvalidValue;
    const int last_pos = a->start_pos + a->n_prompt + a->max_new - 2;   // position of the last fed token
    if (a->start_pos < 0 || last_pos >= a->max_seq || last_pos >= ATTN_CHUNK * ATTN_SPLITS)
        return cudaErrorInvalidValue;

    DeviceCaps caps;
    cudaError_t err = query_caps<F>(caps);
    if (err != cudaSuccess) return err;
    if (caps.ctas_per_sm < 1) return cudaErrorInvalidConfiguration;

    if (a->trace && (!DPS_TRACE || a->trace_first < 0 || a->trace_count < 0)) return cudaErrorInvalidValue;

    int mode = a->sched_mode;
    if (mode == dps::kSchedAuto) mode = caps.clc ? dps::kSchedClc : dps::kSchedAtomic;
    if (mode == dps::kSchedClc && !caps.clc) return cudaErrorNotSupported;
    if (mode == dps::kSchedStatic && !caps.coop) return cudaErrorNotSupported;

    const long long n_pre = a->n_prompt - 1;
    const long long total = n_pre * T_STEP + (long long)a->max_new * (T_STEP + T_LM);
    if (total > 0x7fffffffLL) return cudaErrorInvalidValue;
    // CLC (and oneshot): one CTA per tile, like a non-persistent launch; running CTAs
    // steal the rest. Atomic and static: one persistent CTA per SM slot.
    const bool persistent = mode == dps::kSchedAtomic || mode == dps::kSchedStatic;
    const long long grid = persistent ? (long long)caps.num_sms * caps.ctas_per_sm : total;

    const WorkspaceLayout L = workspace_layout();
    uint8_t *ws = static_cast<uint8_t *>(a->workspace);
    Params p;
    p.embed      = static_cast<const half *>(a->embed);
    p.final_norm = static_cast<const half *>(a->final_norm);
    p.cos_t      = static_cast<const half *>(a->cos_table);
    p.sin_t      = static_cast<const half *>(a->sin_table);
    p.lm         = a->lm_head;
    p.layers     = a->layers;
    p.k_cache    = static_cast<half *>(a->k_cache);
    p.v_cache    = static_cast<half *>(a->v_cache);
    p.sync       = reinterpret_cast<unsigned *>(ws + L.sync);
    p.tokens     = a->tokens;
    p.output_log = a->output_log;
    p.hidden     = reinterpret_cast<half *>(ws + L.hidden);
    p.h1         = reinterpret_cast<half *>(ws + L.h1);
    p.q          = reinterpret_cast<half *>(ws + L.q);
    p.k          = reinterpret_cast<half *>(ws + L.k);
    p.v          = reinterpret_cast<half *>(ws + L.v);
    p.attn_out   = reinterpret_cast<half *>(ws + L.attn_out);
    p.act        = reinterpret_cast<half *>(ws + L.act);
    p.attn_part  = reinterpret_cast<float *>(ws + L.attn_part);
    p.lm_part    = reinterpret_cast<LmPart *>(ws + L.lm_part);
    p.n_pre       = (int)n_pre;
    p.total_tiles = (int)total;
    p.start_pos   = a->start_pos;
    p.max_seq     = a->max_seq;
    p.eos_token   = a->eos_token;
    p.sched_mode  = mode;
    p.attn_scale  = a->attn_scale;
    p.trace       = a->trace;
    p.trace_first = a->trace_first;
    p.trace_count = a->trace_count;

    if ((err = cudaMemsetAsync(p.sync, 0, sizeof(unsigned) * C_NUM * CSTRIDE, stream)) != cudaSuccess) return err;
    if (mode == dps::kSchedStatic) {
        // Static tickets are only deadlock-free if every CTA is resident at once.
        void *args[] = {&p};
        err = cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(qwen_dps_kernel<F>), dim3((unsigned)grid),
                                          dim3(BLOCK_THREADS), args, sizeof(Smem<F>), stream);
    } else {
        qwen_dps_kernel<F><<<(unsigned)grid, BLOCK_THREADS, sizeof(Smem<F>), stream>>>(p);
        err = cudaGetLastError();
    }
    if (used_mode) *used_mode = mode;
    if (grid_ctas) *grid_ctas = grid;
    return err;
}

}  // namespace qwen

extern "C" size_t qwen_dps_workspace_bytes() { return qwen::workspace_layout().total; }

extern "C" cudaError_t qwen_dps_info(int weight_format, QwenDpsInfo *info) {
    switch (weight_format) {
    case QWEN_DPS_FP16: return qwen::info_impl<QWEN_DPS_FP16>(info);
    case QWEN_DPS_FP8:  return qwen::info_impl<QWEN_DPS_FP8>(info);
    case QWEN_DPS_FP4:  return qwen::info_impl<QWEN_DPS_FP4>(info);
    default:            return cudaErrorInvalidValue;
    }
}

extern "C" cudaError_t qwen_dps_launch(const QwenDpsLaunch *a, cudaStream_t stream,
                                       int *used_mode, long long *grid_ctas) {
    switch (a->weight_format) {
    case QWEN_DPS_FP16: return qwen::launch_impl<QWEN_DPS_FP16>(a, stream, used_mode, grid_ctas);
    case QWEN_DPS_FP8:  return qwen::launch_impl<QWEN_DPS_FP8>(a, stream, used_mode, grid_ctas);
    case QWEN_DPS_FP4:  return qwen::launch_impl<QWEN_DPS_FP4>(a, stream, used_mode, grid_ctas);
    default:            return cudaErrorInvalidValue;
    }
}
