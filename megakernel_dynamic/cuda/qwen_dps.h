// qwen_dps.h — host-side interface of the dynamic-persistent Qwen3-0.6B megakernel.
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

// Weight formats (weight-only quantization; activations stay fp16). The block-scaled
// formats are the ones Blackwell's tensor cores use (see CUTLASS blockscaled_gemm):
//   FP8 = MXF8: e4m3 elements, one e8m0 (power-of-two) scale per 32 along K
//   FP4 = NVF4: e2m1 elements (2 per byte, low nibble first), one e4m3 scale per 16
//               along K, and an fp32 scale per tensor
enum QwenDpsWeightFormat : int { QWEN_DPS_FP16 = 0, QWEN_DPS_FP8 = 1, QWEN_DPS_FP4 = 2 };

// One projection matrix, row-major [rows, K] (rows = output features).
struct QwenDpsMatrix {
    const void *data;    // fp16 [rows, K] | e4m3 [rows, K] | packed e2m1 [rows, K/2]
    const void *scale;   // unused | e8m0 [rows, K/32] | e4m3 [rows, K/16]
    float       gscale;  // per-tensor scale (NVF4), 1.0 otherwise
    unsigned    pad;
};

// Per-layer weights, 200 bytes. Python packs this as 25 int64 slots per layer
// (gscale in the low 32 bits of its slot).
struct QwenDpsLayerWeights {
    const void   *input_layernorm;      // fp16 [1024]
    const void   *q_norm;               // fp16 [128]
    const void   *k_norm;               // fp16 [128]
    const void   *post_attn_layernorm;  // fp16 [1024]
    QwenDpsMatrix q, k, v;              // [2048|1024|1024, 1024]
    QwenDpsMatrix o;                    // [1024, 2048]
    QwenDpsMatrix gate, up;             // [3072, 1024]
    QwenDpsMatrix down;                 // [1024, 3072]
};
static_assert(sizeof(QwenDpsLayerWeights) == 200, "layer table layout changed");

// Per-tile timing record, written by DPS_TRACE builds (qwen_dps_trace_C) for tickets
// in [trace_first, trace_first + trace_count). Times are %globaltimer nanoseconds.
struct QwenDpsTraceRecord {
    unsigned long long claim;   // the scheduler warp took the ticket
    unsigned long long start;   // the compute warps received it
    unsigned long long ready;   // its dependency counters were satisfied
    unsigned long long end;     // the tile finished, including its done-signal
    unsigned int prep_ns;       // building the input vector (RMSNorm / copy from global)
    unsigned int wwait_ns;      // waiting for weight stages from the TMA warp
    unsigned int sm;            // %smid
    unsigned int cta;           // blockIdx.x
};
static_assert(sizeof(QwenDpsTraceRecord) == 48, "trace record layout changed");

struct QwenDpsLaunch {
    const void *embed;                      // [151936, 1024] fp16 (embedding lookup)
    const QwenDpsLayerWeights *layers;      // [28], device memory
    const void *final_norm;                 // [1024]
    QwenDpsMatrix lm_head;                  // [151936, 1024] in the weight format
    const void *cos_table, *sin_table;      // [max_seq, 128]
    void *k_cache, *v_cache;                // [28, 8, max_seq, 128]
    void *workspace;                        // qwen_dps_workspace_bytes() bytes
    int  *tokens;                           // [n_prompt + max_new]; prompt prefix filled by caller
    int  *output_log;                       // [max_new]; generated ids (EOS included)
    int   n_prompt;
    int   max_new;
    int   start_pos;                        // KV position of tokens[0]
    int   max_seq;
    int   eos_token;                        // -1 disables early stop
    int   sched_mode;                       // 0 auto, 1 atomic, 2 CLC, 3 oneshot (test)
    int   weight_format;                    // QwenDpsWeightFormat
    float attn_scale;
    QwenDpsTraceRecord *trace;              // nullptr = no tracing (needs a DPS_TRACE build)
    int   trace_first, trace_count;         // ticket range to record
};

struct QwenDpsInfo {
    int       num_sms;
    int       clc_supported;       // device is sm_100+ and the binary was built with CLC
    int       real_mbarrier;       // binary uses mbarrier tx / cp.async.bulk (sm_90+)
    int       block_threads;
    int       smem_bytes;
    int       stage_bytes;
    int       weight_stages;
    int       sched_stages;
    int       ctas_per_sm;
    int       tiles_per_step;      // without LM head
    int       lm_tiles;
    int       max_seq_supported;
    int       trace_build;         // built with DPS_TRACE=1
};

extern "C" size_t qwen_dps_workspace_bytes();
extern "C" cudaError_t qwen_dps_info(int weight_format, QwenDpsInfo *info);
// Returns the scheduler mode actually used (1 atomic, 2 CLC, 3 oneshot) and grid size.
extern "C" cudaError_t qwen_dps_launch(const QwenDpsLaunch *args, cudaStream_t stream,
                                       int *used_mode, long long *grid_ctas);
