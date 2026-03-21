#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "rmsnorm.cuh"
#include "swiglu.cuh"
#include "swiglu.cu"
namespace cg = cooperative_groups;

// Compile-time architecture guard — catch accidental wrong-arch builds
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "megakernel_5090.cu requires sm_80 or later. Use -arch=sm_120 for RTX 5090."
#endif


// Model constants

constexpr int WARP_SIZE       = 32;
constexpr int HIDDEN_SIZE     = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS     = 16;
constexpr int NUM_KV_HEADS    = 8;
constexpr int HEAD_DIM        = 128;
constexpr int Q_SIZE          = NUM_Q_HEADS  * HEAD_DIM;  // 2048
constexpr int KV_SIZE         = NUM_KV_HEADS * HEAD_DIM;  // 1024


#ifndef LDG_NUM_BLOCKS
#define LDG_NUM_BLOCKS 170
#endif

#ifndef LDG_BLOCK_SIZE
#define LDG_BLOCK_SIZE 256
#endif

#ifndef LDG_LM_NUM_BLOCKS
#define LDG_LM_NUM_BLOCKS 680
#endif

#ifndef LDG_LM_BLOCK_SIZE
#define LDG_LM_BLOCK_SIZE 256
#endif

#ifndef LDG_LM_ROWS_PER_WARP
#define LDG_LM_ROWS_PER_WARP 8
#endif


#ifndef LDG_ATTN_BLOCKS
#define LDG_ATTN_BLOCKS 16
#endif

#ifndef LDG_PREFETCH_QK
#define LDG_PREFETCH_QK 1
#endif

#ifndef LDG_PREFETCH_DOWN
#define LDG_PREFETCH_DOWN 1
#endif

#ifndef LDG_PREFETCH_THREAD_STRIDE
#define LDG_PREFETCH_THREAD_STRIDE 1
#endif

#ifndef LDG_PREFETCH_ELEM_STRIDE
#define LDG_PREFETCH_ELEM_STRIDE 1
#endif

#ifndef LDG_PREFETCH_BLOCK_STRIDE
#define LDG_PREFETCH_BLOCK_STRIDE 1
#endif

#ifndef LDG_PREFETCH_GATE
#define LDG_PREFETCH_GATE 1
#endif

#ifndef LDG_PREFETCH_UP
#define LDG_PREFETCH_UP 1
#endif

constexpr int   LDG_NUM_WARPS = LDG_BLOCK_SIZE / WARP_SIZE;
constexpr float LDG_RMS_EPS   = 1e-6f;
constexpr int   LDG_VOCAB_SIZE = 151936;


// Per-layer weight pointers
// Field order here is the ground truth; Python _pack_layer_weights must match.

struct __align__(16) LDGLayerWeight {
    const half* input_layernorm_weight;      // [HIDDEN_SIZE]
    const half* q_proj_weight;               // [Q_SIZE,          HIDDEN_SIZE]
    const half* k_proj_weight;               // [KV_SIZE,         HIDDEN_SIZE]
    const half* v_proj_weight;               // [KV_SIZE,         HIDDEN_SIZE]
    const half* q_norm_weight;               // [HEAD_DIM]
    const half* k_norm_weight;               // [HEAD_DIM]
    const half* o_proj_weight;               // [HIDDEN_SIZE,     Q_SIZE]
    const half* post_attn_layernorm_weight;  // [HIDDEN_SIZE]
    const half* gate_proj_weight;            // [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    const half* up_proj_weight;              // [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    const half* down_proj_weight;            // [HIDDEN_SIZE,     INTERMEDIATE_SIZE]
    const void* padding;                     // struct = 12 × 8 = 96 bytes, 16-byte aligned
};



struct AtomicGridSync {
    unsigned int *counter;
    unsigned int *generation;
    unsigned int  nblocks;
    unsigned int  local_gen;

    __device__ void sync() {
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int my_gen = local_gen;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1) {
                *counter = 0;
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                atomicAdd(generation, 1);
            } else {
                // A compact spin with asm("yield") yields the warp scheduler instead.
                volatile unsigned int *vgen = (volatile unsigned int *)generation;
                while (*vgen <= my_gen) {
                    asm volatile("nanosleep.u32 256;" ::: "memory");
                }
            }
            local_gen = my_gen + 1;
        }
        __syncthreads();
    }
};


// Small helpers

__device__ __forceinline__ float ldg_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

#define LOG2E_HALF __float2half(1.44269504088896340736f)

__device__ __forceinline__ __half ptx_hrcp(__half x) {
    return __float2half(1.0f / __half2float(x));
}
__device__ __forceinline__ __half ptx_hexp2(__half x) {
    return __float2half(exp2f(__half2float(x)));
}
__device__ __forceinline__ __half fast_exp(__half x) {
    return ptx_hexp2(__hmul(x, LOG2E_HALF));
}
__device__ __forceinline__ __half ldg_silu(__half x) {
    return __hmul(x, ptx_hrcp(__hadd(__float2half(1.0f), fast_exp(__hneg(x)))));
}

__device__ __forceinline__ uint2 ldg_load_weights_u2(const uint2 *ptr) {
    uint2 res;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];"
                 : "=r"(res.x), "=r"(res.y) : "l"(ptr));
    return res;
}
__device__ __forceinline__ uint4 ldg_load_weights_u4(const uint4 *ptr) {
    uint4 res;

    // sm_120 (Blackwell): 64B cache line, evict-first hint avoids L2 thrashing for streaming weight reads that won't be reused.
    asm volatile(
        "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
        : "l"(ptr));
    return res;
}


// device_rmsnorm_step  (utility, used when the fused path is needed)
__device__ __forceinline__ void device_rmsnorm_step(
    half       *s_norm_out,
    const half *input,
    half       *residual,
    const uint2 *weight,
    float       eps,
    int         /*block_id — no longer used*/)
{
    int tx = threadIdx.x;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    __shared__ float s_rms_inv;
    __shared__ float shared_reduce[LDG_BLOCK_SIZE / WARP_SIZE];

    float thread_sum_sq = 0.0f;

    #pragma unroll
    for (int i = tx; i < HIDDEN_SIZE / 4; i += LDG_BLOCK_SIZE) {
        uint2 v_in  = reinterpret_cast<const uint2*>(input)[i];
        uint2 v_res = reinterpret_cast<uint2*>(residual)[i];

        half2 *h2_in  = reinterpret_cast<half2*>(&v_in);
        half2 *h2_res = reinterpret_cast<half2*>(&v_res);

        h2_res[0] = __hadd2(h2_in[0], h2_res[0]);
        h2_res[1] = __hadd2(h2_in[1], h2_res[1]);

        reinterpret_cast<uint2*>(residual)[i] = v_res;
        reinterpret_cast<uint2*>(s_norm_out)[i] = v_res;

        float2 f0 = __half22float2(h2_res[0]);
        float2 f1 = __half22float2(h2_res[1]);
        thread_sum_sq += f0.x*f0.x + f0.y*f0.y + f1.x*f1.x + f1.y*f1.y;
    }

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());
    if (warp.thread_rank() == 0) shared_reduce[tx / 32] = warp_sum;
    block.sync();

    float block_sum = 0.0f;
    if (tx < LDG_BLOCK_SIZE / WARP_SIZE) block_sum = shared_reduce[tx];
    block_sum = cg::reduce(warp, block_sum, cg::plus<float>());

    if (tx == 0) s_rms_inv = rsqrtf(block_sum / (float)HIDDEN_SIZE + eps);
    block.sync();

    float inv_rms = s_rms_inv;
    #pragma unroll
    for (int i = tx; i < HIDDEN_SIZE / 4; i += LDG_BLOCK_SIZE) {
        uint2 v_val    = reinterpret_cast<uint2*>(s_norm_out)[i];
        uint2 v_weight = weight[i];
        half2 *h2_val  = reinterpret_cast<half2*>(&v_val);
        half2 *h2_w    = reinterpret_cast<half2*>(&v_weight);

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            float2 f_v = __half22float2(h2_val[j]);
            float2 f_w = __half22float2(h2_w[j]);
            f_v.x *= inv_rms * f_w.x;
            f_v.y *= inv_rms * f_w.y;
            h2_val[j] = __float22half2_rn(f_v);
        }
        reinterpret_cast<uint2*>(s_norm_out)[i] = v_val;
    }
    block.sync();
}


// QKV fused matrix-vector product
__device__ void ldg_matvec_qkv_fp16(
    AtomicGridSync &grid,
    half       *s_norm,
    const half *q_weight, const half *k_weight, const half *v_weight,
    half *q_out, half *k_out, half *v_out)
{
    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + gridDim.x - 1) / gridDim.x;
    int row_start      = blockIdx.x * rows_per_block;
    int row_end        = min(row_start + rows_per_block, TOTAL_ROWS);

    for (int m = row_start + warp_id; m < row_end; m += (LDG_BLOCK_SIZE / 32)) {
        const half *weight_row;
        half       *out_ptr;

        if (m < Q_SIZE) {
            weight_row = q_weight + (long long)m * HIDDEN_SIZE;
            out_ptr    = q_out + m;
        } else if (m < Q_SIZE + KV_SIZE) {
            int local = m - Q_SIZE;
            weight_row = k_weight + (long long)local * HIDDEN_SIZE;
            out_ptr    = k_out + local;
        } else {
            int local = m - Q_SIZE - KV_SIZE;
            weight_row = v_weight + (long long)local * HIDDEN_SIZE;
            out_ptr    = v_out + local;
        }

        float sum = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += 32 * 8) {
            uint4 w_u4 = ldg_load_weights_u4(reinterpret_cast<const uint4*>(weight_row + k));
            uint4 a_u4 = *reinterpret_cast<uint4*>(s_norm + k);

            half2 *w_h2 = reinterpret_cast<half2*>(&w_u4);
            half2 *a_h2 = reinterpret_cast<half2*>(&a_u4);

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 fw = __half22float2(w_h2[j]);
                float2 fa = __half22float2(a_h2[j]);
                sum += fw.x*fa.x + fw.y*fa.y;
            }
        }
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0) *out_ptr = __float2half(sum);
    }
}


// L2 prefetch helper

__device__ void ldg_prefetch_weights_l2(const half *weights, int num_elements) {
    int num_vec = num_elements / 8;
    for (int i = threadIdx.x; i < num_vec; i += LDG_BLOCK_SIZE) {
        uint4 dummy = ldg_load_weights_u4(reinterpret_cast<const uint4*>(weights) + i);
        // Clobber registers so the compiler cannot eliminate the load
        asm volatile("" : : "r"(dummy.x), "r"(dummy.y), "r"(dummy.z), "r"(dummy.w) : "memory");
    }
}


// Attention: QK-norm, RoPE, KV-cache write, online-softmax attention


__device__ void ldg_attention(
    AtomicGridSync &grid,
    half       *q,
    half       *k,
    const half *v,
    half       *k_cache,
    half       *v_cache,
    half       *attn_out,
    int         cache_len,
    int         max_seq_len,
    float       attn_scale,
    const half *q_norm_weight,
    const half *k_norm_weight,
    const half *cos_table,
    const half *sin_table,
    int         position,
    // Weights for L2 prefetch (idle blocks)
    const half *o_w,
    const half *g_w,
    const half *u_w,
    const half *d_w)
{
    if (position >= max_seq_len) {
        grid.sync();
        return;
    }

    int block_id = blockIdx.x;
    int warp_id  = threadIdx.x / WARP_SIZE;
    int lane_id  = threadIdx.x % WARP_SIZE;

    const half *cos_pos = cos_table + position * HEAD_DIM;
    const half *sin_pos = sin_table + position * HEAD_DIM;

        // K-norm + RoPE + KV cache write  (block 0 only)
        if (block_id == 0) {
        for (int h = warp_id; h < NUM_KV_HEADS; h += LDG_NUM_WARPS) {
            half *k_ptr  = k + h * HEAD_DIM;
            half *kc_ptr = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
            half *vc_ptr = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

            // RMSNorm over K head
            float ss = 0.0f;
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
                float val = __half2float(k_ptr[i]);
                ss += val * val;
            }
            ss = ldg_warp_reduce_sum(ss);
            float inv_rms = rsqrtf(ss / (float)HEAD_DIM + 1e-6f);
            inv_rms = __shfl_sync(0xffffffff, inv_rms, 0);

            // Gather the 4 elements each thread owns (stride WARP_SIZE)
            float k_vals[4];
            #pragma unroll
            for (int iter = 0; iter < 4; iter++) {
                int i = lane_id + iter * WARP_SIZE;
                k_vals[iter] = __half2float(k_ptr[i]) * inv_rms
                             * __half2float(k_norm_weight[i]);
            }

            // Both must use the same cos/sin frequency, i.e. rope_idx = i % (HEAD_DIM/2).
            #pragma unroll
            for (int iter = 0; iter < 4; iter++) {
                int i        = lane_id + iter * WARP_SIZE;
                int rope_idx = (iter < 2) ? i : (i - HEAD_DIM / 2); 
                float c = __half2float(cos_pos[rope_idx]);
                float s = __half2float(sin_pos[rope_idx]);

                // Pairs: (vals[0], vals[2]) and (vals[1], vals[3])
                float x = (iter < 2) ? k_vals[iter]     : k_vals[iter - 2];
                float y = (iter < 2) ? k_vals[iter + 2] : k_vals[iter];
                float k_rot = (iter < 2) ? (x*c - y*s) : (x*s + y*c);
                kc_ptr[i] = __float2half(k_rot);
            }

            // V cache — no rotation
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
                vc_ptr[i] = v[h * HEAD_DIM + i];
        }
        __threadfence();
    }

        // Q-norm + RoPE  (one block per Q head — each block handles warp 0)
    if (block_id < LDG_ATTN_BLOCKS && warp_id == 0) {
        int heads_per_block = (NUM_Q_HEADS + LDG_ATTN_BLOCKS - 1) / LDG_ATTN_BLOCKS;
        int q_start = block_id * heads_per_block;
        int q_end   = min(q_start + heads_per_block, NUM_Q_HEADS);

        for (int qh = q_start; qh < q_end; qh++) {
            half *q_ptr = q + qh * HEAD_DIM;

            // RMSNorm
            float ss = 0.0f;
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
                float val = __half2float(q_ptr[i]);
                ss += val * val;
            }
            ss = ldg_warp_reduce_sum(ss);
            float inv_rms = rsqrtf(ss / (float)HEAD_DIM + 1e-6f);
            inv_rms = __shfl_sync(0xffffffff, inv_rms, 0);

            float q_vals[4];
            #pragma unroll
            for (int iter = 0; iter < 4; iter++) {
                int i = lane_id + iter * WARP_SIZE;
                q_vals[iter] = __half2float(q_ptr[i]) * inv_rms
                             * __half2float(q_norm_weight[i]);
            }

            #pragma unroll
            for (int iter = 0; iter < 4; iter++) {
                int i        = lane_id + iter * WARP_SIZE;
                int rope_idx = (iter < 2) ? i : (i - HEAD_DIM / 2);
                float c = __half2float(cos_pos[rope_idx]);
                float s = __half2float(sin_pos[rope_idx]);

                float x = (iter < 2) ? q_vals[iter]     : q_vals[iter - 2];
                float y = (iter < 2) ? q_vals[iter + 2] : q_vals[iter];
                float q_rot = (iter < 2) ? (x*c - y*s) : (x*s + y*c);
                q_ptr[i] = __float2half(q_rot);
            }
        }
    }

    // prefetch weights into L2 while block 0 / attn blocks work
    if (block_id >= LDG_ATTN_BLOCKS) {
        int prefetch_id         = block_id - LDG_ATTN_BLOCKS;
        int num_prefetch_blocks = LDG_NUM_BLOCKS - LDG_ATTN_BLOCKS;
        int total_elements      = HIDDEN_SIZE * Q_SIZE + INTERMEDIATE_SIZE * HIDDEN_SIZE * 3;
        int per_block = (total_elements + num_prefetch_blocks - 1) / num_prefetch_blocks;
        int start = prefetch_id * per_block;
        int end   = min(start + per_block, total_elements);

        // Walk using vec4 loads (8 halfs = 16 bytes per load)
        int vec_start = start / 8;
        int vec_end   = end   / 8;
        for (int i = vec_start + threadIdx.x; i < vec_end; i += LDG_BLOCK_SIZE) {
            int elem = i * 8;
            const uint4 *ptr = (elem < HIDDEN_SIZE * Q_SIZE)
                ? (reinterpret_cast<const uint4*>(o_w) + i)
                : (reinterpret_cast<const uint4*>(g_w) + (i - HIDDEN_SIZE * Q_SIZE / 8));
            uint4 dummy = ldg_load_weights_u4(ptr);
            asm volatile("" : : "r"(dummy.x), "r"(dummy.y), "r"(dummy.z), "r"(dummy.w) : "memory");
        }
    }

    grid.sync();

    // Online-softmax attention  (one block per Q head)

    // Shared memory layout for the warp-level partial results.
    // HEAD_DIM entries per warp for the value accumulator.
    __shared__ __align__(16) float s_max_score[LDG_NUM_WARPS];
    __shared__ __align__(16) float s_sum_exp[LDG_NUM_WARPS];
    __shared__ __align__(16) float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

    // Initialise shared memory
    s_max_score[warp_id] = -INFINITY;
    s_sum_exp[warp_id]   = 0.0f;
    for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
        s_out_acc[warp_id][i] = 0.0f;
    __syncthreads();

    if (block_id < LDG_ATTN_BLOCKS) {
        int heads_per_block = (NUM_Q_HEADS + LDG_ATTN_BLOCKS - 1) / LDG_ATTN_BLOCKS;
        int q_start = block_id * heads_per_block;
        int q_end   = min(q_start + heads_per_block, NUM_Q_HEADS);

        for (int qh = q_start; qh < q_end; qh++) {
            half *q_head = q + qh * HEAD_DIM;
            int   kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);

            // ---- Per-warp partial online softmax ----
            float max_score = -INFINITY;
            float sum_exp   = 0.0f;
            float acc[4]    = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int t = warp_id; t < cache_len; t += LDG_NUM_WARPS) {
                const half *kc = k_cache + kv_head * max_seq_len * HEAD_DIM + t * HEAD_DIM;

                float score = 0.0f;
                #pragma unroll
                for (int i = lane_id; i < HEAD_DIM; i += 32)
                    score += __half2float(q_head[i]) * __half2float(kc[i]);
                score = ldg_warp_reduce_sum(score) * attn_scale;
                score = __shfl_sync(0xffffffff, score, 0);

                float old_max = max_score;
                max_score = fmaxf(max_score, score);
                float e     = expf(score   - max_score);
                float e_old = expf(old_max - max_score);
                sum_exp = sum_exp * e_old + e;

                const half *vc = v_cache + kv_head * max_seq_len * HEAD_DIM + t * HEAD_DIM;
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    acc[i] = acc[i] * e_old + e * __half2float(vc[lane_id + i * 32]);
            }

            // Store partial results for this warp.
            if (lane_id == 0) {
                s_max_score[warp_id] = max_score;
                s_sum_exp[warp_id]   = sum_exp;
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                s_out_acc[warp_id][lane_id + i * 32] = acc[i];
            __syncthreads();

            if (warp_id == 0) {
                int nw = min(LDG_NUM_WARPS, (cache_len + LDG_NUM_WARPS - 1));  // missing / LDG_NUM_WARPS
                nw = min(nw, LDG_NUM_WARPS);

                // Find global maximum across all warps.
                float g_max = -INFINITY;
                for (int w = 0; w < LDG_NUM_WARPS; w++) {
                    if (s_max_score[w] > g_max) g_max = s_max_score[w];
                }

                // Rescale each warp's (sum_exp, acc) to the global max, then add.
                float g_sum = 0.0f;
                float g_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int w = 0; w < LDG_NUM_WARPS; w++) {
                    if (s_max_score[w] == -INFINITY) continue;
                    float rescale = expf(s_max_score[w] - g_max);
                    g_sum += s_sum_exp[w] * rescale;
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                        g_acc[i] += s_out_acc[w][lane_id + i * 32] * rescale;
                }

                // Write normalised output.
                half *out_head = attn_out + qh * HEAD_DIM;
                float inv_sum  = (g_sum > 0.0f) ? (1.0f / g_sum) : 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    out_head[lane_id + i * 32] = __float2half(g_acc[i] * inv_sum);
            }
            __syncthreads();
        }
    }
    grid.sync();
}


// O-projection + Post-attention RMSNorm + SwiGLU MLP

__device__ void ldg_o_proj_postnorm_mlp(
    AtomicGridSync       &grid,
    const half *__restrict__ o_weight,
    const half *__restrict__ post_norm_weight,
    const half *__restrict__ gate_weight,
    const half *__restrict__ up_weight,
    const half *__restrict__ down_weight,
    const half *__restrict__ attn_out,
    const half *__restrict__ hidden_residual,   // FIX 3: half* not float*
    float      *__restrict__ g_activations,
    float      *__restrict__ g_mlp_intermediate,
    float      *__restrict__ g_norm_scratch,    // FIX 5: 1-float global scratch
    half       *__restrict__ hidden_out)      // final output written as half
{
    int block_id  = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id   = threadIdx.x / WARP_SIZE;
    int lane_id   = threadIdx.x % WARP_SIZE;

    // Shared memory buffers
    __shared__ __align__(16) half s_attn[Q_SIZE];
    __shared__ __align__(16) half s_act[HIDDEN_SIZE];
    // g_mlp_intermediate is float32; we load it with __ldg() (cached, read-only).

    for (int i = threadIdx.x; i < Q_SIZE; i += LDG_BLOCK_SIZE)
        s_attn[i] = attn_out[i];
    __syncthreads();

    // O-projection: hidden[m] = (o_weight[m,:] · attn_out) + residual[m]
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start     = block_id * hid_per_block;
    int hid_end       = min(hid_start + hid_per_block, HIDDEN_SIZE);

    for (int m = hid_start + warp_id; m < hid_end; m += LDG_NUM_WARPS) {
        float sum = 0.0f;
        const uint4 *o_row = reinterpret_cast<const uint4*>(o_weight + m * Q_SIZE);

        #pragma unroll 4
        for (int k = lane_id; k < Q_SIZE / 8; k += WARP_SIZE) {
            uint4  w_u4 = ldg_load_weights_u4(o_row + k);
            uint4  a_u4 = *reinterpret_cast<const uint4*>(s_attn + k * 8);
            half2 *wh   = reinterpret_cast<half2*>(&w_u4);
            half2 *ah   = reinterpret_cast<half2*>(&a_u4);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 fw = __half22float2(wh[i]);
                float2 fa = __half22float2(ah[i]);
                sum += fw.x*fa.x + fw.y*fa.y;
            }
        }
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0)
            g_activations[m] = sum + __half2float(hidden_residual[m]);  // fuse residual add
    }

    grid.sync();

    // Post-attention RMSNorm (replicate block 0 across the rest of the block to reduce redundancy)
    {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
            s_act[i] = __float2half(g_activations[i]);
        if(block_id == 0){
            __shared__ float s_norm_sq;
            if(threadIdx.x == 0) s_norm_sq = 0.0f;
            __syncthreads();
            float local_ss = 0.0f;
            for(int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE){
                float val = g_activations[i];
                local_ss += val* val;
            }
            local_ss = ldg_warp_reduce_sum(local_ss);
            if(lane_id == 0 ) atomicAdd(&s_norm_sq,local_ss);
            __syncthreads();
            if(threadIdx.x == 0)
                *g_norm_scratch = rsqrtf(s_norm_sq / (float)HIDDEN_SIZE + LDG_RMS_EPS);
        }
        grid.sync();
        float rstd = *g_norm_scratch;
        for(int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE){
            float w = __half2float(post_norm_weight[i]);
            s_act[i] = __float2half(__half2float(s_act[i]) * rstd * w);
        }
        __syncthreads();
    }

    // SwiGLU: gate_proj + up_proj → SiLU(gate) * up → g_mlp_intermediate
    int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int int_start     = block_id * int_per_block;
    int int_end       = min(int_start + int_per_block, INTERMEDIATE_SIZE);

    for (int m = int_start + warp_id; m < int_end; m += LDG_NUM_WARPS) {
        float gate_sum = 0.0f, up_sum = 0.0f;
        const uint4 *g_row = reinterpret_cast<const uint4*>(gate_weight + m * HIDDEN_SIZE);
        const uint4 *u_row = reinterpret_cast<const uint4*>(up_weight   + m * HIDDEN_SIZE);

        #pragma unroll 4
        for (int k = lane_id; k < HIDDEN_SIZE / 8; k += WARP_SIZE) {
            uint4  g_u4 = ldg_load_weights_u4(g_row + k);
            uint4  u_u4 = ldg_load_weights_u4(u_row + k);
            uint4  a_u4 = *reinterpret_cast<const uint4*>(s_act + k * 8);
            half2 *gh   = reinterpret_cast<half2*>(&g_u4);
            half2 *uh   = reinterpret_cast<half2*>(&u_u4);
            half2 *ah   = reinterpret_cast<half2*>(&a_u4);

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 fa = __half22float2(ah[i]);
                gate_sum += __half2float(gh[i].x)*fa.x + __half2float(gh[i].y)*fa.y;
                up_sum   += __half2float(uh[i].x)*fa.x + __half2float(uh[i].y)*fa.y;
            }
        }
        gate_sum = ldg_warp_reduce_sum(gate_sum);
        up_sum   = ldg_warp_reduce_sum(up_sum);

        if (lane_id == 0) {
            float activated = __half2float(ldg_silu(__float2half(gate_sum))) * up_sum;
            g_mlp_intermediate[m] = activated;
        }
    }

    grid.sync();

    // Down projection: hidden_out[m] = down_weight[m,:] · mlp_inter + o_proj_residual
    for (int m = hid_start + warp_id; m < hid_end; m += LDG_NUM_WARPS) {
        float sum = 0.0f;
        const uint4 *d_row = reinterpret_cast<const uint4*>(down_weight + m * INTERMEDIATE_SIZE);

        #pragma unroll 4
        for (int k = lane_id; k < INTERMEDIATE_SIZE / 8; k += WARP_SIZE) {
            uint4  d_u4 = ldg_load_weights_u4(d_row + k);
            half2 *dh   = reinterpret_cast<half2*>(&d_u4);
            // int    base = k * 8;
            // Use __ldg for read-only cached loads from global memory
            float4 mi0 = *reinterpret_cast<const float4*>(g_mlp_intermediate + k * 8);
            float4 mi1 = *reinterpret_cast<const float4*>(g_mlp_intermediate + k * 8 + 4);

            float2 d0 = __half22float2(dh[0]);
            float2 d1 = __half22float2(dh[1]);
            float2 d2 = __half22float2(dh[2]);
            float2 d3 = __half22float2(dh[3]);
            sum += d0.x*mi0.x + d0.y*mi0.y
                 + d1.x*mi0.z + d1.y*mi0.w
                 + d2.x*mi1.x + d2.y*mi1.y
                 + d3.x*mi1.z + d3.y*mi1.w;
        }
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0)
            hidden_out[m] = __float2half(sum + g_activations[m]);
    }
    grid.sync();
}


// Global barrier / persistent-kernel state

static unsigned int *d_barrier_counter = nullptr;
static unsigned int *d_barrier_sense   = nullptr;
static unsigned int *d_kv_flag         = nullptr;
static unsigned int *d_attn_flag       = nullptr;
static int          *d_mutable_position = nullptr;
static int          *d_mutable_token_id = nullptr;
static int          *h_pinned_position  = nullptr;
static int          *h_pinned_token_id  = nullptr;

// Prefill statics: allocated once on first prefill call.
static int   *d_eos_flag     = nullptr;
static float *d_norm_scratch = nullptr;

static int *d_prefill_dummy_log    = nullptr;
static int *d_prefill_step_counter = nullptr;
static int *d_prefill_output_token = nullptr;
static int *d_prefill_token_ids    = nullptr;
static int  d_prefill_token_ids_cap = 0;

static int *h_pinned_positions     = nullptr;
static int  h_pinned_positions_cap = 0;


static void ensure_barrier_alloc() {
    if (d_barrier_counter) return;
    cudaMalloc(&d_barrier_counter,  sizeof(unsigned int));
    cudaMalloc(&d_barrier_sense,    sizeof(unsigned int));
    cudaMalloc(&d_kv_flag,          sizeof(unsigned int));
    cudaMalloc(&d_attn_flag,        sizeof(unsigned int));
    cudaMalloc(&d_mutable_position, sizeof(int));
    cudaMalloc(&d_mutable_token_id, sizeof(int));
    cudaHostAlloc(&h_pinned_position, sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_pinned_token_id, sizeof(int), cudaHostAllocDefault);
    cudaMemset(d_barrier_counter, 0, sizeof(unsigned int));
    cudaMemset(d_barrier_sense,   0, sizeof(unsigned int));
    cudaMemset(d_kv_flag,         0, sizeof(unsigned int));
    cudaMemset(d_attn_flag,       0, sizeof(unsigned int));

    cudaMalloc(&d_eos_flag, sizeof(int));
    cudaMemset(d_eos_flag, 0, sizeof(int));

    cudaMalloc(&d_norm_scratch, sizeof(float));
    cudaMemset(d_norm_scratch, 0, sizeof(float));
}


// Device step-update kernel (used by generate_nosync)

__global__ void ldg_update_step(
    const int *__restrict__ lm_output,
    int       *__restrict__ d_token_id,
    int       *__restrict__ d_position,
    int       *__restrict__ output_log,
    int       *__restrict__ d_step_counter,
    int       *__restrict__ d_eos_flag,   // set to 1 when EOS is generated
    int                     eos_token_id)
{
    if(*d_eos_flag) return;

    int tok  = *lm_output;
    int step = *d_step_counter;
    *d_token_id       = tok;
    *d_position       = *d_position + 1;
    output_log[step]  = tok;
    *d_step_counter   = step + 1;
    if (tok == eos_token_id) *d_eos_flag = 1;
}

// Forward declaration (kernel attributes helper defined after all kernels)
static inline void ldg_configure_kernel_attributes();


// Shared decode body — called by both the direct and persistent wrappers

__device__ void ldg_decode_body(
    const half *embed_weight,
    const LDGLayerWeight *layer_weights,
    const half *final_norm_weight,
    const half *cos_table,
    const half *sin_table,
    half       *k_cache,    half *v_cache,
    half       *hidden_buffer,
    float      *g_activations,
    half       *g_q, half *g_k, half *g_v,
    half       *g_attn_out,
    float      *g_mlp_intermediate,
    float      *g_normalized,
    float      *g_norm_scratch,   // FIX 5: norm scalar scratch
    int        *d_eos_flag,       // FIX 1: early-exit check
    int         num_layers,
    int         position,
    int         input_token_id,
    int         max_seq_len,
    float       attn_scale,
    AtomicGridSync &grid)
{
    if(*d_eos_flag) return;
    
    int tid     = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // Embedding lookup (block 0 writes; all blocks read after grid.sync)
    if (blockIdx.x == 0) {
        const half *embed_row = embed_weight + (long long)input_token_id * HIDDEN_SIZE;
        for (int i = tid; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
            hidden_buffer[i] = embed_row[i];
    }
    grid.sync();

    __shared__ __align__(16) half s_norm[HIDDEN_SIZE];

    // Transformer layers
    for (int layer = 0; layer < num_layers; layer++) {
        const LDGLayerWeight &lw = layer_weights[layer];

        // Input RMSNorm
        {
            cg::thread_block blk  = cg::this_thread_block();
            cg::thread_block_tile<32> warp = cg::tiled_partition<32>(blk);
            __shared__ float s_rms_inv;
            __shared__ __align__(16) float s_reduce[LDG_BLOCK_SIZE / WARP_SIZE];

            float ss = 0.0f;
            for (int i = tid; i < HIDDEN_SIZE / 4; i += LDG_BLOCK_SIZE) {
                uint2 v = reinterpret_cast<const uint2*>(hidden_buffer)[i];
                half2 *h2_src = reinterpret_cast<half2*>(&v);
                reinterpret_cast<half2*>(s_norm)[i*2]   = h2_src[0];
                reinterpret_cast<half2*>(s_norm)[i*2+1] = h2_src[1];
                float2 a = __half22float2(h2_src[0]), b = __half22float2(h2_src[1]);
                ss += a.x*a.x + a.y*a.y + b.x*b.x + b.y*b.y;
            }
            float ws = cg::reduce(warp, ss, cg::plus<float>());
            if (warp.thread_rank() == 0) s_reduce[tid/32] = ws;
            blk.sync();
            float bs = (tid < LDG_BLOCK_SIZE/WARP_SIZE) ? s_reduce[tid] : 0.0f;
            bs = cg::reduce(warp, bs, cg::plus<float>());
            if (tid == 0) s_rms_inv = rsqrtf(bs / (float)HIDDEN_SIZE + LDG_RMS_EPS);
            blk.sync();

            float inv_rms = s_rms_inv;
            const uint2 *wptr = reinterpret_cast<const uint2*>(lw.input_layernorm_weight);
            for (int i = tid; i < HIDDEN_SIZE/4; i += LDG_BLOCK_SIZE) {
                half2 *s_norm_h2 = reinterpret_cast<half2*>(s_norm);
                uint2  v_weight  = wptr[i];
                half2 *h2_w      = reinterpret_cast<half2*>(&v_weight);
                for (int j = 0; j < 2; j++) {
                    int   idx = i*2 + j;
                    float2 fv = __half22float2(s_norm_h2[idx]);
                    float2 fw = __half22float2(h2_w[j]);
                    fv.x *= inv_rms * fw.x;
                    fv.y *= inv_rms * fw.y;
                    s_norm_h2[idx] = __float22half2_rn(fv);
                }
            }
            blk.sync();
        }
        grid.sync();

        // QKV projection
        ldg_matvec_qkv_fp16(
            grid, s_norm,
            lw.q_proj_weight, lw.k_proj_weight, lw.v_proj_weight,
            g_q, g_k, g_v);
        grid.sync();

        // Attention
        half *lkc = k_cache + (long long)layer * NUM_KV_HEADS * max_seq_len * HEAD_DIM;
        half *lvc = v_cache + (long long)layer * NUM_KV_HEADS * max_seq_len * HEAD_DIM;
        ldg_attention(
            grid, g_q, g_k, g_v, lkc, lvc, g_attn_out,
            position + 1, max_seq_len, attn_scale,
            lw.q_norm_weight, lw.k_norm_weight,
            cos_table, sin_table, position,
            lw.o_proj_weight, lw.gate_proj_weight,
            lw.up_proj_weight, lw.down_proj_weight);

        // Prepare residual for o-proj
        // for (int i = tid; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
        //     g_residual[i] = __half2float(hidden_buffer[i]);
        // grid.sync();

        // O-proj + Post-norm + MLP
        ldg_o_proj_postnorm_mlp(
            grid,
            lw.o_proj_weight,
            lw.post_attn_layernorm_weight,
            lw.gate_proj_weight,
            lw.up_proj_weight,
            lw.down_proj_weight,
            g_attn_out,
            hidden_buffer,    // FIX 3: half* residual, direct — no float intermediate
            g_activations,
            g_mlp_intermediate,
            g_norm_scratch,   // FIX 5: norm scalar scratch
            hidden_buffer);
    }

    // 3. Final RMSNorm → g_normalized (float32 for LM head)
    {
        __shared__ float s_inv_rms;
        __shared__ float s_wss[LDG_BLOCK_SIZE / WARP_SIZE];

        float ss = 0.0f;
        for (int i = tid; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float val = __half2float(hidden_buffer[i]);
            ss += val * val;
        }
        ss = ldg_warp_reduce_sum(ss);
        if (lane_id == 0) s_wss[warp_id] = ss;
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane_id < LDG_BLOCK_SIZE/WARP_SIZE) ? s_wss[lane_id] : 0.0f;
            v = ldg_warp_reduce_sum(v);
            if (lane_id == 0) s_inv_rms = rsqrtf(v / (float)HIDDEN_SIZE + LDG_RMS_EPS);
        }
        __syncthreads();

        float inv_rms = s_inv_rms;
        for (int i = tid; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float w      = __half2float(final_norm_weight[i]);
            g_normalized[i] = __half2float(hidden_buffer[i]) * inv_rms * w;
        }
    }
    grid.sync();
}


// Kernel wrappers

__global__ void __launch_bounds__(LDG_BLOCK_SIZE) ldg_decode_kernel_direct(
    const half           *embed_weight,
    const LDGLayerWeight *layer_weights,
    const half           *final_norm_weight,
    const half           *cos_table,
    const half           *sin_table,
    half *k_cache, half *v_cache,
    half *hidden_buffer,
    float *g_activations, float *g_residual,   // g_residual kept for ABI compat
    half *g_q, half *g_k, half *g_v,
    half *g_attn_out, float *g_mlp_intermediate, float *g_normalized,
    float *g_norm_scratch,
    unsigned int *barrier_counter, unsigned int *barrier_sense,
    unsigned int *kv_flag, unsigned int *attn_flag,
    int *d_eos_flag,
    int num_layers, int position, int input_token_id,
    int max_seq_len, float attn_scale)
{
    AtomicGridSync grid;
    grid.counter    = barrier_counter;
    grid.generation = barrier_sense;
    grid.nblocks    = gridDim.x;
    grid.local_gen  = *barrier_sense;
    ldg_decode_body(
        embed_weight, layer_weights, final_norm_weight,
        cos_table, sin_table,
        k_cache, v_cache, hidden_buffer,
        g_activations,
        g_q, g_k, g_v, g_attn_out, g_mlp_intermediate, g_normalized,
        g_norm_scratch, d_eos_flag,
        num_layers, position, input_token_id, max_seq_len, attn_scale,
        grid);
}

__global__ void __launch_bounds__(LDG_BLOCK_SIZE) ldg_decode_kernel_persistent(
    const half           *embed_weight,
    const LDGLayerWeight *layer_weights,
    const half           *final_norm_weight,
    const half           *cos_table,
    const half           *sin_table,
    half *k_cache, half *v_cache,
    half *hidden_buffer,
    float *g_activations, float *g_residual,   // g_residual kept for ABI compat
    half *g_q, half *g_k, half *g_v,
    half *g_attn_out, float *g_mlp_intermediate, float *g_normalized,
    float *g_norm_scratch,
    unsigned int *barrier_counter, unsigned int *barrier_sense,
    unsigned int *kv_flag, unsigned int *attn_flag,
    int *d_eos_flag,
    int num_layers, const int *d_position, const int *d_token_id,
    int max_seq_len, float attn_scale)
{
    AtomicGridSync grid;
    grid.counter    = barrier_counter;
    grid.generation = barrier_sense;
    grid.nblocks    = gridDim.x;
    grid.local_gen  = *barrier_sense;
    ldg_decode_body(
        embed_weight, layer_weights, final_norm_weight,
        cos_table, sin_table,
        k_cache, v_cache, hidden_buffer,
        g_activations,
        g_q, g_k, g_v, g_attn_out, g_mlp_intermediate, g_normalized,
        g_norm_scratch, d_eos_flag,
        num_layers, *d_position, *d_token_id, max_seq_len, attn_scale,
        grid);
}


// LM head — Phase 1: distributed vocab projection + per-block argmax

__global__ void __launch_bounds__(LDG_LM_BLOCK_SIZE, 1) ldg_lm_head_phase1(
    const float *__restrict__ normalized,
    const half  *__restrict__ weight,
    float       *__restrict__ block_max_vals,
    int         *__restrict__ block_max_idxs)
{
    __shared__ __align__(128) float s_hidden[HIDDEN_SIZE];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE)
        s_hidden[i] = normalized[i];
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start      = blockIdx.x * rows_per_block;
    int row_end        = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

    float local_max     = -INFINITY;
    int   local_max_idx = -1;

    int warp_stride = LDG_LM_BLOCK_SIZE / WARP_SIZE;
    int base        = row_start + warp_id * LDG_LM_ROWS_PER_WARP;

    for (int m_base = base; m_base < row_end; m_base += warp_stride * LDG_LM_ROWS_PER_WARP) {
        int  rows[LDG_LM_ROWS_PER_WARP];
        bool valid[LDG_LM_ROWS_PER_WARP];
        float sum[LDG_LM_ROWS_PER_WARP];

        #pragma unroll
        for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
            rows[r]  = m_base + r;
            valid[r] = rows[r] < row_end;
            sum[r]   = 0.0f;
        }

        #pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            float4 a1 = *reinterpret_cast<const float4*>(s_hidden + k);
            float4 a2 = *reinterpret_cast<const float4*>(s_hidden + k + 4);

            #pragma unroll
            for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                if (!valid[r]) continue;
                const half *w_ptr = weight + rows[r] * HIDDEN_SIZE + k;
                uint4 w_u4 = ldg_load_weights_u4(reinterpret_cast<const uint4*>(w_ptr));
                const half2 *w_h2 = reinterpret_cast<const half2*>(&w_u4);

                float2 wf0 = __half22float2(w_h2[0]);
                float2 wf1 = __half22float2(w_h2[1]);
                float2 wf2 = __half22float2(w_h2[2]);
                float2 wf3 = __half22float2(w_h2[3]);

                sum[r] += wf0.x*a1.x + wf0.y*a1.y
                        + wf1.x*a1.z + wf1.y*a1.w
                        + wf2.x*a2.x + wf2.y*a2.y
                        + wf3.x*a2.z + wf3.y*a2.w;
            }
        }

        #pragma unroll
        for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
            if (!valid[r]) continue;
            float reduced = ldg_warp_reduce_sum(sum[r]);
            if (lane_id == 0 && reduced > local_max) {
                local_max     = reduced;
                local_max_idx = rows[r];
            }
        }
    }

    // Warp-level reduction within each warp (all lanes hold the same value after shfl)
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, local_max,     offset);
        int   other_idx = __shfl_down_sync(0xffffffff, local_max_idx, offset);
        if (other_val > local_max) {
            local_max     = other_val;
            local_max_idx = other_idx;
        }
    }
    local_max     = __shfl_sync(0xffffffff, local_max,     0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ struct { float val; int idx; } s_warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        s_warp_max[warp_id].val = local_max;
        s_warp_max[warp_id].idx = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int   num_warps  = LDG_LM_BLOCK_SIZE / WARP_SIZE;
        float final_max  = (lane_id < num_warps) ? s_warp_max[lane_id].val : -INFINITY;
        int   final_idx  = (lane_id < num_warps) ? s_warp_max[lane_id].idx : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float v = __shfl_down_sync(0xffffffff, final_max, offset);
            int   i = __shfl_down_sync(0xffffffff, final_idx, offset);
            if (v > final_max) { final_max = v; final_idx = i; }
        }

        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = final_max;
            block_max_idxs[blockIdx.x] = final_idx;
        }
    }
}


// LM head — Phase 2: global argmax reduction

__global__ void __launch_bounds__(LDG_LM_BLOCK_SIZE, 1) ldg_lm_head_phase2(
    const float *__restrict__ block_max_vals,
    const int   *__restrict__ block_max_idxs,
    int         *__restrict__ output_token,
    int          num_blocks)
{
    __shared__ struct { float val; int idx; } s_data[LDG_LM_BLOCK_SIZE];

    float thread_max = -INFINITY;
    int   thread_idx = -1;

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        float val = block_max_vals[i];
        if (val > thread_max) { thread_max = val; thread_idx = block_max_idxs[i]; }
    }

    s_data[threadIdx.x].val = thread_max;
    s_data[threadIdx.x].idx = thread_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_data[threadIdx.x + stride].val > s_data[threadIdx.x].val)
                s_data[threadIdx.x] = s_data[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        *output_token = s_data[0].idx;
}


// Launch functions (C linkage for Python extension)

static inline void ldg_configure_kernel_attributes();

// Helper: launch a cooperative kernel (hardware grid sync)
extern "C" void launch_ldg_decode_direct(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LDGLayerWeight *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer,
    void *g_activations, void *g_residual,     // g_residual: ABI compat, unused
    void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized,
    void *block_max_vals, void *block_max_idxs,
    int num_layers, int position, int max_seq_len,
    float attn_scale, cudaStream_t stream)
{
    ldg_configure_kernel_attributes();
    ensure_barrier_alloc();

    cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_barrier_sense,   0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_eos_flag,        0, sizeof(int),          stream);

    ldg_decode_kernel_direct<<<LDG_NUM_BLOCKS, LDG_BLOCK_SIZE, 0, stream>>>(
        (const half*)embed_weight, layer_weights,
        (const half*)final_norm_weight,
        (const half*)cos_table, (const half*)sin_table,
        (half*)k_cache, (half*)v_cache,
        (half*)hidden_buffer,
        (float*)g_activations, (float*)g_residual,
        (half*)g_q, (half*)g_k, (half*)g_v,
        (half*)g_attn_out, (float*)g_mlp_intermediate, (float*)g_normalized,
        d_norm_scratch,
        d_barrier_counter, d_barrier_sense, d_kv_flag, d_attn_flag,
        d_eos_flag,
        num_layers, position, input_token_id, max_seq_len, attn_scale);

    // cudaStreamSynchronize(stream);

    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized, (const half*)lm_head_weight,
        (float*)block_max_vals, (int*)block_max_idxs);

    ldg_lm_head_phase2<<<1, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)block_max_vals, (const int*)block_max_idxs,
        output_token_id, LDG_LM_NUM_BLOCKS);
}

extern "C" void launch_ldg_decode_persistent(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LDGLayerWeight *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer,
    void *g_activations, void *g_residual,     // ABI compat, unused
    void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized,
    void *block_max_vals, void *block_max_idxs,
    int num_layers, int position, int max_seq_len,
    float attn_scale, cudaStream_t stream)
{
    ldg_configure_kernel_attributes();
    ensure_barrier_alloc();

    *h_pinned_position = position;
    *h_pinned_token_id = input_token_id;
    cudaMemcpyAsync(d_mutable_position, h_pinned_position, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_mutable_token_id, h_pinned_token_id, sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Reset both barrier fields
    cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_barrier_sense,   0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_eos_flag,        0, sizeof(int),          stream);

    ldg_decode_kernel_persistent<<<LDG_NUM_BLOCKS, LDG_BLOCK_SIZE, 0, stream>>>(
        (const half*)embed_weight, layer_weights,
        (const half*)final_norm_weight,
        (const half*)cos_table, (const half*)sin_table,
        (half*)k_cache, (half*)v_cache,
        (half*)hidden_buffer,
        (float*)g_activations, (float*)g_residual,
        (half*)g_q, (half*)g_k, (half*)g_v,
        (half*)g_attn_out, (float*)g_mlp_intermediate, (float*)g_normalized,
        d_norm_scratch,
        d_barrier_counter, d_barrier_sense, d_kv_flag, d_attn_flag,
        d_eos_flag,
        num_layers, d_mutable_position, d_mutable_token_id,
        max_seq_len, attn_scale);

    // cudaStreamSynchronize(stream);

    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized, (const half*)lm_head_weight,
        (float*)block_max_vals, (int*)block_max_idxs);

    ldg_lm_head_phase2<<<1, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)block_max_vals, (const int*)block_max_idxs,
        output_token_id, LDG_LM_NUM_BLOCKS);
}

// Processes all prompt tokens on-device in one C call with one final sync.
// No Python overhead per token — just one cudaStreamSynchronize at the end.
extern "C" void launch_ldg_prefill(
    const int *token_ids,
    int        num_tokens,
    const void *embed_weight, const LDGLayerWeight *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer,
    void *g_activations, void *g_residual,
    void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized,
    void *block_max_vals, void *block_max_idxs,
    int num_layers, int start_position, int max_seq_len,
    float attn_scale, cudaStream_t stream)
{
    if (num_tokens <= 0) return;

    ldg_configure_kernel_attributes();
    ensure_barrier_alloc();

    // Allocate prefill scratch buffers once
    if (!d_prefill_dummy_log) {
        cudaMalloc(&d_prefill_dummy_log,    2048 * sizeof(int));
        cudaMalloc(&d_prefill_step_counter, sizeof(int));
        cudaMalloc(&d_prefill_output_token, sizeof(int));
    }

    // Grow-only static buffer — no per-request malloc/free
    if (num_tokens > d_prefill_token_ids_cap) {
        if (d_prefill_token_ids) cudaFree(d_prefill_token_ids);
        cudaMalloc(&d_prefill_token_ids, num_tokens * sizeof(int));
        d_prefill_token_ids_cap = num_tokens;
    }
    // Grow-only pinned positions array for safe async copies
    if (num_tokens > h_pinned_positions_cap) {
        if (h_pinned_positions) cudaFreeHost(h_pinned_positions);
        cudaHostAlloc(&h_pinned_positions, num_tokens * sizeof(int), cudaHostAllocDefault);
        h_pinned_positions_cap = num_tokens;
    }
    // Prefill all positions before any async copies begin
    for(int t = 0; t < num_tokens; t++)
        h_pinned_positions[t] = start_position + t;

    // Copy all token ids to device in one transfer
    cudaMemcpyAsync(d_prefill_token_ids, token_ids, num_tokens * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Process each prompt token: runs all N launches into same stream.
    // No cudaStreamSynchronize between tokens — one sync at the end.
    for (int t = 0; t < num_tokens; t++) {
        // Reset barrier for this step (stream-ordered, no CPU stall)
        cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
        cudaMemsetAsync(d_barrier_sense,   0, sizeof(unsigned int), stream);
        cudaMemsetAsync(d_eos_flag,        0, sizeof(int),          stream);

        // Upload this token's id and position (device-to-device for id)
        cudaMemcpyAsync(d_mutable_position, h_pinned_positions + t, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_mutable_token_id, d_prefill_token_ids + t, sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        ldg_decode_kernel_persistent<<<LDG_NUM_BLOCKS, LDG_BLOCK_SIZE, 0, stream>>>(
            (const half*)embed_weight, layer_weights,
            (const half*)final_norm_weight,
            (const half*)cos_table, (const half*)sin_table,
            (half*)k_cache, (half*)v_cache,
            (half*)hidden_buffer,
            (float*)g_activations, (float*)g_residual,
            (half*)g_q, (half*)g_k, (half*)g_v,
            (half*)g_attn_out, (float*)g_mlp_intermediate, (float*)g_normalized,
            d_norm_scratch,
            d_barrier_counter, d_barrier_sense, d_kv_flag, d_attn_flag,
            d_eos_flag,
            num_layers, d_mutable_position, d_mutable_token_id,
            max_seq_len, attn_scale);
    }

    // Single sync — all prompt tokens processed, KV cache fully populated
    cudaStreamSynchronize(stream);
    // Note: d_prefill_token_ids is intentionally NOT freed here (reused next call)
}

extern "C" void launch_ldg_generate_nosync(
    int first_token_id, int num_steps,
    const void *embed_weight, const LDGLayerWeight *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer,
    void *g_activations, void *g_residual,
    void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized,
    void *block_max_vals, void *block_max_idxs,
    int *output_log,
    int num_layers, int start_position, int max_seq_len,
    float attn_scale, int eos_token_id_arg, cudaStream_t stream)
{
    ldg_configure_kernel_attributes();
    ensure_barrier_alloc();

    static int *d_step_counter = nullptr;
    static int *d_output_token = nullptr;
    if (!d_step_counter) {
        cudaMalloc(&d_step_counter, sizeof(int));
        cudaMalloc(&d_output_token, sizeof(int));
    }
    cudaMemsetAsync(d_step_counter, 0, sizeof(int), stream);
    cudaMemsetAsync(d_output_token, 0, sizeof(int), stream);
    cudaMemsetAsync(d_eos_flag,     0, sizeof(int), stream);

    *h_pinned_position = start_position;
    *h_pinned_token_id = first_token_id;
    cudaMemcpyAsync(d_mutable_position, h_pinned_position, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_mutable_token_id, h_pinned_token_id, sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // All num_steps launches queued upfront — NO per-step sync.
    //
    // When ldg_update_step sets d_eos_flag = 1, subsequent
    // ldg_decode_kernel_persistent launches see the flag (stream-ordered)
    // and return immediately from ldg_decode_body — near-zero cost no-ops.
    // ldg_update_step also guards against writing garbage tokens after EOS.
    // The GPU pipeline stays full for the entire generation.
    for (int step = 0; step < num_steps; step++) {
        cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
        cudaMemsetAsync(d_barrier_sense,   0, sizeof(unsigned int), stream);

        ldg_decode_kernel_persistent<<<LDG_NUM_BLOCKS, LDG_BLOCK_SIZE, 0, stream>>>(
            (const half*)embed_weight, layer_weights,
            (const half*)final_norm_weight,
            (const half*)cos_table, (const half*)sin_table,
            (half*)k_cache, (half*)v_cache,
            (half*)hidden_buffer,
            (float*)g_activations, (float*)g_residual,
            (half*)g_q, (half*)g_k, (half*)g_v,
            (half*)g_attn_out, (float*)g_mlp_intermediate, (float*)g_normalized,
            d_norm_scratch,
            d_barrier_counter, d_barrier_sense, d_kv_flag, d_attn_flag,
            d_eos_flag,
            num_layers, d_mutable_position, d_mutable_token_id,
            max_seq_len, attn_scale);

        ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
            (const float*)g_normalized, (const half*)lm_head_weight,
            (float*)block_max_vals, (int*)block_max_idxs);

        ldg_lm_head_phase2<<<1, LDG_LM_BLOCK_SIZE, 0, stream>>>(
            (const float*)block_max_vals, (const int*)block_max_idxs,
            d_output_token, LDG_LM_NUM_BLOCKS);

        ldg_update_step<<<1, 1, 0, stream>>>(
            d_output_token, d_mutable_token_id,
            d_mutable_position, output_log,
            d_step_counter, d_eos_flag, eos_token_id_arg);

        // No sync here. No EOS memcpy here.
        // ldg_decode_body reads d_eos_flag on the GPU for early exit.
    }

    // Single sync — entire generation complete
    cudaStreamSynchronize(stream);
}

// Kernel attribute tuning
static inline void ldg_configure_kernel_attributes() {
    static bool configured = false;
    if (configured) return;
    configured = true;

    // FIX 9: MaxL1 for decode — weight streaming benefits from large L1
    cudaFuncSetAttribute(ldg_decode_kernel_persistent,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(ldg_decode_kernel_direct,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);

    // LM head: already MaxL1 — unchanged
    cudaFuncSetAttribute(ldg_lm_head_phase1,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(ldg_lm_head_phase2,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);
}
