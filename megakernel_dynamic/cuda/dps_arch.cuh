// dps_arch.cuh — thin wrappers over the PTX the dynamic persistent scheduler needs.
//
// Real instructions on sm_90+ (mbarrier tx-count, cp.async.bulk) and sm_100+
// (clusterlaunchcontrol).  Below sm_90 the mbarrier / bulk-copy pair is emulated
// with shared-memory atomics so the exact same pipeline code runs on older GPUs
// (used for local correctness testing; B200 always takes the real path).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define DPS_REAL_MBARRIER 1
#else
#define DPS_REAL_MBARRIER 0
#endif

// clusterlaunchcontrol.* is available on every sm_100+ target with CUDA 13
// (B200 = sm_100a, GB300 = sm_103a, RTX 5090 = sm_120).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#define DPS_HAS_CLC 1
#else
#define DPS_HAS_CLC 0
#endif

// DPS_TRACE=1 (setup.py builds it as qwen_dps_trace_C) records per-tile timestamps.
#ifndef DPS_TRACE
#define DPS_TRACE 0
#endif

namespace dps {

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// mbarrier: 8 bytes in hardware; the emulation needs 12, so every barrier slot
// is 16 bytes on all architectures.
struct alignas(16) Barrier {
    uint64_t hw;          // real mbarrier word (sm_90+)
    int      emu_count;   // emulation: arrivals per phase
    int      emu_phase;   // emulation: completed phases & 1
};

#if !DPS_REAL_MBARRIER
// Emulated mbarrier.  `remaining` = pending arrivals + pending tx bytes, packed
// into one int so a single atomic decides who completes the phase.
__device__ __forceinline__ int *emu_remaining(Barrier *b) {
    return reinterpret_cast<int *>(&b->hw);
}
__device__ __forceinline__ void emu_complete(Barrier *b) {
    atomicExch(emu_remaining(b), b->emu_count);
    __threadfence_block();
    atomicXor(&b->emu_phase, 1);
}
__device__ __forceinline__ void emu_add(Barrier *b, int delta) {
    __threadfence_block();
    int old = atomicAdd(emu_remaining(b), delta);
    if (old + delta == 0) emu_complete(b);
}
#endif

__device__ __forceinline__ void bar_init(Barrier *b, int count) {
#if DPS_REAL_MBARRIER
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(&b->hw)), "r"(count) : "memory");
#else
    *emu_remaining(b) = count;
    b->emu_count = count;
    b->emu_phase = 0;
#endif
}

// Make barrier initialisation visible to the async proxy (TMA / CLC unit).
__device__ __forceinline__ void bar_init_fence() {
#if DPS_REAL_MBARRIER
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

__device__ __forceinline__ void bar_arrive(Barrier *b) {
#if DPS_REAL_MBARRIER
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(&b->hw)) : "memory");
#else
    emu_add(b, -1);
#endif
}

// Arrive once and announce `bytes` of asynchronous transactions for this phase.
__device__ __forceinline__ void bar_arrive_expect_tx(Barrier *b, uint32_t bytes) {
#if DPS_REAL_MBARRIER
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_u32(&b->hw)), "r"(bytes) : "memory");
#else
    emu_add(b, static_cast<int>(bytes) - 1);
#endif
}

// Block until the phase with parity `parity` has completed (CUTLASS convention:
// a fresh barrier reports parity 1 as already complete).
__device__ __forceinline__ void bar_wait(Barrier *b, uint32_t parity) {
#if DPS_REAL_MBARRIER
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n" :: "r"(smem_u32(&b->hw)), "r"(parity) : "memory");
#else
    volatile int *ph = &b->emu_phase;
    while (static_cast<uint32_t>(*ph) == parity) { }
    __threadfence_block();
#endif
}

// L2 policy for streamed weights: they are read exactly once per token, so keep
// them from evicting the KV cache and activations.
__device__ __forceinline__ uint64_t l2_evict_first_policy() {
    uint64_t pol = 0;
#if DPS_REAL_MBARRIER
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
#endif
    return pol;
}

// Warp-collective global->shared bulk copy that completes `bytes` of tx on `b`.
// Real path: one lane issues cp.async.bulk (TMA, no tensor map needed).
// Emulated path: the whole warp copies, then lane 0 completes the tx.
// dst, src and bytes must be multiples of 16.
__device__ __forceinline__ void bulk_g2s_warp(void *dst, const void *src, uint32_t bytes,
                                              Barrier *b, uint64_t policy) {
    const int lane = threadIdx.x & 31;
#if DPS_REAL_MBARRIER
    if (lane == 0) {
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1], %2, [%3], %4;"
            :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(&b->hw)), "l"(policy)
            : "memory");
    }
    __syncwarp();
#else
    (void)policy;
    const uint4 *s = reinterpret_cast<const uint4 *>(src);
    uint4 *d = reinterpret_cast<uint4 *>(dst);
    for (uint32_t i = lane; i < bytes / 16; i += 32) d[i] = __ldg(s + i);
    __syncwarp();
    if (lane == 0) emu_add(b, -static_cast<int>(bytes));
    __syncwarp();
#endif
}

// Cluster launch control (Blackwell)
// try_cancel atomically cancels the launch of a not-yet-running CTA and writes a
// 16-byte response to `resp`; completion is signalled as 16 tx bytes on `b`.
__device__ __forceinline__ void clc_try_cancel(int4 *resp, Barrier *b) {
#if DPS_HAS_CLC
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];"
        :: "r"(smem_u32(resp)), "r"(smem_u32(&b->hw)) : "memory");
#else
    (void)resp; (void)b;
    __trap();
#endif
}

// Decode a try_cancel response. Returns true and the first ctaid.x of the
// cancelled cluster on success.
__device__ __forceinline__ bool clc_query(const int4 *resp, int *ctaid_x) {
#if DPS_HAS_CLC
    uint32_t valid, x;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        ".reg .b128 r;\n"
        "ld.shared.b128 r, [%2];\n"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "@p clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %1, r;\n"
        "}\n"
        : "=r"(valid), "=r"(x) : "r"(smem_u32(resp)) : "memory");
    // Order the generic-proxy read above before the next async-proxy write.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    *ctaid_x = static_cast<int>(x);
    return valid != 0;
#else
    (void)resp; *ctaid_x = -1;
    return false;
#endif
}

// Global synchronisation
__device__ __forceinline__ unsigned ld_acquire(const unsigned *p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ unsigned ld_relaxed(const unsigned *p) {
    unsigned v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// Release-increment: everything this CTA wrote before the preceding named
// barrier becomes visible to whoever acquires the new value.
__device__ __forceinline__ void signal_release(unsigned *p, unsigned v = 1) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void named_bar_sync(int id, int nthreads) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(nthreads) : "memory");
}

__device__ __forceinline__ void nanosleep_ns(unsigned ns) {
    asm volatile("nanosleep.u32 %0;" :: "r"(ns));
}

// Tracing: a device-wide nanosecond clock (comparable across SMs) and the SM id.
__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ unsigned smid() {
    unsigned s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
}

}  // namespace dps
