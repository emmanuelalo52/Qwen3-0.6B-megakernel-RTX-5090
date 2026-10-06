// dps_scheduler.cuh — dynamic persistent tile scheduler for the Qwen megakernel.
//
// CUDA C++ port of CUTLASS's CuTeDSL
//   cutlass/utils/dynamic_persistent_tile_scheduler.py  (ClcDynamicPersistentTileScheduler)
//   cutlass/pipeline/sm100.py                           (PipelineClcFetchAsync)
//
// Same structure as the CUTLASS kernels that use it:
//   * the grid is launched with one CTA per work tile (non-persistent shape);
//   * the CTA's own launch is its first unit of work (initial_work_tile_info);
//   * one elected thread of a dedicated scheduler warp asks the hardware for more
//     work with clusterlaunchcontrol.try_cancel (advance_to_next_work), which
//     cancels a CTA that has not been launched yet and hands its work to us;
//   * the answer travels to the consumer warps through a ring of shared-memory
//     slots guarded by full/empty mbarriers (get_current_work).
//
// One deliberate difference: CUTLASS GEMM tiles are independent, so the cancelled
// CTA's blockIdx *is* the tile. Megakernel tiles depend on earlier tiles through
// global counters, which is only deadlock-free if tiles are handed out in
// dependency (topological) order — the hardware does not promise an order for
// try_cancel. So a successful launch/cancel is treated as a *permission* to run
// one tile, and the tile itself is an ordered ticket from a global counter.
// The number of permissions equals the grid size equals the number of tiles,
// so every ticket is consumed exactly once.
//
// SchedMode::Atomic drops CLC and launches one persistent CTA per SM that pulls
// tickets until they run out (the software dynamic persistent scheduler). It is
// the fallback for pre-Blackwell GPUs and a baseline to compare against.
//
// SchedMode::Static is the static persistent scheduler: one co-resident CTA per SM
// slot (cooperative launch), and CTA c runs tickets c, c + G, c + 2G, ... (G = grid
// size) with no atomics. Deadlock-free for the same reason as the ordered tickets:
// the lowest unfinished ticket belongs to a resident CTA whose earlier tickets, and
// so its inputs, are all done. When no phase has more tiles than the grid has CTAs,
// each CTA gets at most one tile of every phase.
//
// Claim-ahead. The producer claims a ticket only once a ring slot is free for it,
// so a CTA holds at most Stages tickets it has not started. Claiming further ahead
// lets the TMA warp prefetch more weights, but in the dynamic modes it also hands
// out tiles before anyone knows which will be ready first: on B200 the busiest SM
// ended up with 2-3x its share of each phase (see README, "B200 benchmarks").
#pragma once
#include "dps_arch.cuh"

namespace dps {

// kSchedOneShot is a test mode: grid = tiles, every CTA runs exactly one ticket
// and exits. It reproduces CLC's launch pattern (CTAs that are not resident hold
// no work) on GPUs without CLC, to check the ordering argument above.
enum SchedMode : int { kSchedAuto = 0, kSchedAtomic = 1, kSchedClc = 2, kSchedOneShot = 3, kSchedStatic = 4 };

// try_cancel queries issued back-to-back while draining the grid after an early stop.
constexpr int kDrainBatch = 16;

struct WorkTileInfo {
    int  tile_idx;
    bool is_valid_tile;
#if DPS_TRACE
    uint64_t claim_ns = 0;       // when the scheduler warp took the ticket
#endif
};

// Producer/consumer position in a ring of barriers (cutlass.pipeline.PipelineState).
template <int Stages>
struct PipelineState {
    int      index;
    uint32_t phase;
    __device__ __forceinline__ void advance() {
        if (++index == Stages) { index = 0; phase ^= 1u; }
    }
};

// Producers start on phase 1 so their first pass over a fresh ring does not block.
template <int Stages>
__device__ __forceinline__ PipelineState<Stages> make_producer_state() { return {0, 1u}; }
template <int Stages>
__device__ __forceinline__ PipelineState<Stages> make_consumer_state() { return {0, 0u}; }

template <int Stages>
struct SchedulerStorage {
    Barrier full[Stages];        // producer -> consumers: work[i] is ready
    Barrier empty[Stages];       // consumers -> producer: work[i] may be overwritten
    Barrier clc_bar;             // completes when try_cancel responses land
    int4    clc_resp[kDrainBatch];
    int     work[Stages];        // ticket, or -1 = no more work
#if DPS_TRACE
    uint64_t claim_ns[Stages];
#endif
};

struct SchedulerParams {
    unsigned       *ticket;      // global ordered-ticket counter (zeroed per launch)
    const unsigned *stop_flag;   // non-zero: stop handing out work (EOS reached)
    int             total_tiles;
    int             mode;        // kSchedAtomic, kSchedClc, kSchedOneShot or kSchedStatic
};

template <int Stages>
class DynamicPersistentTileScheduler {
  public:
    __device__ DynamicPersistentTileScheduler(const SchedulerParams &params,
                                              SchedulerStorage<Stages> *storage)
        : params_(params), st_(storage), next_static_(static_cast<int>(blockIdx.x)) {}

    // One thread, before the CTA-wide barrier that publishes the storage.
    __device__ static void init_storage(SchedulerStorage<Stages> *st, int num_consumer_warps) {
        for (int i = 0; i < Stages; ++i) {
            bar_init(&st->full[i], 1);
            bar_init(&st->empty[i], num_consumer_warps);
        }
        bar_init(&st->clc_bar, 1);
    }

    // producer: scheduler warp, single thread. A ticket is claimed only after a slot
    // is free for it (producer_acquire first), so at most Stages are claimed ahead.
    __device__ void run_producer() {
        PipelineState<Stages> prod = make_producer_state<Stages>();
        bar_wait(&st_->empty[prod.index], prod.phase);                // producer_acquire
        WorkTileInfo work = initial_work_tile_info();
        for (;;) {
            st_->work[prod.index] = work.is_valid_tile ? work.tile_idx : -1;
#if DPS_TRACE
            st_->claim_ns[prod.index] = work.claim_ns;
#endif
            bar_arrive(&st_->full[prod.index]);                        // producer_commit
            prod.advance();
            if (!work.is_valid_tile) break;
            bar_wait(&st_->empty[prod.index], prod.phase);            // producer_acquire
            work = advance_to_next_work();
        }
    }

    // consumers: every lane of a consumer warp
    // consumer_wait + get_current_work + consumer_release.
    __device__ WorkTileInfo get_current_work(PipelineState<Stages> &cons) {
        bar_wait(&st_->full[cons.index], cons.phase);
        WorkTileInfo w;
        w.tile_idx      = *reinterpret_cast<volatile int *>(&st_->work[cons.index]);
        w.is_valid_tile = w.tile_idx >= 0;
#if DPS_TRACE
        w.claim_ns = *reinterpret_cast<volatile uint64_t *>(&st_->claim_ns[cons.index]);
#endif
        __syncwarp();
        if ((threadIdx.x & 31) == 0) bar_arrive(&st_->empty[cons.index]);
        cons.advance();
        return w;
    }

  private:
    // The CTA's own launch is the first permission: no query needed.
    __device__ WorkTileInfo initial_work_tile_info() { return take_ticket(); }

    __device__ WorkTileInfo advance_to_next_work() {
        if (params_.mode == kSchedOneShot) return {-1, false};
        if (stop_requested()) {
            if (params_.mode == kSchedClc) drain_grid();
            return {-1, false};
        }
        if (params_.mode == kSchedClc) {
            bar_arrive_expect_tx(&st_->clc_bar, 16);
            clc_try_cancel(&st_->clc_resp[0], &st_->clc_bar);
            bar_wait(&st_->clc_bar, clc_phase_);
            clc_phase_ ^= 1u;
            int ctaid_x;
            // Declined: every CTA of the grid has been launched or cancelled.
            if (!clc_query(&st_->clc_resp[0], &ctaid_x)) return {-1, false};
        }
        return take_ticket();
    }

    __device__ WorkTileInfo take_ticket() {
        WorkTileInfo w;
        if (stop_requested()) {
            w.tile_idx = -1;
        } else if (params_.mode == kSchedStatic) {
            w.tile_idx = next_static_;
            next_static_ += static_cast<int>(gridDim.x);
        } else {
            w.tile_idx = static_cast<int>(atomicAdd(params_.ticket, 1u));
        }
        w.is_valid_tile = w.tile_idx >= 0 && w.tile_idx < params_.total_tiles;
#if DPS_TRACE
        w.claim_ns = globaltimer_ns();
#endif
        return w;
    }

    __device__ bool stop_requested() const { return ld_relaxed(params_.stop_flag) != 0u; }

    // After an early stop the rest of the grid is useless. Letting the hardware
    // launch every remaining CTA (each needs ~200 KB of smem, so one per SM) would
    // take far longer than cancelling them in batches from the CTAs already running.
    __device__ void drain_grid() {
        for (;;) {
            bar_arrive_expect_tx(&st_->clc_bar, 16u * kDrainBatch);
            for (int i = 0; i < kDrainBatch; ++i) clc_try_cancel(&st_->clc_resp[i], &st_->clc_bar);
            bar_wait(&st_->clc_bar, clc_phase_);
            clc_phase_ ^= 1u;
            bool all_cancelled = true;
            for (int i = 0; i < kDrainBatch; ++i) {
                int ctaid_x;
                all_cancelled &= clc_query(&st_->clc_resp[i], &ctaid_x);
            }
            if (!all_cancelled) return;
        }
    }

    SchedulerParams           params_;
    SchedulerStorage<Stages> *st_;
    uint32_t                  clc_phase_ = 0;
    int                       next_static_;   // kSchedStatic: this CTA's next ticket
};

}  // namespace dps
