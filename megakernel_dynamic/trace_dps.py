"""Where does a token step go? Per-tile timeline of the dynamic-persistent megakernel.

    cd megakernel_dynamic/cuda && DPS_TRACE=1 python setup.py build_ext --inplace && cd ../..
    python megakernel_dynamic/trace_dps.py                               # fp16, every scheduler the GPU has
    python megakernel_dynamic/trace_dps.py --formats fp16,fp8,fp4 --out-dir traces
    python megakernel_dynamic/trace_dps.py --load traces/fp8_clc.npz      # re-analyse a saved trace, no GPU

One launch decodes --skip + --steps + 4 tokens after a test prompt with EOS disabled,
and records every tile of --steps decode steps after the first --skip. Per tile:
  claim  the scheduler warp took the ticket      start  the compute warps received it
  ready  its dependency counters were satisfied  end    it finished, done-signal included
  prep   time building its input vector (RMSNorm / copy of activations from global)
  wwait  time waiting for weight stages from the TMA warp
Times come from %globaltimer, so they compare across SMs.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_dps import TRACE_FIELDS, DpsDecoder, first_ticket  # noqa: E402

# Tile graph, mirrors qwen_dps_megakernel.cu.
NL = 28
PHASES = ["QKV", "ATTN", "O-proj", "gate/up", "down", "LM head"]
PHASE_TILES = [256, 64, 128, 384, 256]
T_LAYER = sum(PHASE_TILES)
PHASE_BOUNDS = np.cumsum(PHASE_TILES)
TRACE_DTYPE = np.dtype(TRACE_FIELDS)


def decode_tiles(tickets, n_pre, t_step, t_lm):
    """Ticket -> (step, layer, phase), as decode_tile() does on the GPU."""
    pre = n_pre * t_step
    dec = tickets >= pre
    step = np.where(dec, n_pre + (tickets - pre) // (t_step + t_lm), tickets // t_step)
    rem = np.where(dec, tickets - pre - (step - n_pre) * (t_step + t_lm), tickets - step * t_step)
    lm = rem >= t_step
    layer = np.where(lm, NL, rem // T_LAYER)
    phase = np.where(lm, 5, np.searchsorted(PHASE_BOUNDS, rem - layer * T_LAYER, side="right"))
    return step, layer, phase


def us(x):
    return f"{x / 1e3:7.2f}"


def analyse(rec, meta):
    t_step, t_lm, n_pre = int(meta["tiles_per_step"]), int(meta["lm_tiles"]), int(meta["n_pre"])
    tickets = int(meta["first_ticket"]) + np.arange(len(rec))
    ok = rec["end"] > 0   # tiles the kernel never ran (early stop) keep zeros
    rec, tickets = rec[ok], tickets[ok]
    step, layer, phase = decode_tiles(tickets, n_pre, t_step, t_lm)
    steps = np.unique(step)
    t = {k: rec[k].astype(np.int64) for k in ("claim", "start", "ready", "end")}
    prep, wwait = rec["prep_ns"].astype(np.int64), rec["wwait_ns"].astype(np.int64)

    stamps = np.unique(np.concatenate([t["start"], t["ready"], t["end"]]))
    res = np.diff(stamps).min() if len(stamps) > 1 else 0
    print(f"\n##### {meta['label']}  [{meta['gpu']}]  {len(rec)} tiles over {len(steps)} decode steps, "
          f"{len(np.unique(rec['sm']))} SMs, globaltimer step ~{res} ns")

    # Step time: spacing of the last tile end (the LM head argmax) of consecutive steps.
    ends = np.array([t["end"][step == s].max() for s in steps])
    step_ns = np.diff(ends).mean() if len(ends) > 1 else float("nan")
    print(f"token step: {step_ns / 1e3:.1f} us traced"
          + (f" | {meta['untraced_us']:.1f} us untraced (qwen_dps_C)" if meta.get("untraced_us") else ""))

    # 1. Per-tile cost by phase.
    print("\nper tile (mean us)  tiles/step   queue  dep-wait     prep  w-wait   other    busy")
    queue, dep = t["start"] - t["claim"], t["ready"] - t["start"]
    other = t["end"] - t["ready"] - prep - wwait
    for ph, name in enumerate(PHASES):
        m = phase == ph
        if not m.any():
            continue
        print(f"  {name:<9} {m.sum() / len(steps):17.0f} {us(queue[m].mean())} {us(dep[m].mean())}  "
              f"{us(prep[m].mean())} {us(wwait[m].mean())} {us(other[m].mean())} "
              f"{us((t['end'] - t['start'])[m].mean())}")
    print("  queue = ticket waiting in the CTA's claim-ahead ring; dep-wait = waiting for the counters it reads;\n"
          "  other = dot products, reductions, barriers and the done-signal")

    # 2. Critical path: a step is the chain QKV -> ATTN -> O -> gate/up -> down (x28) -> LM head.
    # Per link: segment = this phase's last tile end - the previous link's last end (segments of
    # the steps after the first sum exactly to their duration), handoff = this phase's first ready
    # tile - the previous link's last end (negative = the phases overlap), spread = last end -
    # first ready within the phase.
    group = (NL + 1) * 6
    key = (step - steps[0]) * group + layer * 6 + phase
    last_end = np.zeros(len(steps) * group, np.int64)
    first_ready = np.full(len(steps) * group, np.iinfo(np.int64).max)
    np.maximum.at(last_end, key, t["end"])
    np.minimum.at(first_ready, key, t["ready"])
    links = [(L, ph) for L in range(NL) for ph in range(5)] + [(NL, 5)]
    chain = [si * group + L * 6 + ph for si in range(len(steps)) for L, ph in links]
    chain = [k for k in chain if last_end[k] > 0]
    seg, hand, spread = ({ph: [] for ph in range(6)} for _ in range(3))
    for prev, cur in zip(chain, chain[1:]):
        if cur < group:
            continue   # first traced step: its first link has no predecessor
        ph = cur % 6
        seg[ph].append(last_end[cur] - last_end[prev])
        hand[ph].append(first_ready[cur] - last_end[prev])
        spread[ph].append(last_end[cur] - first_ready[cur])
    print("\ncritical path    links/step  us/step   segment  handoff   spread  (mean us per link)")
    total = 0.0
    for ph, name in enumerate(PHASES):
        if not seg[ph]:
            continue
        per_step = len(seg[ph]) / (len(steps) - 1)
        total += np.sum(seg[ph]) / (len(steps) - 1)
        print(f"  {name:<9} {per_step:12.0f} {np.sum(seg[ph]) / (len(steps) - 1) / 1e3:9.1f}  "
              f"{us(np.mean(seg[ph]))}  {us(np.mean(hand[ph]))}  {us(np.mean(spread[ph]))}")
    print(f"  {'sum':<9} {'':>12} {total / 1e3:9.1f}")

    # 3. What the SMs do over the traced window.
    lo, hi = t["start"].min(), t["end"].max()
    n_sm = len(np.unique(rec["sm"]))
    busy = (t["end"] - t["ready"]).sum() / (n_sm * (hi - lo))
    waiting = (t["ready"] - t["start"]).sum() / (n_sm * (hi - lo))
    order = np.lexsort((t["start"], rec["sm"]))
    same = rec["sm"][order][1:] == rec["sm"][order][:-1]
    gaps = (t["start"][order][1:] - t["end"][order][:-1])[same]
    print(f"\nSM time: {busy:.0%} in tiles, {waiting:.0%} waiting on dependencies, "
          f"{1 - busy - waiting:.0%} between tiles (mean gap {gaps.mean() / 1e3:.2f} us); "
          f"{len(rec) / len(steps) / n_sm:.0f} tiles per SM per step")


def decode_step_us(dec, ids, n_new, n_base=4, reps=3):
    """Decode-step time without prefill: difference of two launches, best of `reps`."""
    def launch(n):
        best = float("inf")
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dec.generate_ids(ids, n, None)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        return best

    dec.generate_ids(ids, 8, None)
    return (launch(n_new) - launch(n_base)) / (n_new - n_base) * 1e6


def run(args):
    from qwen_dps import load_weights
    from test_dps import PROMPTS, chat_prompt

    weights, tok, _ = load_weights(args.model)
    ids = tok.encode(chat_prompt(tok, PROMPTS[1]), add_special_tokens=False)
    n_pre = len(ids) - 1
    s0, s1 = n_pre + args.skip, n_pre + args.skip + args.steps
    max_new = args.skip + args.steps + 4
    for fmt in args.formats.split(","):
        fmt = fmt.strip()
        probe = DpsDecoder(weights, tok, sched="atomic", weight_format=fmt, trace=True)
        info = probe.info()
        del probe
        scheds = [args.sched] if args.sched else (["atomic", "clc"] if info["clc_supported"] else ["atomic"])
        for sched in scheds:
            untraced = None
            try:   # the same decode on the normal build, to show what tracing costs
                untraced = decode_step_us(DpsDecoder(weights, tok, sched=sched, weight_format=fmt), ids, max_new)
            except ImportError:
                pass
            dec = DpsDecoder(weights, tok, sched=sched, weight_format=fmt, trace=True)
            dec.generate_ids(ids, 8, None)
            dec.generate_ids(ids, max_new, None, trace_steps=(s0, s1))
            t_first, raw = dec.last_trace
            meta = dict(label=f"{fmt} {sched}", gpu=torch.cuda.get_device_name(), first_ticket=t_first,
                        n_pre=n_pre, tiles_per_step=info["tiles_per_step"], lm_tiles=info["lm_tiles"],
                        untraced_us=untraced or 0.0)
            rec = raw.numpy().reshape(-1).view(TRACE_DTYPE)
            analyse(rec, meta)
            if args.out_dir:
                os.makedirs(args.out_dir, exist_ok=True)
                path = os.path.join(args.out_dir, f"{fmt}_{sched}.npz")
                np.savez_compressed(path, records=rec, **meta)
                print(f"saved {path}")
            del dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default="fp16", help="comma-separated subset of fp16,fp8,fp4")
    ap.add_argument("--sched", default=None, choices=["atomic", "clc"], help="default: every supported mode")
    ap.add_argument("--skip", type=int, default=16, help="decode steps before the traced ones")
    ap.add_argument("--steps", type=int, default=8, help="decode steps to trace (at least 2)")
    ap.add_argument("--out-dir", default=None, help="save each trace as <format>_<sched>.npz")
    ap.add_argument("--load", nargs="+", default=None, help="analyse saved .npz traces instead of running")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    if args.steps < 2:
        ap.error("--steps must be at least 2: step times come from consecutive steps")
    if args.load:
        for path in args.load:
            z = np.load(path)
            analyse(z["records"], {k: z[k].item() for k in z.files if k != "records"})
    else:
        run(args)


if __name__ == "__main__":
    main()
