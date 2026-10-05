"""Where is a hung CuTeDSL launch stuck? Reads the kernel's progress counters while it runs.

    python megakernel_dynamic/hang_probe.py                    # cutedsl, fp16, atomic
    python megakernel_dynamic/hang_probe.py --sched clc --format fp8

Launches one generate (compile + launch return without waiting for the GPU), then copies the
global sync counters to the host on a separate non-blocking stream, which runs on the copy
engines and does not wait for the kernel. Reads them --reads times, --interval s apart, so it
also shows whether anything still moves, then exits; process exit kills a hung kernel.
"""

import argparse
import faulthandler
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "cutedsl"))
from qwen_dps import SCHED_MODES, DpsDecoder, load_weights  # noqa: E402
from test_dps import PROMPTS, chat_prompt  # noqa: E402
import qwen_dps_cutedsl as q  # noqa: E402

# counter -> tiles it counts per (step, layer) instance; LM counters are per decode step
COUNTERS = ([("ticket", q.C_TICKET, None)]
            + [(f"qkv[{g}]", q.C_QKV + g, q.QKV_TILES_GROUP) for g in range(q.NKV)]
            + [(f"asplit[{h}]", q.C_ASPLIT + h, q.ATTN_SPLITS) for h in range(q.NKV)]
            + [("attn", q.C_ATTN, q.NKV), ("oproj", q.C_OPROJ, q.T_O), ("gateup", q.C_GATEUP, q.T_GU),
               ("down", q.C_DOWN, q.T_D), ("lm_arrive", q.C_LM_ARRIVE, q.T_LM), ("lm_done", q.C_LM_DONE, 1),
               ("eos", q.C_EOS, None)])


def where(ticket, n_pre):
    """Ticket -> (step, layer or 'LM', phase tile index) in the kernel's tile order."""
    pre = n_pre * q.T_STEP
    if ticket < pre:
        step, rem = divmod(ticket, q.T_STEP)
    else:
        k, rem = divmod(ticket - pre, q.T_STEP + q.T_LM)
        step = n_pre + k
    if rem >= q.T_STEP:
        return f"step {step}, LM head tile {rem - q.T_STEP}"
    layer, x = divmod(rem, q.T_LAYER)
    for name, n in (("QKV", q.T_QKV), ("ATTN", q.T_ATTN), ("O-proj", q.T_O), ("gate/up", q.T_GU), ("down", q.T_D)):
        if x < n:
            return f"step {step}, layer {layer}, {name} tile {x}"
        x -= n
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched", default="atomic", choices=["atomic", "clc"])
    ap.add_argument("--format", default="fp16", choices=["fp16", "fp8", "fp4"])
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--reads", type=int, default=3)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    weights, tok, _ = load_weights()
    dec = DpsDecoder(weights, tok, backend="cutedsl", sched=args.sched, weight_format=args.format)
    ids = tok.encode(chat_prompt(tok, PROMPTS[0]), add_special_tokens=False)
    n_prompt, n_pre = len(ids), len(ids) - 1
    total = n_pre * q.T_STEP + args.max_new * (q.T_STEP + q.T_LM)
    tokens = torch.empty(n_prompt + args.max_new, dtype=torch.int32, device="cuda")
    tokens[:n_prompt].copy_(torch.tensor(ids, dtype=torch.int32))
    out = torch.full((args.max_new,), -1, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()

    faulthandler.dump_traceback_later(60 + args.reads * args.interval, exit=True)   # if the probe itself blocks
    t0 = time.perf_counter()
    dec.ext.generate(tokens, n_prompt, args.max_new, 0, -1, SCHED_MODES[args.sched], out)
    print(f"{args.format} {args.sched}: compiled + launched in {time.perf_counter() - t0:.1f} s; "
          f"prompt {n_prompt} tokens -> {n_pre} prefill steps, {total} tiles in the launch")

    side = torch.cuda.Stream()
    sync = dec.ext.ws[:q.SYNC_BYTES].view(torch.int32)
    host = torch.empty(sync.numel(), dtype=torch.int32, pin_memory=True)
    prev = None
    for r in range(args.reads):
        time.sleep(args.interval)
        with torch.cuda.stream(side):
            host.copy_(sync, non_blocking=True)
        side.synchronize()
        vals = host[::q.CSTRIDE].tolist()
        moved = "" if prev is None else ("  (still moving)" if vals != prev else "  (no change since last read)")
        print(f"\nread {r + 1} at +{time.perf_counter() - t0:.1f} s{moved}")
        for name, idx, unit in COUNTERS:
            v = vals[idx]
            if unit is None:
                extra = f"  next ticket would be: {where(v, n_pre)}" if name == "ticket" and v < total else ""
                print(f"  {name:<10} {v:>9}{extra}")
            elif name.startswith("lm"):
                print(f"  {name:<10} {v:>9}  = {v // unit} decode steps + {v % unit}/{unit}")
            else:
                step, layer = divmod(v // unit, q.NL)
                print(f"  {name:<10} {v:>9}  = next: step {step} layer {layer}, {v % unit}/{unit} done")
        prev = vals
    if prev[q.C_LM_DONE] >= args.max_new:
        print("\nthe launch finished: not hung")
    os._exit(0)   # a hung kernel dies with the process


if __name__ == "__main__":
    main()
