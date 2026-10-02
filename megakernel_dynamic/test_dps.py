"""Correctness + latency check for the dynamic-persistent megakernel.

    python megakernel_dynamic/test_dps.py                       # CUDA backend, fp16/fp8/fp4, every scheduler the GPU supports
    python megakernel_dynamic/test_dps.py --backend cutedsl
    python megakernel_dynamic/test_dps.py --formats fp4 --bench-tokens 256

Reference for each weight format: HF transformers running the same weights the
kernel sees — the original fp16 model for fp16, and for fp8/fp4 the model with
every quantized matrix replaced by its dequantized values ("fake quant"). That
checks the kernel, not the quantization. Quantization quality is reported
separately as agreement with the fp16 model.

Small fp16 accumulation differences can flip a near-tie late in a sequence, so the
report shows how many leading tokens match rather than demanding an exact match.
"""

import argparse
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_dps import (DpsDecoder, SCHED_NAMES, dequantize_matrix, quantize_weights,  # noqa: E402
                      weights_from_model)

PROMPTS = [
    "What is the capital of France?",
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about GPUs.",
    # > 256 prompt tokens: attention spans several KV splits and the split combine.
    "Here are some notes on GPU memory.\n"
    + " ".join(f"Note {i}: shared memory is fast, global memory is large, and L2 sits between them." for i in range(16))
    + "\nSummarize the notes in one sentence.",
]

_HF_MODULES = {"q": ("self_attn", "q_proj"), "k": ("self_attn", "k_proj"), "v": ("self_attn", "v_proj"),
               "o": ("self_attn", "o_proj"), "gate": ("mlp", "gate_proj"), "up": ("mlp", "up_proj"),
               "down": ("mlp", "down_proj")}


def chat_prompt(tok, text: str) -> str:
    p = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    p = re.sub(r"<think>.*?</think>\n*", "", p, flags=re.DOTALL)
    return p.rstrip() + "\n<think>\n\n</think>\n\n"


def hf_greedy(model, ids, max_new, eos):
    inp = torch.tensor([ids], device="cuda")
    out = model.generate(inp, attention_mask=torch.ones_like(inp), max_new_tokens=max_new, do_sample=False,
                         temperature=None, top_p=None, top_k=None, eos_token_id=eos, pad_token_id=eos)
    gen = out[0, len(ids):].tolist()
    return gen[: gen.index(eos)] if eos in gen else gen


def matching_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@torch.no_grad()
def apply_fake_quant(model, weights):
    """Overwrite the HF model in place with the dequantized matrices the kernel uses.
    The embedding lookup stays fp16, so the tied LM head gets its own parameter."""
    fmt = weights["format"]
    for i, layer in enumerate(model.model.layers):
        for name, (block, proj) in _HF_MODULES.items():
            getattr(getattr(layer, block), proj).weight.copy_(dequantize_matrix(layer_w(weights, i, name), fmt).half())
    lm = dequantize_matrix(weights["lm_head"], fmt).half()
    model.lm_head.weight = torch.nn.Parameter(lm, requires_grad=False)


def layer_w(weights, i, name):
    return weights["layers"][i][name]


def run_format(fmt, args, tok, eos):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    weights = weights_from_model(model)
    prompts = [tok.encode(chat_prompt(tok, t), add_special_tokens=False) for t in PROMPTS]
    fp16_ref = [hf_greedy(model, ids, args.max_new, eos) for ids in prompts]
    if fmt != "fp16":
        weights = quantize_weights(weights, fmt)
        apply_fake_quant(model, weights)
    refs = [hf_greedy(model, ids, args.max_new, eos) for ids in prompts] if fmt != "fp16" else fp16_ref

    probe = DpsDecoder(weights, tok, backend=args.backend, sched="atomic")
    info = probe.info()
    del probe
    print(f"\n##### weights={fmt}  [{torch.cuda.get_device_name()}] stage={info['stage_bytes']} B x "
          f"{info['weight_stages']}, claim-ahead {info['sched_stages']}, smem {info.get('smem_bytes', '?')} B")
    scheds = [args.sched] if args.sched else (["atomic", "clc"] if info["clc_supported"] else ["atomic"])

    ok = True
    for sched in scheds:
        dec = DpsDecoder(weights, tok, backend=args.backend, sched=sched)
        print(f"=== backend={args.backend} scheduler={sched} weights={fmt}")
        for ids, ref, base in zip(prompts, refs, fp16_ref):
            got = dec.generate_ids(ids, args.max_new, eos)
            n = matching_prefix(ref, got)
            mode, grid = dec.last_launch
            status = "OK " if n == len(ref) == len(got) else ("~  " if n >= min(len(ref), len(got)) // 2 else "BAD")
            ok &= status != "BAD"
            quality = "" if fmt == "fp16" else f" | vs fp16 model: {matching_prefix(base, got)}/{len(base)} leading"
            print(f"  {status} prompt={len(ids):3d} tok | ref {len(ref):3d} / ours {len(got):3d} tokens, "
                  f"{n} leading match{quality} | sched={SCHED_NAMES[mode]} grid={grid}")
            print(f"      ours: {tok.decode(got)[:110]!r}")
            if n < len(ref):
                print(f"      ref : {tok.decode(ref)[:110]!r}")

        if sched == "oneshot":
            continue   # one CTA launch per tile: correctness only
        # Latency: EOS disabled so every run decodes exactly bench_tokens.
        ids = prompts[1]
        dec.generate_ids(ids, 8, None)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dec.generate_ids(ids, args.bench_tokens, None)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        steps = len(ids) - 1 + args.bench_tokens
        print(f"  latency: {dt * 1e3:.1f} ms for {len(ids)}-token prompt + {args.bench_tokens} new tokens "
              f"({dt / steps * 1e6:.0f} us per token step, single launch)")
        del dec

    del model, weights
    torch.cuda.empty_cache()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda", choices=["cuda", "cutedsl"])
    ap.add_argument("--sched", default=None, choices=["atomic", "clc", "oneshot"],
                    help="default: every supported mode; oneshot = CLC launch pattern without CLC (test only)")
    ap.add_argument("--formats", default="fp16,fp8,fp4", help="comma-separated subset of fp16,fp8,fp4")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--bench-tokens", type=int, default=128)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ok = True
    for fmt in args.formats.split(","):
        ok &= run_format(fmt.strip(), args, tok, tok.eos_token_id)
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
