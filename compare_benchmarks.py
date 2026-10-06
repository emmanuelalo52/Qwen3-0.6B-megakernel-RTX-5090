"""Compare client_benchmark.py logs as a markdown table, like the RTX 5090 table in README.md.

    python compare_benchmarks.py --baseline latency_vllm.json latency_dps_fp16.json [more logs ...]

Every metric is recomputed from the per-request records (successful requests only), so logs
from older client versions work too:
  avg / median / p90 latency        per-request wall time seen by the client (HTTP included)
  tokens/s                          mean over requests of output tokens / latency
  end-to-end tok/s                  all output tokens / all request time
  req/s                             requests / all request time
  min-max latency                   spread of per-request latency
Speedups are against --baseline (latency: baseline / run; rates: run / baseline).

It also checks that the runs are comparable: same prompts, max_tokens and request count;
how many answers match the baseline's text exactly (both decode greedily, but different
kernels can break near-ties differently); and, for matching answers, whether both servers
count output tokens the same way (vLLM counts the EOS token).
"""

import argparse
import json
import statistics


def load(path):
    with open(path) as f:
        d = json.load(f)
    ok = [r for r in d["requests"] if r.get("error") is None and r.get("latency_s")]
    lat = [r["latency_s"] for r in ok]
    out = [r["output_tokens"] for r in ok]
    meta = d.get("meta", {})
    return {
        "path": path,
        "label": meta.get("label") or meta.get("step") or path,
        "meta": meta,
        "by_index": {r["index"]: r for r in ok},
        "n": len(ok),
        "failed": len(d["requests"]) - len(ok),
        "avg": statistics.mean(lat),
        "median": statistics.median(lat),
        "p90": statistics.quantiles(lat, n=10)[-1] if len(lat) > 1 else lat[0],
        "min": min(lat),
        "max": max(lat),
        "tok_s": statistics.mean(o / l for o, l in zip(out, lat)),
        "e2e": sum(out) / sum(lat),
        "rps": len(ok) / sum(lat),
        "mean_out": statistics.mean(out),
    }


def fmt_row(name, cells):
    return f"| {name} | " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="log of the reference server (e.g. vLLM)")
    ap.add_argument("logs", nargs="+", help="logs to compare against the baseline")
    args = ap.parse_args()

    base = load(args.baseline)
    runs = [load(p) for p in args.logs]
    cols = runs + [base]
    labels = [r["label"] for r in cols]
    for r in cols:   # older client versions labelled every log as the vLLM baseline
        if labels.count(r["label"]) > 1:
            r["label"] = f'{r["label"]} ({r["path"]})'

    print(fmt_row("Metric", [r["label"] for r in cols]))
    print("|---|" + "---|" * len(cols))

    def row(name, key, unit, lower_is_better, digits):
        cells = []
        for r in cols:
            v = f"{r[key]:.{digits}f}{unit}"
            if r is not base:
                x = base[key] / r[key] if lower_is_better else r[key] / base[key]
                v += f" (**{x:.2f}x**)"
            cells.append(v)
        print(fmt_row(name, cells))

    row("Avg latency", "avg", " s", True, 3)
    row("Median latency", "median", " s", True, 3)
    row("P90 latency", "p90", " s", True, 3)
    row("Tokens/s (per request)", "tok_s", "", False, 1)
    row("End-to-end throughput", "e2e", " tok/s", False, 1)
    row("Req/s", "rps", "", False, 2)
    print(fmt_row("Latency min-max", [f"{r['min']:.3f}-{r['max']:.3f} s" for r in cols]))
    print(fmt_row("Mean output tokens", [f"{r['mean_out']:.1f}" for r in cols]))
    print(fmt_row("Successful / failed", [f"{r['n']} / {r['failed']}" for r in cols]))

    print("\nComparability checks against the baseline:")
    for r in runs:
        notes = []
        for key in ("max_tokens", "temperature", "concurrency"):
            if r["meta"].get(key) != base["meta"].get(key):
                notes.append(f"{key} {r['meta'].get(key)} vs {base['meta'].get(key)}")
        common = sorted(set(r["by_index"]) & set(base["by_index"]))
        same_q = sum(r["by_index"][i]["question"] == base["by_index"][i]["question"] for i in common)
        if same_q != len(common):
            notes.append(f"only {same_q}/{len(common)} prompts identical")
        same = [i for i in common if r["by_index"][i]["answer"] == base["by_index"][i]["answer"]]
        diff = [base["by_index"][i]["output_tokens"] - r["by_index"][i]["output_tokens"] for i in same]
        offset = statistics.median(diff) if diff else 0
        line = f"  {r['label']}: {len(same)}/{len(common)} answers identical to the baseline"
        if diff:
            line += (f"; on those, output tokens differ by {offset:+g} (median)"
                     + ("" if offset == 0 else " -> the servers count tokens differently; tok/s is skewed"))
        print(line)
        for n in notes:
            print(f"    WARNING: {n}")


if __name__ == "__main__":
    main()
