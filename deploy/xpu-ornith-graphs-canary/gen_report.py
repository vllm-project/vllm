#!/usr/bin/env python3
"""Build AB_COMPARE_<stamp>.{json,md} from arm_*/outputs_* JSON in results/."""

import difflib
import json
import os
import sys

results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
stamp = sys.argv[2]


def load(kind, arm):
    p = os.path.join(results_dir, f"{kind}_{arm}_{stamp}.json")
    return json.load(open(p)) if os.path.exists(p) else None


arms = [a for a in ("A", "B", "C") if load("outputs", a)]
outputs = {a: load("outputs", a)["outputs"] for a in arms}
perf = {a: load("arm", a) for a in arms}

agreement = {}
for a in arms:
    if a == "A":
        continue
    per = {}
    for name, ref in outputs["A"].items():
        got = outputs[a].get(name, "")
        per[name] = {
            "identical": ref == got,
            "similarity": round(difflib.SequenceMatcher(None, ref, got).ratio(), 4),
        }
    agreement[f"A_vs_{a}"] = per

summary = {
    "stamp": stamp,
    "arms_completed": arms,
    "perf": {
        a: {k: v for k, v in (perf[a] or {}).items() if k != "text"}
        for a in arms
        if perf.get(a)
    },
    "output_agreement": agreement,
}
with open(os.path.join(results_dir, f"AB_COMPARE_{stamp}.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

names = {
    "A": "A: eager (prod config)",
    "B": "B: PIECEWISE",
    "C": "C: FA-in-graph FULL",
}
L = [
    f"# Ornith XPU graphs canary A/B — {stamp}",
    "",
    (
        "Model: Ornith-1.0-35B-MXFP4 (compressed-tensors MXFP4 MoE + hybrid "
        "GDN/Mamba), bf16, fp8 KV, greedy single-stream, Arc Pro B70."
    ),
    "",
    (
        "| Arm | Ready s | TTFT ms mean | TTFT ms p50 | Decode tok/s mean | "
        "Decode tok/s p50 |"
    ),
    "| --- | --- | --- | --- | --- | --- |",
]
for a in arms:
    p = perf.get(a)
    L.append(
        f"| {names[a]} | {p.get('ready_s', '-')} | {p['ttft_ms_mean']:.1f} | "
        f"{p['ttft_ms_p50']:.1f} | {p['decode_tok_s_mean']:.1f} | "
        f"{p['decode_tok_s_p50']:.1f} |"
    )
base = perf.get("A")
for a in arms:
    if a == "A" or not base:
        continue
    d = (
        (perf[a]["decode_tok_s_mean"] - base["decode_tok_s_mean"])
        / base["decode_tok_s_mean"]
        * 100
    )
    t = (perf[a]["ttft_ms_mean"] - base["ttft_ms_mean"]) / base["ttft_ms_mean"] * 100
    L.append(f"\n**{a} vs A:** decode {d:+.1f}%, TTFT {t:+.1f}%")
for pair, per in agreement.items():
    ident = sum(1 for v in per.values() if v["identical"])
    diffs = ", ".join(
        f"{k}={v['similarity']}" for k, v in per.items() if not v["identical"]
    )
    L.append(
        f"\n**Output agreement {pair}:** {ident}/{len(per)} byte-identical "
        f"vs eager" + (f"; near-identical: {diffs}" if diffs else "")
    )
md = "\n".join(L) + "\n"
with open(os.path.join(results_dir, f"AB_COMPARE_{stamp}.md"), "w") as fh:
    fh.write(md)
print(md)
