# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-step CPU breakdown: how much of a decode step is host-side KV-cache
management + input preparation vs the (already compiled + cudagraph-captured)
model forward.

Motivation
----------
On the stock torch.compile path (#46423) the decode forward -- including
attention and the KV-cache write -- is already captured as a single FULL
cudagraph for FA3 models like GPT-OSS. What is NOT compiled/captured is the
per-step host work in ``GPUModelRunner``: ``_prepare_inputs`` (block-table H2D
commit, slot-mapping kernel launch, positions/seq-len math) and
``_build_attention_metadata`` (block-table gather, slot-mapping fetch,
per-layer metadata build). This tool quantifies that split so we can size the
prize of compiling the KV-cache management into the graph.

Method
------
Runs a decode-heavy workload under vLLM's built-in torch profiler (with_stack)
and attributes per-decode-step CPU time to the runner's Python functions by
parsing the emitted chrome trace. ``with_stack`` inflates absolute Python
times, so read the *ratio* (host-prep vs forward), not the absolute us.

Usage
-----
    python benchmarks/compile/decode_step_breakdown.py \
        --model openai/gpt-oss-120b --tp 1 --batches 1,32 --max-tokens 64
"""

import argparse
import gzip
import json
import os

# Functions we attribute per-step CPU time to (file is gpu_model_runner.py).
# Keyed by the trailing "(line): name" the profiler emits; lines can drift
# across versions, so we match on the function name suffix instead.
HOST_PREP_FNS = ["_prepare_inputs", "_build_attention_metadata", "_preprocess"]
FORWARD_FNS = ["_model_forward"]
OTHER_FNS = ["execute_model", "sample_tokens", "_update_states"]
# KV-management sub-calls, surfaced to show where the host prep time goes.
KV_MGMT_FNS = [
    "_get_slot_mappings",
    "_get_block_table",
    "_build_attn_group_metadata",
    "_prepare_input_ids",
    "_get_cumsum_and_arange",
]


def _fn_name(trace_event_name: str) -> str | None:
    # e.g. "vllm/vllm/v1/worker/gpu_model_runner.py(1890): _prepare_inputs"
    if "gpu_model_runner.py" not in trace_event_name:
        return None
    tail = trace_event_name.rsplit(":", 1)
    return tail[-1].strip() if len(tail) == 2 else None


def parse_trace(path: str) -> dict[str, tuple[float, int]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    agg: dict[str, list] = {}
    for e in data.get("traceEvents", []):
        if e.get("cat") != "python_function" or e.get("ph") != "X":
            continue
        name = _fn_name(e.get("name", ""))
        if name is None:
            continue
        slot = agg.setdefault(name, [0.0, 0])
        slot[0] += e.get("dur", 0.0)
        slot[1] += 1
    return {k: (v[0], v[1]) for k, v in agg.items()}


def _mean(agg, name):
    tot, cnt = agg.get(name, (0.0, 0))
    return (tot / cnt) if cnt else 0.0


def report(label: str, agg: dict) -> None:
    host = sum(_mean(agg, f) for f in HOST_PREP_FNS)
    fwd = sum(_mean(agg, f) for f in FORWARD_FNS)
    print(f"\n===== {label} =====")
    print(f"{'function':<32}{'mean us/step':>14}")
    for f in OTHER_FNS + HOST_PREP_FNS + FORWARD_FNS:
        print(f"{f:<32}{_mean(agg, f):>14.1f}")
    print("  -- KV-management sub-calls --")
    for f in KV_MGMT_FNS:
        print(f"  {f:<30}{_mean(agg, f):>14.1f}")
    print(f"host KV-mgmt+prep total : {host:>10.1f} us/step")
    print(f"captured forward        : {fwd:>10.1f} us/step")
    if fwd:
        print(f"host-prep / forward     : {host / fwd:>10.1f}x")


def run(args) -> None:
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        profiler_config={"profiler": "torch", "torch_profiler_dir": outdir},
    )
    for bsz in [int(b) for b in args.batches.split(",")]:
        before = set(os.listdir(outdir))
        prompts = ["Hello"] * bsz
        sp = SamplingParams(
            temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True
        )
        llm.generate(prompts, sp, use_tqdm=False)  # warmup, not profiled
        llm.start_profile()
        llm.generate(prompts, sp, use_tqdm=False)
        llm.stop_profile()
        new = [f for f in set(os.listdir(outdir)) - before if f.endswith(".json.gz")]
        if not new:
            print(f"[batch{bsz}] no trace produced")
            continue
        agg = parse_trace(os.path.join(outdir, sorted(new)[-1]))
        report(f"batch{bsz} decode ({args.model}, TP{args.tp})", agg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batches", default="1,32")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--out", default="/tmp/decode_step_breakdown")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
