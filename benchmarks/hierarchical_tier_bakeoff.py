# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark harness: hierarchical expert staging vs baseline (Colibri bakeoff aid).

Reports tokens/s, TTFT proxy, and tier hit rates for a MoE model under
hierarchical staging. When Colibri is installed separately, compare its
published tok/s on the same model/quant by filling --colibri-tok-s.

Example (hal bakeoff default — Mixtral-8x22B Instruct AWQ / Q4):
  python benchmarks/hierarchical_tier_bakeoff.py \\
    --model /tank/nas/models/Mixtral-8x22B-Instruct-v0.1-AWQ \\
    --tier-num-slots 4 --tier-ram-gb 32 --max-tokens 64 --num-prompts 8
"""

from __future__ import annotations

import argparse
import json
import time

from vllm import LLM, SamplingParams
from vllm.model_executor.offloader.hierarchical.manager import get_tier_manager


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tier-num-slots", type=int, default=16)
    p.add_argument("--tier-ram-gb", type=float, default=4.0)
    p.add_argument("--tier-disk-path", default=None)
    p.add_argument("--tier-pilot", action="store_true")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--colibri-tok-s", type=float, default=None,
                   help="Optional Colibri baseline tok/s for the same model")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    prompts = [
        f"Explain mixture-of-experts inference briefly. Example #{i}."
        for i in range(args.num_prompts)
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    llm = LLM(
        model=args.model,
        offload_backend="hierarchical",
        tier_num_slots=args.tier_num_slots,
        tier_ram_gb=args.tier_ram_gb,
        tier_disk_path=args.tier_disk_path,
        tier_pilot=args.tier_pilot,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Warmup
    llm.generate(prompts[:1], sampling)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_s = total_tokens / max(elapsed, 1e-6)
    ttft_proxy = elapsed / max(len(prompts), 1)  # coarse under continuous batching

    mgr = get_tier_manager()
    stats = mgr.stats.snapshot() if mgr is not None else {}

    result = {
        "model": args.model,
        "elapsed_s": elapsed,
        "total_tokens": total_tokens,
        "tok_s": tok_s,
        "ttft_proxy_s": ttft_proxy,
        "tier_stats": stats,
        "colibri_tok_s": args.colibri_tok_s,
        "speedup_vs_colibri": (
            tok_s / args.colibri_tok_s if args.colibri_tok_s else None
        ),
        "tier_num_slots": args.tier_num_slots,
        "tier_pilot": args.tier_pilot,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
