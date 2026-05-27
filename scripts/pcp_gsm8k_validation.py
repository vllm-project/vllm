#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GSM8K accuracy validation for PCP=4 on DeepSeek-V2-Lite-Chat.

Mirrors the PR #28988 / tests/distributed/test_context_parallel.py
config (256 questions, 5-shot, MIN_ACCURACY=0.64) but launches the
engine offline with the new Q-sharded PCP path enabled via
``--prefill-context-parallel-size 4``.

Usage::

    .venv/bin/python scripts/pcp_gsm8k_validation.py \
        --tp 1 --pcp 4 --max-model-len 4096 --max-num-seqs 64 \
        --num-questions 256

The script exits 0 on accuracy >= --min-accuracy and 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _add_repo_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def main() -> int:
    _add_repo_to_path()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V2-Lite-Chat",
        help="HF model id or local path",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pcp", type=int, default=4)
    parser.add_argument("--dcp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--num-questions", type=int, default=256)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.64,
        help="Minimum accuracy threshold (default matches PR #28988)",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"]
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable cuda graphs (slower but easier to debug)",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help=(
            "Attention backend name. Default None lets vLLM auto-select per "
            "compute capability (FlashInferMLA on Blackwell/sm10, "
            "FLASH_ATTN_MLA or FLASHMLA on Hopper/sm9). All three are now "
            "supports_pcp=True; FlashMLASparse is the only MLA backend "
            "that does not."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Sampling seed for determinism"
    )
    parser.add_argument(
        "--result-json",
        default=None,
        help="Optional path to write the eval result dict as JSON",
    )
    args = parser.parse_args()

    # Import inside main to keep --help fast.
    from vllm import LLM

    from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline

    print(
        f"Loading model {args.model} with TP={args.tp} PCP={args.pcp} "
        f"DCP={args.dcp} dtype={args.dtype} kv_cache_dtype={args.kv_cache_dtype} "
        f"max_model_len={args.max_model_len} max_num_seqs={args.max_num_seqs}",
        flush=True,
    )

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        prefill_context_parallel_size=args.pcp,
        decode_context_parallel_size=args.dcp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        distributed_executor_backend="mp",
        trust_remote_code=True,
        seed=args.seed,
    )
    if args.kv_cache_dtype != "auto":
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.attention_backend:
        llm_kwargs["attention_backend"] = args.attention_backend

    t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s", flush=True)

    results = evaluate_gsm8k_offline(
        llm,
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        max_tokens=args.max_tokens,
        temperature=0.0,
    )

    accuracy = float(results["accuracy"])
    print("=" * 60)
    print(f"GSM8K result: accuracy={accuracy:.4f}")
    print(
        f"  questions={args.num_questions}  shots={args.num_shots}  "
        f"max_tokens={args.max_tokens}"
    )
    print(f"  total tokens: {results.get('total_tokens', '?')}")
    print(f"  latency: {results.get('latency', '?'):.1f}s")
    print(f"  min required: {args.min_accuracy}")
    print("=" * 60)

    if args.result_json:
        with open(args.result_json, "w") as f:
            json.dump(
                {
                    "accuracy": accuracy,
                    "min_accuracy": args.min_accuracy,
                    "model": args.model,
                    "tp": args.tp,
                    "pcp": args.pcp,
                    "dcp": args.dcp,
                    "num_questions": args.num_questions,
                    "num_shots": args.num_shots,
                    "results": {k: v for k, v in results.items() if k != "states"},
                },
                f,
                indent=2,
            )
        print(f"Wrote result to {args.result_json}")

    if accuracy < args.min_accuracy:
        print(
            f"FAIL: accuracy {accuracy:.4f} below threshold {args.min_accuracy}",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: accuracy {accuracy:.4f} >= threshold {args.min_accuracy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
