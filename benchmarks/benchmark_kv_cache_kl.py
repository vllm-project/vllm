# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure how far a quantized KV cache moves the model's next-token
distribution from the unquantized (bf16) cache.

For every ``--kv-cache-dtype`` the model scores the same teacher-forced token
sequence (``prompt_logprobs``); the script reports, against the ``auto`` run:

* the mean per-token KL(reference || quantized) over the reference top-k tokens
  (a LOWER BOUND of the true KL: a reference top-k token missing from the
  quantized run's top-k is given that run's k-th logprob), with its standard
  error over blocks of tokens,
* the perplexity change (dPPL) and top-1 agreement.

Why a small ``--max-num-batched-tokens``: backends whose prefill attends the
raw K/V of the tokens being prefilled (for example TurboQuant) never read the
quantized cache for those tokens, so one long prefill reports "lossless".
Scheduling the prompt in small chunks makes every chunk attend to the earlier
tokens through the quantized cache. Tokens inside one chunk still attend to
each other unquantized in such backends, so the chunk size is also the size of
an implicit exact window; keep it small and compare dtypes at the same value.

Example:
    python benchmarks/benchmark_kv_cache_kl.py --model Qwen/Qwen3-4B \\
        --kv-cache-dtypes int4_per_token_head turboquant_4bit_nc \\
        --text-file docs/benchmarking/cli.md \\
        --num-tokens 8192 --max-num-batched-tokens 16

"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

TOP_K = 20  # vLLM's default ``max_logprobs``


def _worker(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.text_file, errors="ignore") as f:
        text = f.read()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) < args.num_tokens:
        # Repeat short files so that small documents can still be used.
        ids = (ids * (args.num_tokens // len(ids) + 1))[: args.num_tokens]
    ids = ids[: args.num_tokens]

    kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.num_tokens + 64,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        max_logprobs=TOP_K,
        seed=0,
    )
    if args.worker_dtype != "auto":
        kwargs["kv_cache_dtype"] = args.worker_dtype
    llm = LLM(**kwargs)
    out = llm.generate(
        [{"prompt_token_ids": ids}],
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=TOP_K),
    )
    prompt_logprobs = out[0].prompt_logprobs

    n = len(ids) - 1
    top_ids = np.full((n, TOP_K), -1, dtype=np.int32)
    top_lp = np.full((n, TOP_K), -1e9, dtype=np.float32)
    actual = np.zeros(n, dtype=np.float32)
    for i in range(1, len(ids)):
        entries = prompt_logprobs[i]
        actual[i - 1] = entries[ids[i]].logprob
        top = sorted(
            ((t, e.logprob) for t, e in entries.items() if e.rank <= TOP_K),
            key=lambda x: -x[1],
        )[:TOP_K]
        for j, (t, lp) in enumerate(top):
            top_ids[i - 1, j], top_lp[i - 1, j] = t, lp
    np.savez(args.worker_out, top_ids=top_ids, top_lp=top_lp, actual=actual)


def _compare(ref: dict, run: dict) -> dict:
    n = len(ref["actual"])
    kl = np.zeros(n)
    for i in range(n):
        valid = run["top_ids"][i] >= 0
        run_lp = dict(zip(run["top_ids"][i][valid], run["top_lp"][i][valid]))
        floor = float(run["top_lp"][i][valid].min())
        q = np.array([run_lp.get(t, floor) for t in ref["top_ids"][i]])
        p = np.exp(ref["top_lp"][i])
        kl[i] = float((p * (ref["top_lp"][i] - q)).sum())
    blocks = kl[: (n // 256) * 256].reshape(-1, 256).mean(axis=1)
    se = float(blocks.std(ddof=1) / np.sqrt(len(blocks))) if len(blocks) > 1 else 0.0
    return {
        "kl_pct": 100.0 * float(kl.mean()),
        "kl_se_pct": 100.0 * se,
        "dppl_pct": 100.0 * float(np.exp((ref["actual"] - run["actual"]).mean()) - 1.0),
        "top1_agree": float((ref["top_ids"][:, 0] == run["top_ids"][:, 0]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-cache-dtypes", nargs="+", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--worker-dtype", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_dtype:
        _worker(args)
        return

    # One engine per dtype, in its own process, so GPU memory is released.
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        runs = {}
        for dtype in ["auto", *[d for d in args.kv_cache_dtypes if d != "auto"]]:
            out = os.path.join(tmp, f"{dtype}.npz")
            subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    *[a for a in sys.argv[1:]],
                    "--worker-dtype",
                    dtype,
                    "--worker-out",
                    out,
                ],
                check=True,
            )
            runs[dtype] = dict(np.load(out))
        for dtype, run in runs.items():
            if dtype != "auto":
                results[dtype] = _compare(runs["auto"], run)

    header = f"{'kv-cache-dtype':28s} {'KL % (lower bound)':>22s}"
    print(f"{header} {'dPPL %':>9s} {'top-1':>7s}")
    for dtype, r in results.items():
        print(
            f"{dtype:28s} {r['kl_pct']:12.4f} +- {r['kl_se_pct']:6.4f} "
            f"{r['dppl_pct']:+9.3f} {r['top1_agree']:7.4f}"
        )
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {"args": vars(args), "results": results}, f, indent=2, default=str
            )


if __name__ == "__main__":
    main()
