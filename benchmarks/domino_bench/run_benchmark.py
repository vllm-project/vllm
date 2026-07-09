# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Domino vs DFlash speculative decoding on HumanEval.

Usage (from repo root):
    source .venv-bench/bin/activate
    python benchmarks/domino_bench/run_benchmark.py

Compares two draft checkpoints:
  1. DFlash baseline  (projector_type="dflash")
  2. Domino           (projector_type="domino", with GRU correction head)

Both use Qwen3-0.6B as the verifier.

Metrics: acceptance length, per-position acceptance, tokens-per-second.
"""

import argparse
import gc
import json
import os

# -- make domino_inference importable from benchmarks/domino_bench --
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from domino_inference import load_draft_model


def chat_prompt(question: str, tokenizer) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_benchmark(args):
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.verifier)

    # ------------------------------------------------------------------
    # Load verifier (shared)
    # ------------------------------------------------------------------
    print("Loading verifier:", args.verifier)
    verifier = AutoModelForCausalLM.from_pretrained(
        args.verifier,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    verifier.eval()
    n = sum(p.numel() for p in verifier.parameters()) / 1e6
    print(f"  Verifier loaded: {n:.1f}M params")

    # ------------------------------------------------------------------
    # Load HumanEval
    # ------------------------------------------------------------------
    print(f"\nLoading HumanEval ({args.num_prompts} prompts)...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    dataset = dataset.select(range(min(args.num_prompts, len(dataset))))

    prompt_fmt = (
        "Write a solution to the following problem and make sure that it "
        "passes the tests:\n```python\n{prompt}\n```"
    )

    results = {}
    for model_name in args.models:
        ckpt_dir = args.checkpoints[model_name]
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}  ({ckpt_dir})")
        print(f"{'=' * 60}")

        # Load draft
        print("Loading draft model...")
        draft = load_draft_model(ckpt_dir, device)

        # Warm-up
        print("Warm-up run...")
        warm_ids = tokenizer.encode("Hello", return_tensors="pt").to(device)
        draft.spec_generate(
            warm_ids,
            verifier,
            max_new_tokens=16,
            temperature=0.0,
            return_metrics=False,
        )
        torch.cuda.synchronize()

        # Benchmark
        total_time_s = 0.0
        total_out_tokens = 0
        total_accept = 0
        total_steps = 0
        all_pos_accept: list[list[int]] = []
        per_prompt: list[dict] = []

        for idx, example in enumerate(dataset):
            prompt_text = prompt_fmt.format(prompt=example["prompt"])
            chat_text = chat_prompt(prompt_text, tokenizer)
            input_ids = tokenizer.encode(chat_text, return_tensors="pt").to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = draft.spec_generate(
                input_ids,
                verifier,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                return_metrics=True,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            n_out = int(result.num_output_tokens)
            avg_accept = sum(result.acceptance_lengths) / max(
                len(result.acceptance_lengths), 1
            )
            tps = n_out / elapsed if elapsed > 0 else 0.0

            total_time_s += elapsed
            total_out_tokens += n_out
            total_accept += sum(result.acceptance_lengths)
            total_steps += len(result.acceptance_lengths)

            for p, acc_list in enumerate(result.pos_accept):
                while len(all_pos_accept) <= p:
                    all_pos_accept.append([])
                all_pos_accept[p].extend(acc_list)

            per_prompt.append(
                {
                    "idx": int(idx),
                    "acceptance_length": round(avg_accept, 3),
                    "output_tokens": n_out,
                    "time_s": round(elapsed, 3),
                    "tokens_per_sec": round(tps, 1),
                }
            )

            if (idx + 1) % 20 == 0 or idx == 0:
                print(
                    f"  [{idx + 1}/{len(dataset)}] "
                    f"accept={avg_accept:.2f}  tokens={n_out}  tps={tps:.1f}"
                )

        # Aggregate
        mean_accept = total_accept / max(total_steps, 1)
        mean_tps = total_out_tokens / total_time_s if total_time_s > 0 else 0.0
        pos_rates = {}
        for p, acc_list in enumerate(all_pos_accept):
            if acc_list:
                pos_rates[f"pos_{p}"] = {
                    "accept_rate": round(sum(acc_list) / len(acc_list), 4),
                    "n": len(acc_list),
                }

        results[model_name] = {
            "checkpoint": ckpt_dir,
            "projector_type": draft.projector_type,
            "mean_acceptance_length": round(mean_accept, 3),
            "total_output_tokens": total_out_tokens,
            "total_time_s": round(total_time_s, 2),
            "tokens_per_sec": round(mean_tps, 1),
            "num_prompts": len(dataset),
            "num_decode_steps": total_steps,
            "pos_acceptance": pos_rates,
            "per_prompt": per_prompt,
        }

        del draft
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = args.output or "benchmark_results.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    header = (
        f"{'Model':<12} {'Accept len':>10} {'TPS':>10} {'Tokens':>8} {'Time(s)':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<12} {r['mean_acceptance_length']:>10.3f} "
            f"{r['tokens_per_sec']:>10.1f} {r['total_output_tokens']:>8} "
            f"{r['total_time_s']:>10.1f}"
        )

    # Per-position acceptance rates
    print(f"\n{'=' * 60}")
    print("Per-position acceptance rate")
    print(f"{'=' * 60}")
    all_pos = sorted(
        {p for r in results.values() for p in r["pos_acceptance"]},
        key=lambda x: int(x.split("_")[1]),
    )
    if all_pos:
        print(f"{'Model':<12}", end="")
        for p in all_pos:
            print(f"  {p:>8}", end="")
        print()
        for name, r in results.items():
            print(f"{name:<12}", end="")
            for p in all_pos:
                if p in r["pos_acceptance"]:
                    print(f"  {r['pos_acceptance'][p]['accept_rate']:>8.3f}", end="")
                else:
                    print(f"  {'N/A':>8}", end="")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINTS = {
    "dflash": "/home/arnab/code/personalCode/speculators/dflash_checkpoints/1",
    "domino": "/home/arnab/code/personalCode/speculators/domino_checkpoints/1",
}


def main():
    parser = argparse.ArgumentParser(description="Domino vs DFlash benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(DEFAULT_CHECKPOINTS),
        default=["dflash", "domino"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=164,
        help="Number of HumanEval prompts (default: 164)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Max new tokens per prompt"
    )
    parser.add_argument(
        "--verifier", type=str, default="Qwen/Qwen3-0.6B", help="Verifier model name"
    )
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()
    args.checkpoints = DEFAULT_CHECKPOINTS
    run_benchmark(args)


if __name__ == "__main__":
    main()
