#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal acceptance experiment for pairwise_fp4 quantization.

Drives five experiment groups on GSM8K using Qwen2.5-0.5B-Instruct:
  A: BF16 baseline
  B: FP4 no-rotation  (pairwise_fp4, top_ratio=0)
  C: FP4 weight_only  (pairwise_fp4, mode=weight_only, top_ratio=0.1)
  D: FP4 activation_only (pairwise_fp4, mode=activation_only, top_ratio=0.1)
  E: FP4 joint         (pairwise_fp4, mode=joint, top_ratio=0.1)

Usage:
    # Smoke test (20 samples, single group)
    python scripts/eval_pairwise_fp4_gsm8k.py --group A --num-samples 20

    # Run all groups with 50 samples each
    python scripts/eval_pairwise_fp4_gsm8k.py --group all --num-samples 50

    # Full dataset, specific group
    python scripts/eval_pairwise_fp4_gsm8k.py --group C --num-samples 0

    # With rotation diagnostics
    python scripts/eval_pairwise_fp4_gsm8k.py --group C --num-samples 20 --diagnose
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = (
    "/root/autodl-tmp/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    "/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
DEFAULT_DATA = "/root/autodl-tmp/huggingface/data/gsm8k/test.parquet"

GROUPS = {
    "A": "BF16 baseline",
    "B": "FP4 no-rotation",
    "C": "FP4 weight_only",
    "D": "FP4 activation_only",
    "E": "FP4 joint",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_gsm8k(path: str, num_samples: int = 0) -> list[dict]:
    """Load GSM8K parquet and return list of {question, answer, prompt}."""
    df = pd.read_parquet(path)
    samples = []
    for _, row in df.iterrows():
        prompt_msgs = row["prompt"]
        if hasattr(prompt_msgs, "tolist"):
            prompt_msgs = prompt_msgs.tolist()
        else:
            prompt_msgs = list(prompt_msgs)

        # Ground truth from reward_model.ground_truth
        rm = row["reward_model"]
        gt = str(rm.get("ground_truth", ""))

        # Full question from extra_info or prompt
        ei = row.get("extra_info", {})
        question = ei.get("question", prompt_msgs[0]["content"])

        samples.append({
            "question": question,
            "ground_truth": gt,
            "prompt_messages": prompt_msgs,
        })

    if num_samples > 0:
        samples = samples[:num_samples]

    return samples


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_NUMBER_RE = re.compile(r"(-?[\d,]+\.?\d*)")


def extract_answer(text: str) -> str | None:
    """Extract numeric answer from model output.

    Looks for '#### <number>' pattern first (GSM8K convention),
    then falls back to last number in text.
    """
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    ans = ans.strip().replace(",", "")
    # Remove trailing .0
    if ans.endswith(".0"):
        ans = ans[:-2]
    return ans


# ---------------------------------------------------------------------------
# Experiment group configuration
# ---------------------------------------------------------------------------


def get_llm_kwargs(group: str, model_path: str) -> dict:
    """Return kwargs for vllm.LLM() constructor for the given group."""
    base = {
        "model": model_path,
        "trust_remote_code": True,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.85,
    }

    if group == "A":
        base["dtype"] = "bfloat16"
        return base

    # All FP4 groups use pairwise_fp4
    base["quantization"] = "pairwise_fp4"
    base["dtype"] = "bfloat16"
    base["enforce_eager"] = True  # FP4 emulation incompatible with CUDA graphs

    # Use emulation backend (env var set separately)

    if group == "B":
        # No rotation
        cfg = {"mode": "weight_only", "top_ratio": 0.0}
    elif group == "C":
        cfg = {"mode": "weight_only", "top_ratio": 0.1}
    elif group == "D":
        cfg = {"mode": "activation_only", "top_ratio": 0.1,
               "use_prebuilt_plan": True}
    elif group == "E":
        cfg = {"mode": "joint", "top_ratio": 0.1,
               "use_prebuilt_plan": True}
    else:
        raise ValueError(f"Unknown group: {group}")

    base["hf_overrides"] = {
        "quantization_config_dict_json": json.dumps(cfg),
    }
    return base


def prepare_prebuilt_plan(group: str, model_path: str) -> str | None:
    """For activation_only/joint, build a prebuilt plan file.

    Returns path to plan file, or None if not needed.
    """
    if group not in ("D", "E"):
        return None

    from vllm.model_executor.layers.quantization.pairwise_fp4.utils import (
        RotationPlan,
        empty_angles,
        empty_pairs,
        save_plan,
    )

    mode = "activation_only" if group == "D" else "joint"
    # For v1 without calibration data, use empty plan
    # (activation rotation requires calibration data we don't have)
    plan = RotationPlan(
        mode=mode,
        layer_index="prebuilt_default",
        pairs=empty_pairs(),
        angles=empty_angles(),
    )
    plan_dir = tempfile.mkdtemp(prefix=f"pairwise_fp4_plan_{group}_")
    plan_path = os.path.join(plan_dir, "plan.json")
    save_plan(plan, plan_path)
    return plan_path


def update_config_with_plan(llm_kwargs: dict, plan_path: str) -> None:
    """Inject prebuilt plan path into the config."""
    if plan_path is None:
        return
    cfg_json = llm_kwargs.get("hf_overrides", {}).get(
        "quantization_config_dict_json", "{}"
    )
    cfg = json.loads(cfg_json)
    cfg["rotation_plan_path"] = plan_path
    cfg["use_prebuilt_plan"] = True
    llm_kwargs.setdefault("hf_overrides", {})[
        "quantization_config_dict_json"
    ] = json.dumps(cfg)


# ---------------------------------------------------------------------------
# Rotation diagnostics
# ---------------------------------------------------------------------------


def diagnose_rotation(llm, group: str) -> dict:
    """Check rotation parameters on loaded model layers."""
    diag = {
        "group": group,
        "layers_checked": 0,
        "layers_with_rotation": 0,
        "sample_pairs_shapes": [],
        "sample_angles_stats": [],
    }

    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        diag["error"] = "Could not access model internals"
        return diag

    for name, module in model.named_modules():
        if hasattr(module, "rotation_pairs"):
            diag["layers_checked"] += 1
            pairs = module.rotation_pairs
            angles = module.rotation_angles

            has_rotation = pairs.numel() > 0
            if has_rotation:
                diag["layers_with_rotation"] += 1

            if diag["layers_checked"] <= 3:  # Sample first 3
                diag["sample_pairs_shapes"].append({
                    "layer": name,
                    "pairs_shape": list(pairs.shape),
                    "pairs_numel": pairs.numel(),
                    "angles_shape": list(angles.shape),
                })
                if angles.numel() > 0:
                    diag["sample_angles_stats"].append({
                        "layer": name,
                        "mean": angles.float().mean().item(),
                        "std": angles.float().std().item(),
                        "min": angles.float().min().item(),
                        "max": angles.float().max().item(),
                    })

        # Also check quantization attributes
        if hasattr(module, "weight") and hasattr(module, "weight_scale"):
            if not hasattr(module, "rotation_pairs"):
                # Has FP4 but no rotation tracking — might be a non-linear layer
                pass

    return diag


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    group: str
    group_name: str
    num_samples: int
    num_correct: int
    accuracy: float
    elapsed_sec: float
    config: dict
    sample_outputs: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    error: str | None = None


def run_experiment(
    group: str,
    samples: list[dict],
    model_path: str,
    do_diagnose: bool = False,
) -> ExperimentResult:
    """Run a single experiment group."""
    from vllm import LLM, SamplingParams

    group_name = GROUPS[group]
    print(f"\n{'='*60}")
    print(f"  Group {group}: {group_name}")
    print(f"  Samples: {len(samples)}")
    print(f"{'='*60}")

    llm_kwargs = get_llm_kwargs(group, model_path)
    config_snapshot = {k: v for k, v in llm_kwargs.items() if k != "model"}

    # Prepare prebuilt plan for D/E
    plan_path = prepare_prebuilt_plan(group, model_path)
    update_config_with_plan(llm_kwargs, plan_path)

    print(f"  Config: {json.dumps(config_snapshot, indent=2)}")

    try:
        t0 = time.time()
        llm = LLM(**llm_kwargs)

        # Diagnostics
        diag = {}
        if do_diagnose:
            print("  Running rotation diagnostics...")
            diag = diagnose_rotation(llm, group)
            print(f"    Layers checked: {diag.get('layers_checked', 0)}")
            print(f"    Layers with rotation: {diag.get('layers_with_rotation', 0)}")
            for s in diag.get("sample_pairs_shapes", []):
                print(f"    {s['layer']}: pairs={s['pairs_shape']}, "
                      f"numel={s['pairs_numel']}")

        # Build prompts
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=["<|endoftext|>", "<|im_end|>"],
        )

        # Use chat-style prompts
        prompts = []
        for s in samples:
            prompts.append(s["prompt_messages"])

        print(f"  Generating {len(prompts)} responses...")
        outputs = llm.chat(prompts, sampling_params=sampling_params)

        elapsed = time.time() - t0
        print(f"  Generation done in {elapsed:.1f}s")

        # Score
        correct = 0
        sample_outputs = []
        for i, (output, sample) in enumerate(zip(outputs, samples)):
            gen_text = output.outputs[0].text
            predicted = extract_answer(gen_text)
            gt = normalize_answer(sample["ground_truth"])
            pred_norm = normalize_answer(predicted) if predicted else ""
            is_correct = pred_norm == gt

            if is_correct:
                correct += 1

            # Save sample outputs for first 5
            if i < 5:
                sample_outputs.append({
                    "index": i,
                    "question": sample["question"][:100] + "...",
                    "ground_truth": gt,
                    "predicted": predicted,
                    "correct": is_correct,
                    "output_snippet": gen_text[:200],
                })

        accuracy = correct / len(samples) if samples else 0.0
        print(f"  Accuracy: {correct}/{len(samples)} = {accuracy:.1%}")

        result = ExperimentResult(
            group=group,
            group_name=group_name,
            num_samples=len(samples),
            num_correct=correct,
            accuracy=accuracy,
            elapsed_sec=elapsed,
            config=config_snapshot,
            sample_outputs=sample_outputs,
            diagnostics=diag,
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"  ERROR: {e}")
        print(tb)
        result = ExperimentResult(
            group=group,
            group_name=group_name,
            num_samples=len(samples),
            num_correct=0,
            accuracy=0.0,
            elapsed_sec=0.0,
            config=config_snapshot,
            error=f"{type(e).__name__}: {e}\n{tb}",
        )

    # Cleanup GPU memory
    if "llm" in dir():
        del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def print_summary(results: list[ExperimentResult]) -> None:
    """Print summary table."""
    print(f"\n{'='*70}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Group':<6} {'Name':<22} {'Accuracy':>10} {'Correct':>8} "
          f"{'Total':>6} {'Time(s)':>8} {'Status':>8}")
    print(f"  {'-'*6} {'-'*22} {'-'*10} {'-'*8} {'-'*6} {'-'*8} {'-'*8}")

    for r in results:
        status = "ERROR" if r.error else "OK"
        print(f"  {r.group:<6} {r.group_name:<22} {r.accuracy:>9.1%} "
              f"{r.num_correct:>8} {r.num_samples:>6} "
              f"{r.elapsed_sec:>7.1f}s {status:>8}")

    print(f"{'='*70}")


def save_results(results: list[ExperimentResult], output_path: str) -> None:
    """Save detailed results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise FP4 GSM8K evaluation"
    )
    parser.add_argument(
        "--group", type=str, default="all",
        help="Experiment group: A/B/C/D/E or 'all' (default: all)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of samples (0 = full dataset, default: 20)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Model path",
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA,
        help="GSM8K test parquet path",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Output JSON path (default: auto-generated)",
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Run rotation diagnostics after model loading",
    )
    args = parser.parse_args()

    # Determine groups to run
    if args.group.lower() == "all":
        groups = list(GROUPS.keys())
    else:
        groups = [g.strip().upper() for g in args.group.split(",")]
        for g in groups:
            if g not in GROUPS:
                print(f"Error: unknown group '{g}'. Valid: {list(GROUPS.keys())}")
                sys.exit(1)

    # Load data
    print(f"Loading GSM8K from {args.data}...")
    samples = load_gsm8k(args.data, args.num_samples)
    print(f"Loaded {len(samples)} samples")

    # Run experiments
    results = []
    for group in groups:
        result = run_experiment(
            group=group,
            samples=samples,
            model_path=args.model,
            do_diagnose=args.diagnose,
        )
        results.append(result)

    # Print summary
    print_summary(results)

    # Save results
    if not args.output:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"pairwise_fp4_gsm8k_results_{ts}.json"
    save_results(results, args.output)


if __name__ == "__main__":
    main()
