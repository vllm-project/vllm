#!/usr/bin/env python3
"""Compare layer outputs between two backend dumps.

Loads manifest files from two dump runs and computes per-tensor diff metrics.

Usage:
    python tools/compare_layer_outputs.py \
        --baseline ./layer_dumps/cuda_manifest.json \
        --target ./layer_dumps/npu_v1_manifest.json \
        --output ./layer_dumps/comparison_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def compute_metrics(
    baseline: torch.Tensor, target: torch.Tensor, atol: float, rtol: float
) -> dict:
    """Compute comparison metrics between two tensors."""
    if baseline.shape != target.shape:
        return {
            "shape_match": False,
            "baseline_shape": list(baseline.shape),
            "target_shape": list(target.shape),
            "pass": False,
        }

    # Cast to float32 for stable numeric comparison
    b = baseline.float()
    t = target.float()
    diff = (b - t).abs()

    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    # Cosine similarity (flatten to 1D)
    b_flat = b.flatten()
    t_flat = t.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        b_flat.unsqueeze(0), t_flat.unsqueeze(0)
    ).item()

    allclose = torch.allclose(b, t, atol=atol, rtol=rtol)

    return {
        "shape_match": True,
        "shape": list(baseline.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "cosine_sim": cos_sim,
        "allclose": allclose,
        "pass": allclose,
    }


def load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_tensor_path(manifest_path: Path, rel_path: str) -> Path:
    """Resolve a relative tensor path against the manifest's directory."""
    return manifest_path.parent / rel_path


def collect_tensor_pairs(
    baseline_manifest: dict,
    target_manifest: dict,
    baseline_root: Path,
    target_root: Path,
    steps_filter: str,
) -> list[tuple[str, str, Path, Path]]:
    """Collect matching (step, tensor_name, baseline_path, target_path) pairs."""
    pairs = []

    baseline_steps = baseline_manifest.get("steps", {})
    target_steps = target_manifest.get("steps", {})

    for step_name in sorted(baseline_steps.keys()):
        if steps_filter == "prefill" and step_name != "prefill":
            continue
        if steps_filter == "decode" and step_name == "prefill":
            continue
        if step_name not in target_steps:
            continue

        b_step = baseline_steps[step_name]
        t_step = target_steps[step_name]

        # Direct tensors at step level (norm, lm_head etc.)
        for key in b_step:
            if key == "layers":
                continue
            if key in t_step and isinstance(b_step[key], str):
                pairs.append((
                    step_name,
                    key,
                    resolve_tensor_path(baseline_root, b_step[key]),
                    resolve_tensor_path(target_root, t_step[key]),
                ))

        # Layer-level tensors
        b_layers = b_step.get("layers", {})
        t_layers = t_step.get("layers", {})
        for layer_name in sorted(b_layers.keys()):
            if layer_name not in t_layers:
                continue
            for tensor_name in b_layers[layer_name]:
                if tensor_name not in t_layers[layer_name]:
                    continue
                pairs.append((
                    step_name,
                    f"{layer_name}/{tensor_name}",
                    resolve_tensor_path(
                        baseline_root, b_layers[layer_name][tensor_name]
                    ),
                    resolve_tensor_path(
                        target_root, t_layers[layer_name][tensor_name]
                    ),
                ))

    return pairs


def print_table(results: list[dict]):
    """Print a compact terminal table of results."""
    header = f"{'Step':<20} {'Layer/Tensor':<30} {'MaxDiff':>10} {'MeanDiff':>10} {'CosSim':>8} {'Status':>6}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        if not r["metrics"].get("shape_match", True):
            status = "SHAPE!"
            print(
                f"{r['step']:<20} {r['tensor']:<30} {'N/A':>10} {'N/A':>10} "
                f"{'N/A':>8} {status:>6}"
            )
        else:
            m = r["metrics"]
            status = "PASS" if m["pass"] else "FAIL"
            print(
                f"{r['step']:<20} {r['tensor']:<30} "
                f"{m['max_abs_diff']:>10.6f} {m['mean_abs_diff']:>10.6f} "
                f"{m['cosine_sim']:>8.6f} {status:>6}"
            )

    print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description="Compare layer outputs between two backend dumps"
    )
    parser.add_argument(
        "--baseline", required=True, help="Path to baseline manifest.json"
    )
    parser.add_argument(
        "--target", required=True, help="Path to target manifest.json"
    )
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--output", default=None, help="Output JSON report path")
    parser.add_argument(
        "--steps",
        choices=["all", "prefill", "decode"],
        default="all",
        help="Which steps to compare",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    target_path = Path(args.target)

    baseline_manifest = load_manifest(baseline_path)
    target_manifest = load_manifest(target_path)

    print(
        f"Comparing: {baseline_manifest['backend']} (baseline) "
        f"vs {target_manifest['backend']} (target)"
    )
    print(f"Model: {baseline_manifest['model']}")
    print(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    print()

    pairs = collect_tensor_pairs(
        baseline_manifest,
        target_manifest,
        baseline_path,
        target_path,
        args.steps,
    )

    if not pairs:
        print("No matching tensor pairs found. Check manifests.")
        sys.exit(1)

    results = []
    pass_count = 0
    fail_count = 0

    for step_name, tensor_name, b_path, t_path in pairs:
        if not b_path.exists():
            print(f"WARNING: baseline tensor not found: {b_path}")
            continue
        if not t_path.exists():
            print(f"WARNING: target tensor not found: {t_path}")
            continue

        b_tensor = torch.load(b_path, map_location="cpu", weights_only=True)
        t_tensor = torch.load(t_path, map_location="cpu", weights_only=True)

        metrics = compute_metrics(b_tensor, t_tensor, args.atol, args.rtol)
        results.append({
            "step": step_name,
            "tensor": tensor_name,
            "metrics": metrics,
        })

        if metrics.get("pass", False):
            pass_count += 1
        else:
            fail_count += 1

    print_table(results)
    print(f"\nSummary: {pass_count} passed, {fail_count} failed, {len(results)} total")

    # Save JSON report
    if args.output:
        report = {
            "baseline": str(baseline_path),
            "target": str(target_path),
            "baseline_backend": baseline_manifest["backend"],
            "target_backend": target_manifest["backend"],
            "model": baseline_manifest["model"],
            "atol": args.atol,
            "rtol": args.rtol,
            "summary": {
                "total": len(results),
                "passed": pass_count,
                "failed": fail_count,
            },
            "results": results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
