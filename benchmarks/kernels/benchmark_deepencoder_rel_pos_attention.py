# SPDX-License-Identifier: Apache-2.0
"""Benchmark dense and decomposed-bias DeepEncoder attention paths."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from vllm.model_executor.models.deepencoder import add_decomposed_rel_pos

flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)

SHAPES = {
    "tiny": (1, 4, 4),
    "window": (25, 14, 14),
    "global": (1, 64, 64),
}


def dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_h: torch.Tensor,
    rel_w: torch.Tensor,
) -> torch.Tensor:
    bias = (rel_h + rel_w).flatten(-2)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)


def flex_candidate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_h: torch.Tensor,
    rel_w: torch.Tensor,
    key_width: int,
) -> torch.Tensor:
    compact_h = rel_h.squeeze(-1)
    compact_w = rel_w.squeeze(-2)

    def score_mod(score, b, h, q_idx, kv_idx):
        key_h = kv_idx // key_width
        key_w = kv_idx % key_width
        return score + compact_h[b, h, q_idx, key_h] + compact_w[b, h, q_idx, key_w]

    return flex_attention_compiled(q, k, v, score_mod=score_mod)


def make_inputs(
    batch: int,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    device = torch.device("cuda")
    heads = 12
    dim = 64
    tokens = height * width
    q_flat = torch.randn(batch * heads, tokens, dim, device=device, dtype=dtype)
    k_flat = torch.randn_like(q_flat)
    v_flat = torch.randn_like(q_flat)
    rel_pos_h = torch.randn(2 * height - 1, dim, device=device, dtype=dtype) * 0.02
    rel_pos_w = torch.randn(2 * width - 1, dim, device=device, dtype=dtype) * 0.02
    rel_h, rel_w = add_decomposed_rel_pos(
        q_flat, rel_pos_h, rel_pos_w, (height, width), (height, width)
    )
    q = q_flat.view(batch, heads, tokens, dim)
    k = k_flat.view(batch, heads, tokens, dim)
    v = v_flat.view(batch, heads, tokens, dim)
    rel_h = rel_h.view(batch, heads, tokens, height, 1)
    rel_w = rel_w.view(batch, heads, tokens, 1, width)
    return q, k, v, rel_h, rel_w


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / denom).max().item(),
    }


def timed_samples_ms(
    fn: Callable[[], torch.Tensor], warmup: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def peak_memory(fn: Callable[[], torch.Tensor]) -> dict[str, int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    del output
    return {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }


def graph_result(
    fn: Callable[[], torch.Tensor], expected: torch.Tensor
) -> dict[str, Any]:
    try:
        for _ in range(3):
            output = fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn()
        graph.replay()
        torch.cuda.synchronize()
        result = {"ok": True, "errors": errors(output, expected)}
        del output
        return result
    except Exception as exc:
        return {
            "ok": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }


def run_case(name: str, warmup: int, repeats: int) -> dict[str, Any]:
    batch, height, width = SHAPES[name]
    dtype = torch.float32 if name == "tiny" else torch.bfloat16
    q, k, v, rel_h, rel_w = make_inputs(batch, height, width, dtype)
    reference = lambda: dense_reference(q, k, v, rel_h, rel_w)
    candidate = lambda: flex_candidate(q, k, v, rel_h, rel_w, width)

    expected = reference()
    torch.cuda.synchronize()
    compile_start = time.perf_counter()
    actual = candidate()
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start
    correctness = errors(actual, expected)

    rounds = 5
    per_round = max(1, repeats // rounds)
    reference_samples = []
    candidate_samples = []
    for round_idx in range(rounds):
        order = (reference, candidate) if round_idx % 2 == 0 else (candidate, reference)
        for fn in order:
            samples = timed_samples_ms(fn, warmup, per_round)
            if fn is reference:
                reference_samples.extend(samples)
            else:
                candidate_samples.extend(samples)

    reference_memory = peak_memory(reference)
    candidate_memory = peak_memory(candidate)
    reference_graph = graph_result(reference, expected)
    candidate_graph = graph_result(candidate, expected)
    del actual, expected
    torch.cuda.empty_cache()

    return {
        "name": name,
        "batch": batch,
        "height": height,
        "width": width,
        "heads": 12,
        "head_dim": 64,
        "dtype": str(dtype),
        "compile_plus_first_call_seconds": compile_seconds,
        "correctness": correctness,
        "reference_samples_ms": reference_samples,
        "candidate_samples_ms": candidate_samples,
        "reference_median_ms": statistics.median(reference_samples),
        "candidate_median_ms": statistics.median(candidate_samples),
        "reference_peak_bytes": reference_memory,
        "candidate_peak_bytes": candidate_memory,
        "reference_graph": reference_graph,
        "candidate_graph": candidate_graph,
    }


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    report = {
        "environment": {
            "commit": git_commit(),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cases": [run_case(name, args.warmup, args.repeats) for name in args.shapes],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
