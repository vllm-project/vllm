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

import vllm
from vllm.model_executor.kernels.deepencoder_attention import (
    deepencoder_rel_pos_attention,
)
from vllm.model_executor.models.deepencoder import (
    RelPosAttention,
    add_decomposed_rel_pos,
)

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


def triton_candidate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_h: torch.Tensor,
    rel_w: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    return deepencoder_rel_pos_attention(
        q,
        k,
        v,
        rel_h.squeeze(-1),
        rel_w.squeeze(-2),
        height,
        width,
        q.size(-1) ** -0.5,
    )


def dense_layer_reference(layer: RelPosAttention, inputs: torch.Tensor) -> torch.Tensor:
    batch, height, width, _ = inputs.shape
    qkv = (
        layer.qkv(inputs)
        .reshape(batch, height * width, 3, layer.num_heads, -1)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.reshape(3, batch * layer.num_heads, height * width, -1).unbind(0)
    rel_h, rel_w = add_decomposed_rel_pos(
        q,
        layer.rel_pos_h,
        layer.rel_pos_w,
        (height, width),
        (height, width),
    )
    q = q.view(batch, layer.num_heads, height * width, -1)
    k = k.view(batch, layer.num_heads, height * width, -1)
    v = v.view(batch, layer.num_heads, height * width, -1)
    rel_h = rel_h.view(
        batch, layer.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3)
    )
    rel_w = rel_w.view(
        batch, layer.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3)
    )
    bias = (rel_h + rel_w).flatten(-2)
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    output = (
        output.view(batch, layer.num_heads, height, width, -1)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch, height, width, -1)
    )
    return layer.proj(output)


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


def repeat_counts(repeats: int, max_rounds: int = 5) -> list[int]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rounds = min(max_rounds, repeats)
    per_round, remainder = divmod(repeats, rounds)
    return [per_round + (round_idx < remainder) for round_idx in range(rounds)]


def peak_memory(fn: Callable[[], torch.Tensor]) -> dict[str, int]:
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    del output
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "baseline_allocated": baseline_allocated,
        "baseline_reserved": baseline_reserved,
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
        "incremental_allocated": peak_allocated - baseline_allocated,
        "incremental_reserved": peak_reserved - baseline_reserved,
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


@torch.inference_mode()
def run_case(
    name: str, warmup: int, repeats: int, implementation: str
) -> dict[str, Any]:
    batch, height, width = SHAPES[name]
    dtype = torch.bfloat16
    if implementation == "operator":
        q, k, v, rel_h, rel_w = make_inputs(batch, height, width, dtype)
        reference = lambda: dense_reference(q, k, v, rel_h, rel_w)
        candidate = lambda: triton_candidate(q, k, v, rel_h, rel_w, height, width)
    else:
        torch.manual_seed(0)
        layer = (
            RelPosAttention(
                dim=768,
                num_heads=12,
                use_rel_pos=True,
                use_triton_attention=name == "global",
                input_size=(height, width),
            )
            .cuda()
            .to(dtype)
            .eval()
        )
        layer.rel_pos_h.data.normal_(std=0.02)
        layer.rel_pos_w.data.normal_(std=0.02)
        inputs = torch.randn(
            batch,
            height,
            width,
            768,
            device="cuda",
            dtype=dtype,
        )
        reference = lambda: dense_layer_reference(layer, inputs)
        candidate = lambda: layer(inputs)

    expected = reference()
    torch.cuda.synchronize()
    first_call_start = time.perf_counter()
    actual = candidate()
    torch.cuda.synchronize()
    first_call_seconds = time.perf_counter() - first_call_start
    correctness = errors(actual, expected)

    reference_samples = []
    candidate_samples = []
    for round_idx, round_repeats in enumerate(repeat_counts(repeats)):
        order = (reference, candidate) if round_idx % 2 == 0 else (candidate, reference)
        for fn in order:
            samples = timed_samples_ms(fn, warmup, round_repeats)
            if fn is reference:
                reference_samples.extend(samples)
            else:
                candidate_samples.extend(samples)

    candidate_memory = peak_memory(candidate)
    reference_memory = peak_memory(reference)
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
        "implementation": implementation,
        "first_call_seconds": first_call_seconds,
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


@torch.inference_mode()
def run_memory_only(name: str, path: str, implementation: str) -> dict[str, Any]:
    batch, height, width = SHAPES[name]
    dtype = torch.bfloat16
    if implementation == "operator":
        q, k, v, rel_h, rel_w = make_inputs(batch, height, width, dtype)
        reference = lambda: dense_reference(q, k, v, rel_h, rel_w)
        candidate = lambda: triton_candidate(q, k, v, rel_h, rel_w, height, width)
    else:
        torch.manual_seed(0)
        layer = (
            RelPosAttention(
                dim=768,
                num_heads=12,
                use_rel_pos=True,
                use_triton_attention=name == "global",
                input_size=(height, width),
            )
            .cuda()
            .to(dtype)
            .eval()
        )
        layer.rel_pos_h.data.normal_(std=0.02)
        layer.rel_pos_w.data.normal_(std=0.02)
        inputs = torch.randn(
            batch,
            height,
            width,
            768,
            device="cuda",
            dtype=dtype,
        )
        reference = lambda: dense_layer_reference(layer, inputs)
        candidate = lambda: layer(inputs)
    selected = reference if path == "reference" else candidate
    for _ in range(3):
        selected()
    torch.cuda.synchronize()
    return {
        "name": name,
        "path": path,
        "implementation": implementation,
        "memory": peak_memory(selected),
    }


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--implementation",
        choices=["operator", "production"],
        default="operator",
    )
    parser.add_argument("--memory-only-path", choices=["reference", "candidate"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.memory_only_path:
        cases = [
            run_memory_only(name, args.memory_only_path, args.implementation)
            for name in args.shapes
        ]
    else:
        cases = [
            run_case(name, args.warmup, args.repeats, args.implementation)
            for name in args.shapes
        ]
    report = {
        "environment": {
            "commit": git_commit(),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "vllm_source": vllm.__file__,
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
