# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DeepSeek-V4 sparse-prefill gather plus attention on SM90.

The BF16 path measures packed-cache dequant/gather followed by FlashMLA. The
Q8KV8 path measures packed-cache dequant/requant, query casting, and the native
FP8 sparse MLA kernel. Sparse-index construction is excluded because it is
shared by both paths.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import statistics
import subprocess
from dataclasses import dataclass

import torch

from vllm.models.deepseek_v4.common.ops import (
    dequantize_and_gather_k_cache,
    dequantize_and_gather_k_cache_fp8,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.nvidia.ops.q8kv8_sparse_prefill import (
    sparse_mla_q8kv8_prefill,
)
from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd


@dataclass(frozen=True)
class Shape:
    queries: int
    gathered_tokens_per_request: int
    heads: int
    topk: int


DEFAULT_SHAPES = (
    Shape(128, 1152, 64, 640),
    Shape(512, 2176, 64, 640),
    Shape(512, 2176, 64, 1152),
)


def _require_cupti() -> str:
    try:
        from cupti import cupti as _cupti_api  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CUPTI timing requires cupti-python>=13; CUDA-event fallback is disabled"
        ) from exc
    version = importlib.metadata.version("cupti-python")
    if int(version.split(".", 1)[0]) < 13:
        raise RuntimeError(f"CUPTI timing requires cupti-python>=13, got {version}")
    return version


def _make_inputs(shape: Shape, batch_size: int):
    if shape.queries % batch_size:
        raise ValueError("queries must be divisible by batch-size")
    if shape.topk > shape.gathered_tokens_per_request:
        raise ValueError("topk cannot exceed gathered tokens per request")

    device = torch.device("cuda")
    block_size = 64
    tokens_per_request = shape.gathered_tokens_per_request
    blocks_per_request = math.ceil(tokens_per_request / block_size)
    num_blocks = batch_size * blocks_per_request
    generator = torch.Generator(device=device)
    generator.manual_seed(20260831 + shape.queries + tokens_per_request + shape.topk)

    source_kv = torch.randn(
        (batch_size * tokens_per_request, 512),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    source_kv.mul_(0.05)
    packed_cache = torch.empty(
        (num_blocks, block_size, 584), dtype=torch.uint8, device=device
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        batch_size, blocks_per_request
    )
    logical_positions = torch.arange(
        tokens_per_request, dtype=torch.int64, device=device
    )
    slot_mapping = torch.cat(
        [
            request * blocks_per_request * block_size + logical_positions
            for request in range(batch_size)
        ]
    )
    quantize_and_insert_k_cache(
        source_kv, packed_cache.view(num_blocks, -1), slot_mapping, block_size
    )

    seq_lens = torch.full(
        (batch_size,), tokens_per_request, dtype=torch.int32, device=device
    )
    q = torch.randn(
        (shape.queries, shape.heads, 512),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    q.mul_(0.05)
    queries_per_request = shape.queries // batch_size
    indices = torch.empty(
        (shape.queries, 1, shape.topk), dtype=torch.int32, device=device
    )
    for request in range(batch_size):
        row_start = request * queries_per_request
        row_end = row_start + queries_per_request
        indices[row_start:row_end, 0] = (
            torch.randint(
                tokens_per_request,
                (queries_per_request, shape.topk),
                dtype=torch.int32,
                device=device,
                generator=generator,
            )
            + request * tokens_per_request
        )

    topk_length = torch.full(
        (shape.queries,), shape.topk, dtype=torch.int32, device=device
    )
    attn_sink = torch.linspace(-0.05, 0.05, shape.heads, device=device)
    return (
        q,
        packed_cache,
        seq_lens,
        block_table,
        indices,
        topk_length,
        attn_sink,
        block_size,
    )


def _benchmark_shape(shape: Shape, batch_size: int) -> None:
    from flashinfer.testing import bench_gpu_time_with_cupti

    (
        q,
        packed_cache,
        seq_lens,
        block_table,
        indices,
        topk_length,
        attn_sink,
        block_size,
    ) = _make_inputs(shape, batch_size)
    kv_shape = (batch_size, shape.gathered_tokens_per_request, 512)
    kv_bf16 = torch.empty(kv_shape, dtype=torch.bfloat16, device="cuda")
    kv_fp8_flat = torch.empty(
        (batch_size * shape.gathered_tokens_per_request + 1, 512),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    kv_fp8 = kv_fp8_flat[:-1].view(kv_shape)
    kv_fp8_flat[-1].zero_()
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    bf16_output = torch.empty_like(q)
    q8_output = torch.empty_like(q)
    max_logits = torch.empty(
        (shape.queries, shape.heads), dtype=torch.float32, device="cuda"
    )
    lse = torch.empty_like(max_logits)
    identity_scale = torch.ones((), dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / math.sqrt(512)

    def bf16_path() -> None:
        dequantize_and_gather_k_cache(
            kv_bf16,
            packed_cache,
            seq_lens,
            None,
            block_table,
            block_size,
            0,
        )
        flash_mla_sparse_fwd(
            q,
            kv_bf16.view(-1, 1, 512),
            indices,
            sm_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=bf16_output,
        )

    def q8kv8_path() -> None:
        dequantize_and_gather_k_cache_fp8(
            kv_fp8,
            packed_cache,
            seq_lens,
            None,
            block_table,
            block_size,
            0,
        )
        kv_fp8_flat[-1].zero_()
        q_fp8.copy_(q)
        sparse_mla_q8kv8_prefill(
            q_fp8,
            kv_fp8_flat.unsqueeze(1),
            indices,
            identity_scale,
            identity_scale,
            attn_sink,
            topk_length,
            sm_scale,
            out=q8_output,
            max_logits=max_logits,
            lse=lse,
        )

    bf16_path()
    q8kv8_path()
    torch.accelerator.synchronize()
    error = (q8_output.float() - bf16_output.float()).abs().flatten()
    mean_error = error.mean().item()
    p99_error = torch.quantile(error, 0.99).item()
    max_error = error.max().item()
    if not torch.isfinite(q8_output.float()).all():
        raise AssertionError("Q8KV8 output contains non-finite values")
    if mean_error >= 0.03 or p99_error >= 0.2:
        raise AssertionError(
            f"correctness gate failed: mean={mean_error:.6f}, p99={p99_error:.6f}"
        )

    for _ in range(3):
        bf16_path()
        q8kv8_path()
    torch.accelerator.synchronize()
    bench = lambda fn: bench_gpu_time_with_cupti(
        fn, use_cuda_graph=True, cold_l2_cache=False
    )
    bf16_ms = statistics.median(bench(bf16_path))
    q8kv8_ms = statistics.median(bench(q8kv8_path))
    print(
        f"{shape.queries:>7} {shape.gathered_tokens_per_request:>8} "
        f"{shape.heads:>5} {shape.topk:>6} "
        f"{bf16_ms * 1e3:>12.1f} {q8kv8_ms * 1e3:>12.1f} "
        f"{bf16_ms / q8kv8_ms:>8.2f}x "
        f"{mean_error:>10.6f} {p99_error:>10.6f} {max_error:>10.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Q8KV8 sparse prefill benchmark requires SM90")
    if not hasattr(torch.ops._C, "sparse_mla_q8kv8_prefill_sm90"):
        raise RuntimeError("vLLM was not built with the Q8KV8 prefill kernel")
    cupti_version = _require_cupti()

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"commit: {commit}")
    print(f"gpu: {torch.cuda.get_device_name()}")
    print(f"capability: {torch.cuda.get_device_capability()}")
    print(f"torch: {torch.__version__}; cuda: {torch.version.cuda}")
    print(
        f"timing: FlashInfer CUPTI {cupti_version}, CUDA graph enabled, "
        "warmup outside timing"
    )
    print("boundary: packed-cache gather/requant + Q cast + sparse attention")
    print(
        f"{'queries':>7} {'kv/req':>8} {'heads':>5} {'topk':>6} "
        f"{'BF16 (us)':>12} {'Q8KV8 (us)':>12} {'speedup':>9} "
        f"{'mean err':>10} {'p99 err':>10} {'max err':>10}"
    )
    for shape in DEFAULT_SHAPES:
        _benchmark_shape(shape, args.batch_size)


if __name__ == "__main__":
    main()
