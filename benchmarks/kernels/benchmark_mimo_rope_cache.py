# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark MiMo partial RoPE, V scaling, packed-cache update and attention.

Run from the repository with an installed vLLM GPU environment:
    .venv/bin/python -m benchmarks.kernels.benchmark_mimo_rope_cache \
        --backend both --output mimo-rope-cache.json

This measures a layer's preparation/attention pipeline, excluding projections,
metadata construction and model serving. Graph samples have warm intra-sample L2.
"""

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)
from vllm.v1.attention.ops.triton_rope_and_cache_diffkv import (
    triton_rope_and_cache_diffkv,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)

CASES = {
    # batch, query length, KV length, Q heads, KV heads, SWA128
    "full-decode": (1, 1, 32768, 64, 4, False),
    "swa-decode": (1, 1, 128, 64, 8, True),
    "swa-batch8": (8, 1, 128, 64, 8, True),
    "swa-tp-batch64": (64, 1, 128, 8, 1, True),
    "full-verify": (4, 8, 4096, 8, 1, False),
    "swa-prefill": (1, 256, 512, 64, 8, True),
    "prefill-2048-swa": (1, 2048, 4096, 64, 8, True),
    "prefill-8192-swa": (1, 8192, 8192, 64, 8, True),
}
COMPILE_OPTIONS = {"combo_kernels": True, "benchmark_combo_kernel": True}
CALLS_PER_GRAPH = 8


def prepare(qkv, positions, cos_sin, q_heads, kv_heads):
    q, k, v = qkv.split([q_heads * 192, kv_heads * 192, kv_heads * 128], -1)
    q, k = RotaryEmbedding.forward_static(positions, q, k, 192, 64, cos_sin, True)
    return (
        q.view(-1, q_heads, 192),
        k.view(-1, kv_heads, 192),
        (v * 0.707).view(-1, kv_heads, 128),
    )


def capture(fn):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(CALLS_PER_GRAPH):
            fn()
    return graph


def measure(graphs, rounds, repeats):
    samples = {name: [] for name in graphs}
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device="cuda")
    orders = []
    for index in range(rounds):
        order = list(graphs) if index % 2 == 0 else list(reversed(graphs))
        orders.append(order)
        for name in order:
            flush.zero_()
            graphs[name].replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(repeats):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(
                start.elapsed_time(end) * 1000 / repeats / CALLS_PER_GRAPH
            )
    return {
        name: {"median_us": statistics.median(values), "samples_us": values}
        for name, values in samples.items()
    }, orders


def benchmark(case, backend, native_prepare, rounds, repeats):
    batch, qlen, seqlen, qh, kh, swa = CASES[case]
    repeats = min(5, repeats) if qlen >= 2048 else repeats
    tokens, pages_per_seq = batch * qlen, (seqlen + 15) // 16
    qkv = torch.randn(tokens, qh * 192 + kh * 320, dtype=torch.bfloat16)
    q, k, v = qkv.split([qh * 192, kh * 192, kh * 128], -1)
    q, k, v = q.view(tokens, qh, 192), k.view(tokens, kh, 192), v.view(tokens, kh, 128)
    query_out = torch.empty_like(q)
    inv = 1 / (1000000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    frequencies = torch.arange(seqlen, dtype=torch.float32)[:, None] * inv
    cos_sin = torch.cat([frequencies.cos(), frequencies.sin()], -1).bfloat16()
    positions = torch.arange(seqlen - qlen, seqlen, dtype=torch.int64).repeat(batch)
    cache = torch.randn(batch * pages_per_seq, 16, kh, 320, dtype=torch.bfloat16)
    slots = (
        torch.arange(batch)[:, None] * pages_per_seq * 16
        + torch.arange(seqlen - qlen, seqlen)[None, :]
    ).flatten()
    starts = torch.arange(batch + 1, dtype=torch.int32) * qlen
    lengths = torch.full((batch,), seqlen, dtype=torch.int32)
    tables = torch.arange(batch * pages_per_seq, dtype=torch.int32).view(batch, -1)
    output = torch.empty(tokens, qh, 128, dtype=torch.bfloat16)
    sinks = torch.randn(qh, dtype=torch.float32) if swa else None
    seg_out = torch.empty(128, qh, 16, 128, dtype=torch.float32)
    seg_max = torch.empty(128, qh, 16, dtype=torch.float32)
    seg_sum, scale = torch.empty_like(seg_max), torch.ones((), dtype=torch.float32)
    fa_version = None
    if backend == "flash":
        fa_version = get_flash_attn_version(
            head_size=192, head_size_v=128, has_sinks=swa
        )
        if fa_version not in (3, 4):
            raise RuntimeError(f"DiffKV requires FA3/FA4, got {fa_version}")
    key_cache = canonicalize_singleton_dim_strides(cache[..., :192])
    value_cache = canonicalize_singleton_dim_strides(cache[..., 192:])

    def attention(query):
        common = dict(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=starts,
            seqused_k=lengths,
            softmax_scale=192**-0.5,
            causal=True,
            block_table=tables,
            max_seqlen_q=qlen,
        )
        if backend == "flash":
            from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

            flash_attn_varlen_func(
                **common,
                max_seqlen_k=seqlen,
                fa_version=fa_version,
                window_size=[127, 0] if swa else [-1, -1],
                s_aux=sinks,
            )
        else:
            unified_attention_diffkv(
                **common,
                window_size=(127, 0) if swa else (-1, -1),
                sinks=sinks,
                softcap=0.0,
                seq_threshold_3D=128,
                num_par_softmax_segments=16,
                softmax_segm_output=seg_out,
                softmax_segm_max=seg_max,
                softmax_segm_expsum=seg_sum,
            )

    def baseline():
        qq, kk, vv = native_prepare(qkv, positions, cos_sin, qh, kh)
        triton_reshape_and_cache_flash_diffkv(
            kk, vv, cache, slots, "auto", scale, scale
        )
        attention(qq)
        return qq

    def fused():
        triton_rope_and_cache_diffkv(
            q,
            k,
            v,
            positions,
            cos_sin,
            cache,
            slots,
            0.707,
            True,
            query_out=query_out,
        )
        attention(query_out)

    expected_q = baseline().clone()
    expected_cache, expected_output = cache.clone(), output.clone()
    original_input = qkv.clone()
    fused()
    correctness = {}
    for name, actual, expected in (
        ("q", query_out, expected_q),
        ("cache", cache, expected_cache),
        ("attention", output, expected_output),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        correctness[name] = {
            "bitwise": torch.equal(actual, expected),
            "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
        }
    torch.testing.assert_close(qkv, original_input, atol=0, rtol=0)
    # Both JIT compilation and graph capture finish before timed samples.
    graphs = {
        "A_native": capture(baseline),
        "B_fused": capture(fused),
        "A_control": capture(baseline),
    }
    timings, orders = measure(graphs, rounds, repeats)
    return {
        "case": case,
        "backend": backend,
        "fa_version": fa_version,
        "batch": batch,
        "query_len": qlen,
        "seq_len": seqlen,
        "q_heads": qh,
        "kv_heads": kh,
        "swa128": swa,
        "repeats": repeats,
        "rounds": rounds,
        "correctness": correctness,
        "input_unchanged": True,
        "timings": timings,
        "round_orders": orders,
        "speedup": timings["A_native"]["median_us"] / timings["B_fused"]["median_us"],
        "aa_control_ratio": timings["A_native"]["median_us"]
        / timings["A_control"]["median_us"],
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["triton", "flash", "both"], default="both"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--case", choices=list(CASES), nargs="+", default=list(CASES))
    args = parser.parse_args()
    if args.repeats < 1 or args.rounds < 2:
        parser.error("repeats must be positive and rounds must be at least two")
    torch.manual_seed(472)
    torch.set_default_device("cuda")
    result = {
        "metadata": {
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "vllm": importlib.metadata.version("vllm"),
            "seed": 472,
            "dtype": "bfloat16",
            "head_dim": 192,
            "value_dim": 128,
            "rotary_dim": 64,
            "value_scale": 0.707,
            "compile_options": COMPILE_OPTIONS,
            "calls_per_graph": CALLS_PER_GRAPH,
            "repeats": args.repeats,
            "rounds": args.rounds,
            "correctness_tolerance": {"atol": 0.02, "rtol": 0.02},
            "timing": (
                "CUDA events; 256MiB flush before each sample; warm intra-sample L2; "
                "no projection/metadata/serving"
            ),
        },
        "results": [],
    }
    native_prepare = torch.compile(prepare, fullgraph=True, options=COMPILE_OPTIONS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for backend in ["triton", "flash"] if args.backend == "both" else [args.backend]:
        for case in args.case:
            row = benchmark(case, backend, native_prepare, args.rounds, args.repeats)
            result["results"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
