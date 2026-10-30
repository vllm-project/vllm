# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-M (decode) NVFP4 linear benchmark across vLLM's NVFP4 kernels.

Each provider is a vLLM ``NvFp4LinearKernel`` driven through
``process_weights_after_loading`` + ``apply_weights``, so the timed region is
exactly what the model runs (activation quant, padding, GEMM). Timing uses
FlashInfer CUPTI on a CUDA graph that walks ``copies`` distinct weight copies
whose total size exceeds L2, so every call reads its weight from DRAM.

Bandwidth is reported against the theoretical minimum traffic: FP4 weight +
E4M3 block scales read once (activation and output are <0.1% at M <= 16).

Example:
    python benchmarks/kernels/benchmark_nvfp4_small_m.py \
        --output-json results/micro/nvfp4_small_m.json

"""

import argparse
import json
import math
import platform
import random
import statistics
import subprocess
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.b12x import B12xNvFp4LinearKernel
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearKernel
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferB12xNvFp4LinearKernel,
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.flashinfer import autotune

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP4_BYTES_PER_PARAM = 0.5 + 1 / 16

# Qwen3.8-27B (hidden 5120), TP1, fused as in vllm/model_executor/models/qwen3_5.py
# name: (K, N, layers per forward)
QWEN3_8_27B_SHAPES = {
    "gdn.in_proj_qkvz": (5120, 16384, 48),
    "gdn.in_proj_ba": (5120, 96, 48),
    "gdn.out_proj": (6144, 5120, 48),
    "attn.qkv": (5120, 14336, 16),
    "attn.o_proj": (6144, 5120, 16),
    "mlp.gate_up": (5120, 34816, 64),
    "mlp.down": (17408, 5120, 64),
}

# provider: (kernel class, autotune, bypass is_supported)
# FlashInferCuteDsl W4A4 is gated to sm_10x in vLLM; flashinfer#5100 added the
# SM12x kernels, so the gate is bypassed here to measure it on SM120.
PROVIDERS: dict[str, tuple[type[NvFp4LinearKernel], bool, bool]] = {
    "fi-cutlass": (FlashInferCutlassNvFp4LinearKernel, False, False),
    "fi-cutlass-tuned": (FlashInferCutlassNvFp4LinearKernel, True, False),
    "fi-b12x": (FlashInferB12xNvFp4LinearKernel, False, False),
    "b12x-native": (B12xNvFp4LinearKernel, False, False),
    "fi-cute-w4a4": (FlashInferCuteDslNvFp4LinearKernel, False, True),
    "fi-cute-w4a4-tuned": (FlashInferCuteDslNvFp4LinearKernel, True, True),
    "fi-cute-w4a16": (FlashInferCuteDslNvFp4W4A16LinearKernel, False, False),
    "fi-cute-w4a16-tuned": (FlashInferCuteDslNvFp4W4A16LinearKernel, True, False),
}

DATASHEET_BW_GBPS = {"workstation": 1792.0, "server": 1597.0}


@dataclass
class Result:
    shape: str
    k: int
    n: int
    m: int
    provider: str
    us: float | None
    us_samples: list[float]
    weight_bytes: int
    gbps: float | None
    pct_bw: dict[str, float]
    max_abs_err: float | None
    cos_sim: float | None
    error: str | None = None


def make_kernel(provider: str) -> NvFp4LinearKernel:
    kernel_cls, _, bypass = PROVIDERS[provider]
    if not bypass:
        return kernel_cls(NvFp4LinearLayerConfig())
    kernel = object.__new__(kernel_cls)
    kernel.config = NvFp4LinearLayerConfig()
    return kernel


def make_layer(
    kernel: NvFp4LinearKernel,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    keep_ref: bool = False,
) -> torch.nn.Module:
    """Build a layer with checkpoint-format params, then convert for *kernel*."""
    n, k = weight.shape
    w_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / weight.abs().max()).float()
    w_fp4, w_sf = ops.scaled_fp4_quant(weight, w_scale, is_sf_swizzled_layout=False)
    layer = torch.nn.Module()
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    p = lambda t: torch.nn.Parameter(t, requires_grad=False)  # noqa: E731
    layer.weight = p(w_fp4)
    layer.weight_scale = p(w_sf)
    layer.weight_global_scale = p(w_scale.reciprocal())
    layer.input_global_scale = p(x_scale.reciprocal())
    layer.input_global_scale_inv = p(x_scale)
    layer.alpha = p((x_scale * w_scale).reciprocal())
    if keep_ref:
        layer.ref_weight = dequantize_to_dtype(
            w_fp4, w_sf, w_scale.reciprocal(), dtype=torch.bfloat16, swizzle=False
        )
    kernel.process_weights_after_loading(layer)
    return layer


def bench_us(fn, warmup: int, iters: int, shuffles: int) -> list[float]:
    from flashinfer.testing import bench_gpu_time_with_cupti

    samples = []
    for _ in range(shuffles):
        for _ in range(warmup):
            fn()
        torch.accelerator.synchronize()
        samples.append(
            statistics.median(bench_gpu_time_with_cupti(fn, repeat_iters=iters)) * 1e3
        )
    return samples


def run_case(shape: str, m: int, provider: str, args: argparse.Namespace) -> Result:
    k, n, _ = QWEN3_8_27B_SHAPES[shape]
    weight_bytes = int(k * n * FP4_BYTES_PER_PARAM)
    base = dict(
        shape=shape, k=k, n=n, m=m, provider=provider, weight_bytes=weight_bytes
    )
    try:
        kernel = make_kernel(provider)
        _, tuned, _ = PROVIDERS[provider]
        copies = max(2, math.ceil(args.rotate_mb * 1e6 / weight_bytes))
        copies = min(copies, args.max_copies)
        xs = [torch.randn(m, k, dtype=torch.bfloat16) for _ in range(copies)]
        x_scale = (
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.stack(xs).abs().max()
        ).float()
        layers = [
            make_layer(
                kernel,
                torch.randn(n, k, dtype=torch.bfloat16) / k**0.5,
                x_scale,
                keep_ref=i == 0,
            )
            for i in range(copies)
        ]

        if tuned:
            with autotune(True):
                for x, layer in zip(xs, layers):
                    kernel.apply_weights(layer, x)
            torch.accelerator.synchronize()

        out = kernel.apply_weights(layers[0], xs[0])
        ref = F.linear(xs[0], layers[0].ref_weight)
        max_abs_err = (out.float() - ref.float()).abs().max().item()
        cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)

        def walk():
            for x, layer in zip(xs, layers):
                kernel.apply_weights(layer, x)

        walk()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            walk()
        samples = [
            s / copies
            for s in bench_us(graph.replay, args.warmup, args.iters, args.shuffles)
        ]
        us = statistics.median(samples)
        gbps = weight_bytes / (us * 1e3)
        return Result(
            **base,
            us=us,
            us_samples=samples,
            gbps=gbps,
            pct_bw={k: 100 * gbps / v for k, v in DATASHEET_BW_GBPS.items()},
            max_abs_err=max_abs_err,
            cos_sim=cos.item(),
        )
    except Exception as e:  # noqa: BLE001
        return Result(
            **base,
            us=None,
            us_samples=[],
            gbps=None,
            pct_bw={},
            max_abs_err=None,
            cos_sim=None,
            error=f"{type(e).__name__}: {e}"[:500],
        )


def run_quant_only(shape: str, m: int, args: argparse.Namespace) -> Result:
    """Standalone scaled_fp4_quant cost (the W4A4 providers' extra launch)."""
    k, n, _ = QWEN3_8_27B_SHAPES[shape]
    x = torch.randn(m, k, dtype=torch.bfloat16)
    scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / x.abs().max()).float()
    fn = lambda: ops.scaled_fp4_quant(x, scale, is_sf_swizzled_layout=True)  # noqa: E731
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    samples = bench_us(graph.replay, args.warmup, args.iters, args.shuffles)
    return Result(
        shape=shape, k=k, n=n, m=m, provider="quant-only",
        us=statistics.median(samples), us_samples=samples, weight_bytes=0,
        gbps=None, pct_bw={}, max_abs_err=None, cos_sim=None,
    )  # fmt: skip


def metadata() -> dict:
    import flashinfer

    def cmd(*c: str) -> str:
        try:
            return subprocess.run(c, capture_output=True, text=True).stdout.strip()
        except OSError:
            return "unavailable"

    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "l2_bytes": getattr(props, "L2_cache_size", None),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "vllm_commit": cmd("git", "rev-parse", "HEAD"),
        "nvidia_smi": cmd(
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,"
            "clocks.mem,power.limit,power.draw",
            "--format=csv,noheader",
        ),
        "host": platform.node(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=list(QWEN3_8_27B_SHAPES))
    parser.add_argument("--m", nargs="+", type=int, default=[1, 2, 3, 4, 8, 12, 16])
    parser.add_argument("--providers", nargs="+", default=list(PROVIDERS))
    parser.add_argument("--rotate-mb", type=float, default=512.0)
    parser.add_argument("--max-copies", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--shuffles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if not current_platform.is_device_capability_family(120):
        raise RuntimeError("This benchmark targets SM12x (RTX PRO 6000 Blackwell).")
    try:
        import cupti  # noqa: F401
    except ImportError as e:
        # Without it FlashInfer silently falls back to CUDA-event timing.
        raise RuntimeError("CUPTI timing requires `uv pip install cupti-python`") from e
    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)

    cases = [(s, m, p) for s in args.shapes for m in args.m for p in args.providers]
    # Shuffle so clock/thermal drift is shared across providers.
    random.Random(args.seed).shuffle(cases)
    results = [run_quant_only(s, m, args) for s in args.shapes for m in args.m]
    for shape, m, provider in cases:
        r = run_case(shape, m, provider, args)
        results.append(r)
        status = r.error or f"{r.us:8.2f} us {r.pct_bw.get('workstation', 0):5.1f}%"
        print(f"{shape:18s} M={m:<3d} {provider:20s} {status}", flush=True)
        torch.accelerator.empty_cache()

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "meta": metadata(),
                    "args": vars(args),
                    "results": [asdict(r) for r in results],
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
