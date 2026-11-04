# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the Kimi-K3 latent MoE addmm against CuTe residual GEMM.

The benchmark covers ``BF16[M, 3584] @ BF16[7168, 3584].T + BF16[M, 7168]``
with FP32 accumulation and BF16 output. Both backends execute through CUDA
Graph replay. Weights and residuals rotate across buffers exceeding L2 so the
comparison models the full latent MoE projection-and-add path.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cuda.bindings.driver import CUstream
from quack.compile_utils import make_fake_tensor

N = 7168
K = 3584


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    block_size: int
    outputs_per_block: int
    k_unroll: int
    vector_width: int = 8


def parse_config(value: str) -> Config:
    try:
        parts = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "config must be BLOCK,OUTPUTS,K_UNROLL[,VECTOR_WIDTH]"
        ) from error
    if len(parts) == 3:
        return Config(*parts)
    if len(parts) == 4:
        return Config(*parts)
    raise argparse.ArgumentTypeError(
        "config must be BLOCK,OUTPUTS,K_UNROLL[,VECTOR_WIDTH]"
    )


def production_residual_config(m: int) -> Config | None:
    """The measured Latent-MoE residual config for M, from the K3 table."""
    from vllm.models.kimi_k3.nvidia.low_latency_gemm import KIMI_K3_PROJECTIONS

    spec = KIMI_K3_PROJECTIONS.get((N, K))
    config = spec.residual_config(m) if spec is not None else None
    if config is None:
        return None
    return Config(
        config.block_size,
        config.outputs_per_block,
        config.k_unroll,
        config.vector_width,
    )


def candidate_configs(mode: str, selected: Config | None, m: int) -> list[Config]:
    if mode == "selected":
        if selected is not None:
            return [selected]
        # No explicit --config: fall back to the production table for this M.
        config = production_residual_config(m)
        return [config] if config is not None else []
    if mode == "baseline":
        return [Config(224, 4, 2)]
    return [
        Config(block_size, outputs_per_block, k_unroll, vector_width)
        for vector_width in (4, 8)
        for block_size in (32, 64, 128, 224, 448)
        if block_size % 32 == 0 and K % (block_size * vector_width) == 0
        for outputs_per_block in (1, 2, 4, 7, 8)
        if N % outputs_per_block == 0
        for k_unroll in (1, 2, 4)
    ]


def load_kernel_class(path: Path):
    spec = importlib.util.spec_from_file_location("cute_skinny_device", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CuTe kernel from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CuteSkinnyGemm


def stream() -> CUstream:
    return CUstream(torch.cuda.current_stream().cuda_stream)


def compile_kernel(kernel_class, m: int, config: Config, max_registers: int):
    element_type = cutlass.BFloat16
    n = cute.sym_int(divisibility=config.outputs_per_block)
    k = cute.sym_int(divisibility=config.block_size * config.vector_width)
    a = make_fake_tensor(element_type, (m, k), divisibility=config.vector_width)
    b = make_fake_tensor(element_type, (n, k), divisibility=config.vector_width)
    residual = make_fake_tensor(element_type, (m, n), divisibility=1)
    c = make_fake_tensor(element_type, (m, n), divisibility=1)
    kernel = kernel_class(
        element_type=element_type,
        num_rows=m,
        block_size=config.block_size,
        outputs_per_block=config.outputs_per_block,
        vector_width=config.vector_width,
        k_unroll=config.k_unroll,
        has_residual=True,
        use_pdl=True,
    )
    return cute.compile(
        kernel,
        a,
        b,
        residual,
        c,
        stream(),
        options=(
            "--enable-tvm-ffi --keep-cubin "
            f"--ptxas-options -maxrregcount={max_registers} "
            "--ptxas-options -lineinfo"
        ),
    )


def resource_usage(compiled) -> dict[str, Any]:
    executor = getattr(compiled, "_default_executor", None)
    context = getattr(executor, "exec_context", None)
    functions = getattr(context, "kernel_functions", None)
    if not functions:
        return {"resource_metrics_available": False}

    def attribute(name, function) -> int:
        error, value = cuda.cuFuncGetAttribute(name, function)
        if error != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuFuncGetAttribute failed with {error}")
        return int(value)

    registers = [
        attribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function)
        for function in functions
    ]
    local_bytes = [
        attribute(
            cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            function,
        )
        for function in functions
    ]
    return {
        "resource_metrics_available": True,
        "registers_per_thread": max(registers, default=0),
        "spill_bytes": max(local_bytes, default=0),
    }


def rotating_buffer_count(m: int, multiplier: float, limit: int) -> int:
    properties = torch.cuda.get_device_properties(0)
    bytes_per_pair = (N * K + m * N) * 2
    target = math.ceil(multiplier * properties.L2_cache_size)
    return max(2, min(limit, math.ceil(target / bytes_per_pair)))


def graph_samples(
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None],
    activation: torch.Tensor,
    weights: Sequence[torch.Tensor],
    residuals: Sequence[torch.Tensor],
    repeats: int,
    replays: int,
) -> tuple[list[float], list[torch.Tensor]]:
    outputs = [torch.empty_like(residual) for residual in residuals]
    for weight, residual, output in zip(weights, residuals, outputs):
        launch(activation, weight, residual, output)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for weight, residual, output in zip(weights, residuals, outputs):
            launch(activation, weight, residual, output)
    for _ in range(20):
        graph.replay()
    torch.accelerator.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / (replays * len(weights)))
    return samples, outputs


def summarize(samples: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    mean = statistics.mean(samples)
    return {
        "median_us": statistics.median(samples),
        "p10_us": percentile(0.1),
        "p90_us": percentile(0.9),
        "mean_us": mean,
        "cv_pct": statistics.pstdev(samples) / mean * 100.0,
        "samples_us": list(samples),
    }


def correctness(
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
) -> dict[str, Any]:
    actual = output.float()
    reference = activation.float() @ weight.float().t() + residual.float()
    error = (actual - reference).abs()
    scaled_error = error / (reference.abs() + 1.0)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    ).item()
    return {
        "valid": cosine > 0.999,
        "cosine": cosine,
        "max_abs_error": error.max().item(),
        "max_scaled_error": scaled_error.max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "sweep", "selected"), default="baseline"
    )
    parser.add_argument("--config", type=parse_config)
    parser.add_argument("--m", type=int, action="append")
    parser.add_argument("--config-shard", type=int, default=0)
    parser.add_argument("--num-config-shards", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--cache-multiplier", type=float, default=3.0)
    parser.add_argument("--max-buffers", type=int, default=32)
    parser.add_argument("--max-registers", type=int, default=64)
    args = parser.parse_args()

    token_counts = args.m or list(range(1, 17))
    if any(not 1 <= m <= 16 for m in token_counts):
        raise ValueError("expected 1 <= M <= 16")
    if not 0 <= args.config_shard < args.num_config_shards:
        raise ValueError("config shard must be in [0, num_config_shards)")
    torch.accelerator.set_device_index(0)
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("this benchmark requires SM103")

    kernel_class = load_kernel_class(args.kernel)
    properties = torch.cuda.get_device_properties(0)
    metadata = {
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for m in token_counts:
            configs = candidate_configs(args.mode, args.config, m)
            torch.manual_seed(20260722 + m)
            count = rotating_buffer_count(m, args.cache_multiplier, args.max_buffers)
            activation = torch.randn((m, K), device="cuda", dtype=torch.bfloat16)
            weights = [
                torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
                for _ in range(count)
            ]
            residuals = [
                torch.randn((m, N), device="cuda", dtype=torch.bfloat16)
                for _ in range(count)
            ]
            candidates: list[tuple[str, Config | None]] = [("cublas_addmm", None)]
            candidates.extend(
                ("cute_residual", config)
                for index, config in enumerate(configs)
                if index % args.num_config_shards == args.config_shard
            )
            for backend, config in candidates:
                row: dict[str, Any] = {
                    "m": m,
                    "n": N,
                    "k": K,
                    "backend": backend,
                    "mode": args.mode,
                    "config": dataclasses.asdict(config) if config else {},
                    "num_buffers": count,
                    "cache_multiplier": args.cache_multiplier,
                    **metadata,
                }
                try:
                    if backend == "cublas_addmm":
                        launch = lambda a, b, residual, c: torch.addmm(
                            residual, a, b.t(), out=c
                        )
                    else:
                        if config is None:
                            raise AssertionError("missing CuTe config")
                        compiled = compile_kernel(
                            kernel_class, m, config, args.max_registers
                        )
                        launch = lambda a, b, residual, c, fn=compiled: fn(
                            a, b, residual, c, stream()
                        )
                        row.update(resource_usage(compiled))
                    samples, outputs = graph_samples(
                        launch,
                        activation,
                        weights,
                        residuals,
                        args.repeats,
                        args.replays,
                    )
                    row.update(
                        correctness(outputs[0], activation, weights[0], residuals[0])
                    )
                    row.update(summarize(samples))
                except Exception as error:  # noqa: BLE001
                    row.update(
                        {
                            "valid": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                output_file.write(json.dumps(row, sort_keys=True) + "\n")
                output_file.flush()
                print(json.dumps(row, sort_keys=True), flush=True)

            del activation, weights, residuals
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
