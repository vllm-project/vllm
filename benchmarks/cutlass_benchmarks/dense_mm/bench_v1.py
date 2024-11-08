## Cutlass benchmark V1

from typing import Callable, Iterable

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from utils import make_rand_tensors

import vllm._custom_ops as ops


# bench
def bench_fn(label: str, sub_label: str, description: str, fn: Callable, *args,
             **kwargs) -> TMeasurement:
    min_run_time = 1

    globals = {
        "args": args,
        "kwargs": kwargs,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(*args, **kwargs)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench_int8(dtype: torch.dtype, m: int, k: int, n: int, label: str,
               sub_label: str) -> Iterable[TMeasurement]:
    assert dtype == torch.int8
    a, b = make_rand_tensors(torch.int8, m, n, k)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)
    azp = torch.zeros((m, ), device="cuda", dtype=torch.int32)
    azp_adj = torch.zeros((n, ), device="cuda", dtype=torch.int32)

    timers = []
    # pytorch impl - bfloat16
    timers.append(
        bench_fn(label, sub_label, "pytorch_bf16_bf16_bf16_matmul-no-scales",
                 torch.mm, a.to(dtype=torch.bfloat16),
                 b.to(dtype=torch.bfloat16)))

    # pytorch impl - float16
    timers.append(
        bench_fn(label, sub_label,
                 "pytorch_fp16_fp16_fp16_matmul-no-scales", torch.mm,
                 a.to(dtype=torch.float16), b.to(dtype=torch.float16)))

    # cutlass impl
    timers.append(
        bench_fn(label, sub_label, "cutlass_i8_i8_bf16_scaled_mm",
                 ops.cutlass_scaled_mm, a, b, scale_a, scale_b,
                 torch.bfloat16))

    # cutlass with bias
    timers.append(
        bench_fn(label, sub_label, "cutlass_i8_i8_bf16_scaled_mm_bias",
                 ops.cutlass_scaled_mm, a, b, scale_a, scale_b, torch.bfloat16,
                 bias))

    # cutlass with azp per-tensor
    timers.append(
        bench_fn(label, sub_label, "cutlass_i8_i8_bf16_scaled_mm_azp",
                 ops.cutlass_scaled_mm_azp, a, b, scale_a, scale_b,
                 torch.bfloat16, azp_adj))

    # cutlass with azp per-tensor + bias
    timers.append(
        bench_fn(label, sub_label, "cutlass_i8_i8_bf16_scaled_mm_azp_bias",
                 ops.cutlass_scaled_mm_azp, a, b, scale_a, scale_b,
                 torch.bfloat16, azp_adj, None, bias))

    # cutlass with azp per-token
    timers.append(
        bench_fn(label, sub_label, "cutlass_i8_i8_bf16_scaled_mm_azp_pt",
                 ops.cutlass_scaled_mm_azp, a, b, scale_a, scale_b,
                 torch.bfloat16, azp_adj, azp))

    # cutlass with azp per-token + bias
    timers.append(
        bench_fn(label, sub_label, "cutlass_i8_i8_bf16_scaled_mm_azp_pt_bias",
                 ops.cutlass_scaled_mm_azp, a, b, scale_a, scale_b,
                 torch.bfloat16, azp_adj, azp, bias))

    return timers


def bench_fp8(dtype: torch.dtype, m: int, k: int, n: int, label: str,
              sub_label: str) -> Iterable[TMeasurement]:
    assert dtype == torch.float8_e4m3fn
    a, b = make_rand_tensors(torch.float8_e4m3fn, m, n, k)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)

    timers = []

    # pytorch impl w. bf16
    timers.append(
        bench_fn(label, sub_label, "pytorch_bf16_bf16_bf16_matmul-no-scales",
                 torch.mm, a.to(dtype=torch.bfloat16, device="cuda"),
                 b.to(dtype=torch.bfloat16, device="cuda")))

    # pytorch impl: bf16 output, without fp8 fast accum
    timers.append(
        bench_fn(label,
                 sub_label,
                 "pytorch_fp8_fp8_bf16_scaled_mm",
                 torch._scaled_mm,
                 a,
                 b,
                 scale_a=scale_a,
                 scale_b=scale_b,
                 out_dtype=torch.bfloat16))

    # pytorch impl: bf16 output, with fp8 fast accum
    timers.append(
        bench_fn(label,
                 sub_label,
                 "pytorch_fp8_fp8_bf16_scaled_mm_fast_accum",
                 torch._scaled_mm,
                 a,
                 b,
                 scale_a=scale_a,
                 scale_b=scale_b,
                 out_dtype=torch.bfloat16,
                 use_fast_accum=True))

    # pytorch impl: fp16 output, without fp8 fast accum
    timers.append(
        bench_fn(label,
                 sub_label,
                 "pytorch_fp8_fp8_fp16_scaled_mm",
                 torch._scaled_mm,
                 a,
                 b,
                 scale_a=scale_a,
                 scale_b=scale_b,
                 out_dtype=torch.float16))

    # pytorch impl: fp16 output, with fp8 fast accum
    timers.append(
        bench_fn(label,
                 sub_label,
                 "pytorch_fp8_fp8_fp16_scaled_mm_fast_accum",
                 torch._scaled_mm,
                 a,
                 b,
                 scale_a=scale_a,
                 scale_b=scale_b,
                 out_dtype=torch.float16,
                 use_fast_accum=True))

    # cutlass impl: bf16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_fp8_fp8_bf16_scaled_mm",
                 ops.cutlass_scaled_mm, a, b, scale_a, scale_b,
                 torch.bfloat16))
    # cutlass impl: fp16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_fp8_fp8_fp16_scaled_mm",
                 ops.cutlass_scaled_mm, a, b, scale_a, scale_b, torch.float16))

    # cutlass impl: bf16 output, with bias
    timers.append(
        bench_fn(label, sub_label, "cutlass_fp8_fp8_bf16_scaled_mm_bias",
                 ops.cutlass_scaled_mm, a, b, scale_a, scale_b, torch.bfloat16,
                 bias))

    # cutlass impl: fp16 output, with bias
    timers.append(
        bench_fn(label, sub_label, "cutlass_fp8_fp8_fp16_scaled_mm_bias",
                 ops.cutlass_scaled_mm, a, b, scale_a, scale_b, torch.float16,
                 bias.to(dtype=torch.float16)))

    return timers


def bench_v1(dtype: torch.dtype, m: int, k: int, n: int, label: str,
             sub_label: str) -> Iterable[TMeasurement]:
    if dtype == torch.int8:
        return bench_int8(dtype, m, k, n, label, sub_label)
    if dtype == torch.float8_e4m3fn:
        return bench_fp8(dtype, m, k, n, label, sub_label)
    raise ValueError("unsupported type")
