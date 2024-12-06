## Cutlass benchmark V1

from typing import Callable, Iterable

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from utils import make_rand_sparse_tensors, to_fp16, to_bf16

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

    # Create tensors
    b_compressed, e, a, b = make_rand_sparse_tensors(torch.int8, m, n, k)
    aT = a.t()
    bT = b.t()
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)

    out = ops.cutlass_scaled_sparse_mm(b_compressed, e, aT, scale_b, scale_a, torch.bfloat16)
    out_ref = ops.cutlass_scaled_mm(a, bT, scale_a, scale_b, torch.bfloat16)

    if not torch.allclose(out.t(), out_ref):
        print("Incorrect result")
        exit()

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

    # cutlass impl: bf16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_i8_i8_bf16_scaled_sparse_mm",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a,
                 torch.bfloat16))

    # cutlass with bias: bf16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_i8_i8_bf16_scaled_sparse_mm_bias",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a, torch.bfloat16,
                 bias))
    
    # cutlass impl: fp16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_i8_i8_fp16_scaled_sparse_mm",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a,
                 torch.float16))

    # cutlass with bias: fp16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_i8_i8_fp16_scaled_sparse_mm_bias",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a, torch.float16,
                 bias.to(dtype=torch.float16)))

    return timers


def bench_fp8(dtype: torch.dtype, m: int, k: int, n: int, label: str,
              sub_label: str) -> Iterable[TMeasurement]:
    assert dtype == torch.float8_e4m3fn

    # Create tensors
    b_compressed, e, a, b = make_rand_sparse_tensors(torch.float8_e4m3fn, m, n, k)
    aT = a.t()
    bT = b
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)

    out = ops.cutlass_scaled_sparse_mm(b_compressed, e, aT, scale_b, scale_a, torch.bfloat16)
    out_ref = ops.cutlass_scaled_mm(a, bT, scale_a, scale_b, torch.bfloat16)

    if not torch.allclose(out, out_ref, rtol=1e-2, atol=1e-2):
        print(f"Incorrect result for {m}, {k}, {n}")
        exit()

    timers = []

    # pytorch impl w. bf16
    timers.append(
        bench_fn(label, sub_label, "pytorch_bf16_bf16_bf16_matmul-no-scales",
                 torch.mm, a.to(dtype=torch.bfloat16, device="cuda"),
                 bT.to(dtype=torch.bfloat16, device="cuda")))

    # pytorch impl: bf16 output, without fp8 fast accum
    timers.append(
        bench_fn(label,
                 sub_label,
                 "pytorch_fp8_fp8_bf16_scaled_mm",
                 torch._scaled_mm,
                 a,
                 bT,
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
                 bT,
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
                 bT,
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
                 bT,
                 scale_a=scale_a,
                 scale_b=scale_b,
                 out_dtype=torch.float16,
                 use_fast_accum=True))

    # cutlass impl: bf16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_fp8_fp8_bf16_scaled_sparse_mm",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a,
                 torch.bfloat16))
    # cutlass impl: fp16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_fp8_fp8_fp16_scaled_sparse_mm",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a, torch.float16))

    return timers


def bench_fp16(dtype: torch.dtype, m: int, k: int, n: int, label: str,
              sub_label: str) -> Iterable[TMeasurement]:
    assert dtype == torch.float16

    m, k, n = 1, 128, 256

    # Create tensors
    b_compressed, e, a, b = make_rand_sparse_tensors(torch.float16, m, n, k)
    aT = a.t()
    bT = b.t()
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)

    out = ops.cutlass_scaled_sparse_mm(b_compressed, e, aT, scale_b, scale_a, torch.bfloat16)
    out_ref = to_bf16(a@bT)

    if not torch.allclose(out.t(), out_ref, rtol=1e-2, atol=1e-2):
        print("Incorrect result")
        print(out.t())
        print(out_ref)
        exit()
    else:
        print("Correct result")

    timers = []

    # # pytorch impl w. bf16
    # timers.append(
    #     bench_fn(label, sub_label, "pytorch_bf16_bf16_bf16_matmul-no-scales",
    #              torch.mm, a.to(dtype=torch.bfloat16, device="cuda"),
    #              b.to(dtype=torch.bfloat16, device="cuda")))

    # # pytorch impl: bf16 output
    # timers.append(
    #     bench_fn(label,
    #              sub_label,
    #              "pytorch_fp16_fp16_bf16_scaled_mm",
    #              torch._scaled_mm,
    #              a,
    #              b,
    #              scale_a=scale_a,
    #              scale_b=scale_b,
    #              out_dtype=torch.bfloat16))

    # # pytorch impl: fp16 output
    # timers.append(
    #     bench_fn(label,
    #              sub_label,
    #              "pytorch_fp16_fp16_fp16_scaled_mm",
    #              torch._scaled_mm,
    #              a,
    #              b,
    #              scale_a=scale_a,
    #              scale_b=scale_b,
    #              out_dtype=torch.float16))

    # cutlass impl: bf16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_fp16_fp16_bf16_scaled_sparse_mm",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a,
                 torch.bfloat16))

    # cutlass impl: fp16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_fp16_fp16_fp16_scaled_sparse_mm",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a, torch.float16))

    # # cutlass impl: bf16 output, with bias
    # timers.append(
    #     bench_fn(label, sub_label, "cutlass_fp16_fp16_bf16_scaled_sparse_mm_bias",
    #              ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a, torch.bfloat16,
    #              bias))

    # # cutlass impl: fp16 output, with bias
    # timers.append(
    #     bench_fn(label, sub_label, "cutlass_fp16_fp16_fp16_scaled_sparse_mm_bias",
    #              ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a, torch.float16,
    #              bias.to(dtype=torch.float16)))

    return timers


def bench_bf16(dtype: torch.dtype, m: int, k: int, n: int, label: str,
              sub_label: str) -> Iterable[TMeasurement]:
    assert dtype == torch.bfloat16

    # Create tensors
    b_compressed, e, a, b = make_rand_sparse_tensors(torch.bfloat16, m, n, k)
    aT = a.t()
    bT = b.t()
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)

    out = ops.cutlass_scaled_sparse_mm(b_compressed, e, aT, scale_b, scale_a, torch.bfloat16)
    out_ref = to_bf16(a@bT)

    if not torch.allclose(out.t(), out_ref):
        print("Incorrect result")
        exit()

    timers = []

    # # pytorch impl w. bf16
    # timers.append(
    #     bench_fn(label, sub_label, "pytorch_bf16_bf16_bf16_matmul-no-scales",
    #              torch.mm, a.to(dtype=torch.bfloat16, device="cuda"),
    #              b.to(dtype=torch.bfloat16, device="cuda")))

    # # pytorch impl: bf16 output
    # timers.append(
    #     bench_fn(label,
    #              sub_label,
    #              "pytorch_fp16_fp16_bf16_scaled_mm",
    #              torch._scaled_mm,
    #              a,
    #              b,
    #              scale_a=scale_a,
    #              scale_b=scale_b,
    #              out_dtype=torch.bfloat16))

    # # pytorch impl: fp16 output
    # timers.append(
    #     bench_fn(label,
    #              sub_label,
    #              "pytorch_fp16_fp16_fp16_scaled_mm",
    #              torch._scaled_mm,
    #              a,
    #              b,
    #              scale_a=scale_a,
    #              scale_b=scale_b,
    #              out_dtype=torch.float16))

    # cutlass impl: bf16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_bf16_bf16_bf16_scaled_sparse_mm",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a,
                 torch.bfloat16))

    # cutlass impl: fp16 output
    timers.append(
        bench_fn(label, sub_label, "cutlass_bf16_bf16_fp16_scaled_sparse_mm",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a, torch.float16))

    # cutlass impl: bf16 output, with bias
    timers.append(
        bench_fn(label, sub_label, "cutlass_bf16_bf16_bf16_scaled_sparse_mm_bias",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a, torch.bfloat16,
                 bias))

    # cutlass impl: fp16 output, with bias
    timers.append(
        bench_fn(label, sub_label, "cutlass_bf16_bf16_fp16_scaled_sparse_mm_bias",
                 ops.cutlass_scaled_sparse_mm, b_compressed, e, aT, scale_b, scale_a, torch.float16,
                 bias.to(dtype=torch.float16)))

    return timers


def bench_v1(dtype: torch.dtype, m: int, k: int, n: int, label: str,
          sub_label: str) -> Iterable[TMeasurement]:
    if dtype == torch.int8:
        return bench_int8(dtype, m, k, n, label, sub_label)
    if dtype == torch.float8_e4m3fn:
        return bench_fp8(dtype, m, k, n, label, sub_label)
    if dtype == torch.float16:
        return bench_fp16(dtype, m, k, n, label, sub_label)
    if dtype == torch.bfloat16:
        return bench_bf16(dtype, m, k, n, label, sub_label)
    raise ValueError("unsupported type")
