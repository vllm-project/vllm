# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# fmt: off
# ruff: noqa: E501
import dataclasses
import time
from typing import Optional

# Import DeepGEMM functions
import torch

# Import vLLM functions
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.cutlass_moe import (
    block_scaled_cutlass_moe_fp8,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk,
    modular_triton_fused_moe,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import scaled_dequantize
from vllm.platforms import current_platform
from vllm.triton_utils import triton


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y



vllm_config = VllmConfig(parallel_config=ParallelConfig(
    pipeline_parallel_size=1))
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192

def torch_experts(a: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  topk_weight: torch.Tensor,
                  topk_ids: torch.Tensor,
                  global_num_experts: int = -1,
                  expert_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert (global_num_experts == -1
            or (global_num_experts == w1.shape[0] and expert_map is None)
            or (expert_map is not None
                and global_num_experts == expert_map.shape[0]))
    topk = topk_ids.shape[1]
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


def torch_moe(a: torch.Tensor,
              w1: torch.Tensor,
              w2: torch.Tensor,
              score: torch.Tensor,
              topk: int,
              global_num_experts: int = -1,
              expert_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    return torch_experts(a, w1, w2, topk_weight, topk_ids, global_num_experts,
                         expert_map)


# Copied from
# https://github.com/deepseek-ai/DeepGEMM/blob/78cacf70d41d15d688bd493ebc85845f7f2a3d5d/tests/test_core.py#L17
def per_block_cast_to_fp8(
        x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert tensor to FP8 format with per-block scaling."""
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (
        x_amax / 448.0).view(x_view.size(0), x_view.size(2))

@dataclasses.dataclass
class MOETensors:
    a: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    ab_strides1: torch.Tensor
    c_strides1: torch.Tensor
    ab_strides2: torch.Tensor
    c_strides2: torch.Tensor

    @staticmethod
    def make_moe_tensors(m: int, k: int, n: int, e: int,
                         dtype: torch.dtype) -> "MOETensors":
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        return MOETensors(a=a,
                          w1=w1,
                          w2=w2,
                          ab_strides1=ab_strides1,
                          c_strides1=c_strides1,
                          ab_strides2=ab_strides2,
                          c_strides2=c_strides2)

@dataclasses.dataclass
class MOETensors8Bit(MOETensors):
    # quantized
    a_q: Optional[torch.Tensor] = None  # a -> a_q
    w1_q: Optional[torch.Tensor] = None  # w1 -> w1_q
    w2_q: Optional[torch.Tensor] = None  # w2 -> w2_q
    a_scale: Optional[torch.Tensor] = None
    w1_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    # dequantized
    a_d: Optional[torch.Tensor] = None  # a -> a_q -> a_d
    w1_d: Optional[torch.Tensor] = None  # w1 -> w1_q -> w1_d
    w2_d: Optional[torch.Tensor] = None  # w2 -> w2_q -> w2_d

    @staticmethod
    def make_moe_tensors_blocked_8bit(m: int, k: int, n: int, e: int,
                                      per_act_block: bool,
                                      block_size: tuple[int, int],
                                      dtype: torch.dtype) -> "MOETensors8Bit":
        q_dtype = torch.float8_e4m3fn

        moe_tensors_fp16 = MOETensors.make_moe_tensors(m, k, n, e, dtype)

        n_b1_scales = 2 * n // block_size[0]
        k_b1_scales = k // block_size[1]
        n_b2_scales = k // block_size[0]
        k_b2_scales = n // block_size[1]

        # Get the right scale for tests.
        if per_act_block:
            a_q, a_scale = per_token_group_quant_fp8(moe_tensors_fp16.a,
                                                     block_size[1])
        else:
            _, a_scale = ops.scaled_fp8_quant(moe_tensors_fp16.a,
                                              use_per_token_if_dynamic=False)
            a_q, _ = ops.scaled_fp8_quant(moe_tensors_fp16.a,
                                          a_scale,
                                          use_per_token_if_dynamic=False)

        w1_q = torch.empty(moe_tensors_fp16.w1.shape,
                           device="cuda",
                           dtype=q_dtype)
        w2_q = torch.empty(moe_tensors_fp16.w2.shape,
                           device="cuda",
                           dtype=q_dtype)
        w1_scale = torch.randn((e, n_b1_scales, k_b1_scales),
                               device="cuda",
                               dtype=torch.float32)
        w2_scale = torch.randn((e, n_b2_scales, k_b2_scales),
                               device="cuda",
                               dtype=torch.float32)

        for expert in range(e):
            w1_q[expert], w1_scale[expert] = per_block_cast_to_fp8(
                moe_tensors_fp16.w1[expert])
            w2_q[expert], w2_scale[expert] = per_block_cast_to_fp8(
                moe_tensors_fp16.w2[expert])

        def block_dequant_w(w, w_q, scale, block_size):
            for expert in range(w.size(0)):
                w[expert] = scaled_dequantize(w_q[expert], scale[expert],
                                              block_size)

        if per_act_block:
            a_d = scaled_dequantize(a_q, a_scale, [1, block_size[1]], dtype)
        else:
            a_d = a_q.float().mul(a_scale).to(dtype)
        w1_d = torch.empty_like(moe_tensors_fp16.w1)
        w2_d = torch.empty_like(moe_tensors_fp16.w2)
        block_dequant_w(w1_d, w1_q, w1_scale, block_size)
        block_dequant_w(w2_d, w2_q, w2_scale, block_size)

        return MOETensors8Bit(a=moe_tensors_fp16.a,
                              w1=w1_d,
                              w2=w2_d,
                              ab_strides1=moe_tensors_fp16.ab_strides1,
                              c_strides1=moe_tensors_fp16.c_strides1,
                              ab_strides2=moe_tensors_fp16.ab_strides2,
                              c_strides2=moe_tensors_fp16.c_strides2,
                              a_q=a_q,
                              w1_q=w1_q,
                              w2_q=w2_q,
                              a_scale=a_scale,
                              w1_scale=w1_scale,
                              w2_scale=w2_scale,
                              a_d=a_d,
                              w1_d=w1_d,
                              w2_d=w2_d)


def benchmark_shape(e: int,
                    m: int,
                    n: int,
                    k: int,
                    topk: int,
                    warmup: int = 100,
                    repeat: int = 10000,
                    verbose: bool = False) -> dict:
    """Benchmark all implementations for a specific (m, n, k) shape."""
    if verbose:
        print(f"\n=== Benchmarking shape: m={m}, n={n}, k={k} ===")

    # Create test tensors
    mt = MOETensors8Bit.make_moe_tensors_blocked_8bit(m, k, n, e,
                                                      per_act_block=True,
                                                      block_size=(128, 128),
                                                      dtype=torch.bfloat16)

    a = mt.a
    a1_scale = mt.a_scale
    a_d = mt.a_d
    w1 = mt.w1_q
    w2 = mt.w2_q
    w1_d = mt.w1_d
    w2_d = mt.w2_d
    w1_scale = mt.w1_scale
    w2_scale = mt.w2_scale
    w1_scale_t = mt.w1_scale.transpose(1, 2).contiguous()
    w2_scale_t = mt.w2_scale.transpose(1, 2).contiguous()

    score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

    C_ref = torch_moe(a_d, w1_d, w2_d, score, topk)

    # === vLLM Triton Implementation ===
    def vllm_triton_gemm():
        m_fused_moe = modular_triton_fused_moe(use_fp8_w8a8=True,
                                           use_int8_w8a8=False,
                                           use_int8_w8a16=False,
                                           use_int4_w4a16=False,
                                           use_mxfp4_w4a4=False,
                                           per_act_token_quant=False,
                                           block_shape=[128, 128])
        return m_fused_moe(
            a,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
        )

    # === vLLM CUTLASS Implementation ===
    def vllm_cutlass_gemm():
        return block_scaled_cutlass_moe_fp8(
                    a,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    w1_scale_t,
                    w2_scale_t,
                    [128, 128],
                    ab_strides1=mt.ab_strides1,
                    ab_strides2=mt.ab_strides2,
                    c_strides1=mt.c_strides1,
                    c_strides2=mt.c_strides2,
                    a1_scale=a1_scale,
                    per_act_block=True,
                    global_num_experts=w1.size(0),
                )

    # Run correctness check first
    if verbose:
        print("Running correctness check...")
    C_vllm_triton = vllm_triton_gemm()
    C_vllm_cutlass = vllm_cutlass_gemm()

    vllm_triton_diff = calc_diff(C_vllm_triton, C_ref)
    vllm_cutlass_diff = calc_diff(C_vllm_cutlass, C_ref)

    if verbose:
        print(f"vLLM Triton vs Reference difference: {vllm_triton_diff:.6f}")
        print(f"vLLM CUTLASS vs Reference difference: {vllm_cutlass_diff:.6f}")

    # Benchmark implementations
    implementations = {
        "vLLM Triton": vllm_triton_gemm,
        "vLLM CUTLASS": vllm_cutlass_gemm
    }

    benchmark_results = {
        "shape": {
            "m": m,
            "n": n,
            "k": k
        },
        "num_experts": e,
        "topk": topk,
        "implementations": {}
    }

    for name, func in implementations.items():
        # Warmup
        for _ in range(warmup):
            func()
            torch.cuda.synchronize()

        # Timing loop
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeat):
            func()
        torch.cuda.synchronize()
        end = time.time()

        # Calculate timing and TFLOPS
        avg_time_ms = (end - start) / repeat * 1000
        avg_time_us = avg_time_ms * 1000
        tflops = 2 * m * n * k / (avg_time_ms * 1e-3) / 1e12
        gb_s = (m * k + k * n + m * n * 2) / 1e9 / (avg_time_ms * 1e-3)

        benchmark_results["implementations"][name] = {
            "time_ms": avg_time_ms,
            "time_us": avg_time_us,
            "tflops": tflops,
            "gb_s": gb_s,
            "diff": {
                "vLLM Triton":
                0.0 if name == "vLLM Triton" else calc_diff(func(), C_vllm_triton),
                "Reference":
                vllm_triton_diff if name == "vLLM Triton" else vllm_cutlass_diff,
            }
        }

        if verbose:
            print(
                f"{name}: {avg_time_ms:.3f} ms, {tflops:.2f} TFLOPS, {gb_s:.2f} GB/s"
            )

    # Calculate speedups
    baseline = benchmark_results["implementations"]["vLLM Triton"][
        "time_ms"]
    for name, data in benchmark_results["implementations"].items():
        if name != "vLLM Triton":
            speedup = baseline / data["time_ms"]
            benchmark_results["implementations"][name][
                "speedup_vs_triton"] = speedup
            if verbose:
                print(f"Triton is {1/speedup:.2f}x "
                      f"{'faster' if 1/speedup > 1 else 'slower'} than {name}")

    vllm_triton_time = benchmark_results["implementations"]["vLLM Triton"][
        "time_ms"]
    vllm_cutlass_time = benchmark_results["implementations"]["vLLM CUTLASS"][
        "time_ms"]
    cutlass_vs_triton = vllm_triton_time / vllm_cutlass_time
    benchmark_results["implementations"]["vLLM CUTLASS"][
        "speedup_vs_triton"] = cutlass_vs_triton
    if verbose:
        print(
            f"vLLM CUTLASS is {cutlass_vs_triton:.2f}x "
            f"{'faster' if cutlass_vs_triton > 1 else 'slower'} than vLLM Triton"
        )

    return benchmark_results


def format_table_row(values, widths):
    """Format a row with specified column widths."""
    return "| " + " | ".join(f"{val:{w}}"
                             for val, w in zip(values, widths)) + " |"


def print_table(headers, rows, title=None):
    """Print a table with headers and rows."""
    if title:
        print(f"\n{title}")

    # Calculate column widths based on headers and data
    widths = [
        max(len(str(h)), max(len(str(row[i])) for row in rows))
        for i, h in enumerate(headers)
    ]

    # Create separator line
    separator = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    # Print table
    print(separator)
    print(format_table_row(headers, widths))
    print(separator)
    for row in rows:
        print(format_table_row(row, widths))
    print(separator)


def format_speedup(value):
    """Format speedup value with indicator if it's faster or slower."""
    return f"{value:.2f}x {'faster' if value > 1.0 else 'slower'}"


def run_benchmarks(verbose: bool = False):
    """Run benchmarks for a set of common shapes."""
    print("===== STARTING FP8 GEMM BENCHMARK =====")

    # Make sure we're using the GPU
    if not torch.cuda.is_available():
        print("CUDA not available! Tests require GPU.")
        return

    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Triton version: {triton.__version__}")
    print(f"Using device: {torch.cuda.get_device_name()}")

    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Define benchmark shapes (m, n, k)
    shapes = [
        (64, 24576, 1536),
        (64, 32768, 512),
        (64, 7168, 16384),
        (64, 4096, 7168),
        (64, 7168, 2048),
        (128, 24576, 1536),
        (128, 32768, 512),
        (128, 7168, 16384),
        (128, 4096, 7168),
        (128, 7168, 2048),
        (4096, 24576, 1536),
        (4096, 32768, 512),
        (4096, 7168, 16384),
        (4096, 4096, 7168),
        (4096, 7168, 2048),
    ]
    num_experts = [8, 40]
    topks = [1, 6, 8]

    all_results = []
    for e in num_experts:
        for m, n, k in shapes:
            for topk in topks:
                print(f"Benchmarking shape: m={m}, n={n}, k={k}, e={e}, topk={topk}")
                result = benchmark_shape(e, m, n, k, topk, verbose=verbose)
                all_results.append(result)

    # Print results in a nicely formatted table
    print("\n===== PERFORMANCE COMPARISON =====")

    # Print vLLM Triton table
    triton_headers = [
        "m", "n", "k", "e", "topk", "Time (μs)", "TFLOPS", "GB/s"]
    triton_rows = []
    for result in all_results:
        shape = result["shape"]
        num_experts = result["num_experts"]
        topk = result["topk"]
        impl_data = result["implementations"]["vLLM Triton"]
        triton_rows.append([
            shape["m"], shape["n"], shape["k"], num_experts, topk,
            f"{impl_data['time_us']:.1f}",
            f"{impl_data['tflops']:.1f}", f"{impl_data['gb_s']:.1f}"
        ])

    print_table(triton_headers,
                triton_rows,
                title="vLLM Triton Implementation:")

    # Print vLLM CUTLASS table
    cutlass_headers = [
        "m", "n", "k", "e", "topk", "Time (μs)", "TFLOPS", "GB/s", "vs Triton"
    ]
    cutlass_rows = []
    for result in all_results:
        shape = result["shape"]
        num_experts = result["num_experts"]
        topk = result["topk"]
        impl_data = result["implementations"]["vLLM CUTLASS"]
        vs_triton = impl_data.get("speedup_vs_triton", 1.0)
        cutlass_rows.append([
            shape["m"], shape["n"], shape["k"], num_experts, topk,
            f"{impl_data['time_us']:.1f}",
            f"{impl_data['tflops']:.1f}", f"{impl_data['gb_s']:.1f}",
            format_speedup(vs_triton)
        ])

    print_table(cutlass_headers,
                cutlass_rows,
                title="vLLM CUTLASS Implementation:")

    # Calculate and print averages
    print("\n===== AVERAGE PERFORMANCE =====")

    implementations = ["vLLM Triton", "vLLM CUTLASS"]
    avg_metrics = {
        impl: {
            "tflops": 0,
            "gb_s": 0,
            "time_ms": 0
        }
        for impl in implementations
    }

    for result in all_results:
        for impl in implementations:
            impl_data = result["implementations"][impl]
            avg_metrics[impl]["tflops"] += impl_data["tflops"]
            avg_metrics[impl]["gb_s"] += impl_data["gb_s"]
            avg_metrics[impl]["time_ms"] += impl_data["time_ms"]

    num_shapes = len(all_results)
    avg_headers = ["Implementation", "Avg TFLOPS", "Avg GB/s", "Avg Time (ms)"]
    avg_rows = []

    for impl in implementations:
        avg_tflops = avg_metrics[impl]["tflops"] / num_shapes
        avg_mem_bw = avg_metrics[impl]["gb_s"] / num_shapes
        avg_time = avg_metrics[impl]["time_ms"] / num_shapes
        avg_rows.append([
            impl, f"{avg_tflops:.2f}", f"{avg_mem_bw:.2f}", f"{avg_time:.2f}"
        ])

    print_table(avg_headers, avg_rows)

    # Calculate average speedups
    avg_speedups = {
        "vLLM CUTLASS vs vLLM Triton": 0
    }

    for result in all_results:
        vllm_triton_time = result["implementations"]["vLLM Triton"]["time_ms"]
        vllm_cutlass_time = result["implementations"]["vLLM CUTLASS"][
            "time_ms"]

        avg_speedups[
            "vLLM CUTLASS vs vLLM Triton"] += vllm_triton_time / vllm_cutlass_time

    print("\n===== AVERAGE SPEEDUPS =====")
    speedup_headers = ["Comparison", "Speedup"]
    speedup_rows = []
    for comparison, total in avg_speedups.items():
        avg_speedup = total / num_shapes
        status = "faster" if avg_speedup > 1 else "slower"
        speedup_rows.append([comparison, f"{avg_speedup:.2f}x {status}"])

    print_table(speedup_headers, speedup_rows)

    # Average accuracy comparison
    print("\n===== ACCURACY COMPARISON =====")
    avg_diff = {impl: 0 for impl in implementations}

    for result in all_results:
        for impl in implementations:
            avg_diff[impl] += result["implementations"][impl]["diff"][
                "Reference"]

    diff_headers = ["Implementation", "Avg Diff vs Reference"]
    diff_rows = []
    for impl in implementations:
        diff_rows.append([impl, f"{avg_diff[impl] / num_shapes:.6f}"])

    print_table(diff_headers, diff_rows)


if __name__ == "__main__":
    assert current_platform.is_device_capability(90), "This benchmark requires SM90" # noqa: E501
    with set_current_vllm_config(vllm_config):
        run_benchmarks(verbose=False)
