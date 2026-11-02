# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused gated activation + NVFP4 quantization vs the two-kernel path.

For one activation and (M, N), input [M, 2N] -> NVFP4 [M, N] + block scales:
  cuda_act_quant     torch.ops._C.<act>_and_mul, then scaled_fp4_quant
  inductor_act_quant torch.compile(forward_native), then scaled_fp4_quant
                     (what vLLM runs by default: custom ops off, FP4 quant
                     is always a custom op)
  fused              torch.ops._C.<act>_and_mul_nvfp4_quant

    python benchmarks/kernels/benchmark_act_mul_nvfp4_quant.py \
        --activation gelu_tanh --n 21504 2112
    python benchmarks/kernels/benchmark_act_mul_nvfp4_quant.py --check
"""

import argparse

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton

if not current_platform.has_device_capability(100):
    raise RuntimeError("NVFP4 requires compute capability of 10.0 (Blackwell)")

# forward_native is compiled once per (M, N); lift the per-function limit.
torch._dynamo.config.recompile_limit = 8888

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# activation -> (layer factory, unfused CUDA op, fused CUDA op)
ACTIVATIONS = {
    "silu": (SiluAndMul, "silu_and_mul", "silu_and_mul_nvfp4_quant"),
    "gelu_tanh": (
        lambda: GeluAndMul(approximate="tanh"),
        "gelu_tanh_and_mul",
        "gelu_tanh_and_mul_nvfp4_quant",
    ),
}
PROVIDERS = ["cuda_act_quant", "inductor_act_quant", "fused"]
M_VALS = [1, 16, 128, 1024, 4096, 8192]


def global_scale(t: torch.Tensor) -> torch.Tensor:
    amax = torch.abs(t).max().to(torch.float32)
    return FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax


def make_case(activation: str, M: int, N: int, dtype=torch.bfloat16):
    """Return ({provider: fn}, global_scale, x) for one input x of [M, 2N]."""
    make_layer, act_name, fused_name = ACTIVATIONS[activation]
    layer = make_layer()
    x = torch.randn((M, 2 * N), device="cuda", dtype=dtype)
    gs = global_scale(layer.forward_native(x))
    act_op = getattr(torch.ops._C, act_name)
    fused_op = getattr(torch.ops._C, fused_name)
    act_buf = torch.empty((M, N), device="cuda", dtype=dtype)
    out, sf = ops.scaled_fp4_quant(act_buf, gs)
    native = torch.compile(layer.forward_native, dynamic=False)

    def cuda_act_quant():
        act_op(act_buf, x)
        return ops.scaled_fp4_quant(act_buf, gs)

    def inductor_act_quant():
        return ops.scaled_fp4_quant(native(x), gs)

    def fused():
        fused_op(out, sf, x, gs)
        return out, sf

    fns = {
        "cuda_act_quant": cuda_act_quant,
        "inductor_act_quant": inductor_act_quant,
        "fused": fused,
    }
    return fns, gs, x


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=M_VALS,
        x_log=False,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDERS,
        ylabel="us (lower is better)",
        plot_name="act_mul + NVFP4 quant latency (us)",
        args={},
    )
)
def benchmark(M, provider, activation, N):
    fns, _, _ = make_case(activation, M, N)
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fns[provider], quantiles=[0.5, 0.2, 0.8]
    )
    return ms * 1000, max_ms * 1000, min_ms * 1000


def check(activation: str) -> None:
    """Fraction of FP4 codes that differ from an fp32 single-rounding reference
    (the fused kernels round once, like the Inductor path; gelu_tanh must stay
    <= 0.1%, silu uses fast-math intrinsics and is only reported), plus the
    largest dequantized difference from cuda_act_quant as information: that
    path rounds twice, so a rare one-code FP4 step at large magnitude is
    expected and is not asserted on."""
    for M, N in [(1, 128), (7, 2112), (5, 21504), (256, 4096)]:
        fns, gs, x = make_case(activation, M, N)
        ref_q, ref_s = fns["cuda_act_quant"]()
        out_q, out_s = fns["fused"]()
        # The kernels fold gs into the block scales; dequantize with 1/gs.
        ref = dequantize_to_dtype(ref_q, ref_s, 1.0 / gs, torch.bfloat16)
        out = dequantize_to_dtype(out_q, out_s, 1.0 / gs, torch.bfloat16)
        max_abs = (out.float() - ref.float()).abs().max().item()
        gate, up = x[:, :N].float(), x[:, N:].float()
        act = (
            F.gelu(gate, approximate="tanh")
            if activation == "gelu_tanh"
            else F.silu(gate)
        )
        ref32_q, _ = ops.scaled_fp4_quant((act * up).to(torch.bfloat16), gs)
        mismatch = (out_q != ref32_q).float().mean().item()
        if activation == "gelu_tanh":
            assert mismatch <= 1e-3, f"{activation} M={M} N={N}: {mismatch:.2e}"
        print(
            f"{activation} M={M} N={N}: OK (packed FP4 bytes differing from the "
            f"fp32 single-rounding reference: {mismatch:.2e}; max |dequant - "
            f"cuda_act_quant| {max_abs:.3f})"
        )


@default_vllm_config()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fused activation + NVFP4 quantization benchmark"
    )
    parser.add_argument(
        "--activation",
        nargs="+",
        choices=list(ACTIVATIONS),
        default=["gelu_tanh", "silu"],
    )
    parser.add_argument("--n", nargs="+", type=int, default=[21504, 2112])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    if args.check:
        for activation in args.activation:
            check(activation)
        return

    for activation in args.activation:
        for N in args.n:
            print(f"\n{activation}, N={N} (input [M, {2 * N}] -> [M, {N}] NVFP4)")
            benchmark.run(
                print_data=True, save_path=args.save_path, activation=activation, N=N
            )


if __name__ == "__main__":
    main()
