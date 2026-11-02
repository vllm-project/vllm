# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import time

import numpy as np
import torch

from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

# Check if CPU MoE operations are available
try:
    from vllm._custom_ops import (
        CPUActMethod,
        CPUQuantMethod,
        cpu_fused_moe,
        cpu_native_fp8_supported,
        cpu_prepack_moe_weight,
        fused_experts_cpu,
    )
except (ImportError, AttributeError) as e:
    print(f"Import error: {e}")
    sys.exit(1)

from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
    prepare_mxfp4_moe_layer_for_cpu,
)

# ISA selection following test_cpu_fused_moe.py pattern
ISA_CHOICES = ["vec"]
if torch.cpu._is_amx_tile_supported():
    ISA_CHOICES.append("amx")
if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
    ISA_CHOICES.append("neon")


def _quantize_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack BF16 weights into MXFP4 E2M1 values and E8M0 block scales."""
    block_size = 32
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    grouped = weight.float().reshape(-1, block_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    exponent = torch.ceil(
        torch.maximum(torch.log2(amax / values[-1]), torch.tensor(-127.0))
    )
    normalized = grouped / torch.exp2(exponent)
    magnitude = (normalized.abs().unsqueeze(-1) > bounds).sum(dim=-1)
    sign = ((2 - torch.sign(normalized)) // 2).to(torch.uint8)
    codes = ((sign << 3) | magnitude.to(torch.uint8)).reshape(
        *weight.shape[:-1], weight.shape[-1]
    )
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    scales = (exponent.squeeze(-1) + 127).to(torch.uint8)
    return packed.contiguous(), scales.reshape(*weight.shape[:-1], -1).contiguous()


@torch.inference_mode()
def main(
    batch_size: int,
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    topk_num: int,
    use_bias: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    activation: str = "silu",
    isa: str = "vec",
    seed: int = 0,
    iters: int = 20,
    compare_native_fp8: bool = False,
    tokens_per_expert: int | None = None,
) -> None:
    if tokens_per_expert is not None:
        if not compare_native_fp8:
            raise ValueError("--tokens-per-expert requires comparison mode")
        if tokens_per_expert <= 0:
            raise ValueError("--tokens-per-expert must be positive")
        batch_size = expert_num * tokens_per_expert
        topk_num = 1

    if compare_native_fp8:
        if iters <= 0:
            raise ValueError("--iters must be positive in comparison mode")
        if not cpu_native_fp8_supported(torch.float8_e4m3fn):
            raise RuntimeError("Native CPU E4M3 support is unavailable")
        if not hasattr(torch.ops._C, "quantize_fp8e4m3_vec"):
            raise RuntimeError("quantize_fp8e4m3_vec is unavailable in this build")
        if hidden_size % 32 or intermediate_size % 32:
            raise ValueError("MXFP4 benchmark dimensions must be multiples of 32")

    set_random_seed(seed)
    # up_dim = 2 * intermediate_size for gate + up projection
    up_dim = 2 * intermediate_size

    input_tensor = torch.randn((batch_size, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )

    w13 = torch.randn((expert_num, up_dim, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w2 = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype) / (
        0.5 * intermediate_size**0.5
    )

    w13_bias = None
    w2_bias = None
    if use_bias:
        bias_dtype = torch.float32 if compare_native_fp8 else dtype
        w13_bias = torch.randn((expert_num, up_dim), dtype=bias_dtype) / (
            0.5 * up_dim**0.5
        )
        w2_bias = torch.randn((expert_num, hidden_size), dtype=bias_dtype) / (
            0.5 * hidden_size**0.5
        )

    if tokens_per_expert is None:
        router_logits = torch.randn((batch_size, expert_num), dtype=dtype)
        score = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(score, topk_num)
        topk_ids = topk_ids.to(torch.int32)
    else:
        topk_ids = (torch.arange(batch_size, dtype=torch.int32) % expert_num).reshape(
            batch_size, 1
        )
        topk_weights = torch.ones((batch_size, 1), dtype=torch.float32)

    # FLOPs estimation: 2 * batch * topk * (hidden * up_dim + intermediate * hidden)
    flops_per_token = (
        2 * topk_num * (hidden_size * up_dim + intermediate_size * hidden_size)
    )
    total_flops = batch_size * flops_per_token

    def run_benchmark(run_moe, iters: int) -> list[float]:
        times = []
        for _ in range(iters):
            start_time = time.perf_counter_ns()
            run_moe()
            end_time = time.perf_counter_ns()
            times.append((end_time - start_time) / 1e6)
        return times

    def print_timing(times: list[float], label: str = "") -> None:
        print(f"{label}\tmin (ms) = ", min(times))
        print(f"{label}\tmax (ms) = ", max(times))
        print(f"{label}\tmean (ms) = ", np.mean(times))
        print(f"{label}\tstd = ", np.std(times))
        print(f"{label}\tmedian (ms) = ", np.median(times))

    if not compare_native_fp8:
        packed_w13 = cpu_prepack_moe_weight(w13, isa)
        packed_w2 = cpu_prepack_moe_weight(w2, isa)
        output = torch.empty_like(input_tensor)

        def run_unquantized() -> None:
            cpu_fused_moe(
                output,
                input_tensor,
                packed_w13,
                packed_w2,
                w13_bias,
                w2_bias,
                topk_weights,
                topk_ids,
                activation,
                isa,
            )

        # warmup
        run_benchmark(run_unquantized, 5)
        # benchmark
        times = run_benchmark(run_unquantized, iters)
        if not times:
            print("No iterations to measure. Set --iters > 0.")
            return
        print_timing(times)

        tflops = total_flops / (np.mean(times) * 1e-3) / 1e12
        print(f"\tthroughput (TFLOP/s) = {tflops:.4f}")
        return

    if activation == "swigluoai":
        alpha, limit = 1.702, 7.0
        act_method = CPUActMethod.SWIGLU
    else:
        alpha, limit = None, None
        act_method = CPUActMethod.SILU_AND_MUL

    w13_quant, w13_scale = _quantize_mxfp4(w13)
    w2_quant, w2_scale = _quantize_mxfp4(w2)
    packed_w13, packed_w2, packed_w13_scale, packed_w2_scale = (
        prepare_mxfp4_moe_layer_for_cpu(w13_quant, w2_quant, w13_scale, w2_scale)
    )
    w13_metadata = torch.ops._C.mxfp4_moe_scale_metadata(packed_w13_scale)
    w2_metadata = torch.ops._C.mxfp4_moe_scale_metadata(packed_w2_scale)
    w13_channels = torch.ops._C.mxfp4_moe_channel_metadata(packed_w13_scale)
    w2_channels = torch.ops._C.mxfp4_moe_channel_metadata(packed_w2_scale)

    output_native = torch.empty_like(input_tensor)
    output_baseline = torch.empty_like(input_tensor)
    quantize_activation = torch.ops._C.quantize_fp8e4m3_vec

    def run_native() -> None:
        activation_input, activation_scale = quantize_activation(
            input_tensor, True, None
        )
        fused_experts_cpu(
            output_native,
            activation_input,
            packed_w13,
            packed_w2,
            topk_weights,
            topk_ids,
            CPUQuantMethod.MXFP4_W4A8,
            packed_w13_scale,
            packed_w2_scale,
            None,
            None,
            None,
            w1_bias=w13_bias,
            w2_bias=w2_bias,
            alpha=alpha,
            limit=limit,
            act_method=act_method,
            a1_scale=activation_scale,
            w1_block_metadata=w13_metadata,
            w2_block_metadata=w2_metadata,
            w1_channel_metadata=w13_channels,
            w2_channel_metadata=w2_channels,
        )

    def run_baseline() -> None:
        fused_experts_cpu(
            output_baseline,
            input_tensor,
            packed_w13,
            packed_w2,
            topk_weights,
            topk_ids,
            CPUQuantMethod.MXFP4,
            packed_w13_scale,
            packed_w2_scale,
            None,
            None,
            None,
            w1_bias=w13_bias,
            w2_bias=w2_bias,
            alpha=alpha,
            limit=limit,
            act_method=act_method,
        )

    run_native()
    run_baseline()
    if not torch.isfinite(output_native).all().item():
        raise RuntimeError("Native E4M3 MoE produced non-finite output")
    if not torch.isfinite(output_baseline).all().item():
        raise RuntimeError("Baseline MoE produced non-finite output")
    occupancy = (
        f"exact {tokens_per_expert}" if tokens_per_expert is not None else "random"
    )
    print(
        "Comparing CPU MXFP4 W4A8 with W4A16; "
        "weight quantization and prepacking are excluded."
    )
    print(
        f"M={batch_size}, topk={topk_num}, experts={expert_num}, "
        f"expert occupancy={occupancy}"
    )
    print("Output check: finite outputs")

    run_benchmark(run_native, 5)
    native_times = run_benchmark(run_native, iters)
    run_benchmark(run_baseline, 5)
    baseline_times = run_benchmark(run_baseline, iters)
    print_timing(native_times, "native E4M3")
    print_timing(baseline_times, "BF16 activation")
    print(f"\tspeedup = {np.median(baseline_times) / np.median(native_times):.3f}x")

    native_tflops = total_flops / (np.mean(native_times) * 1e-3) / 1e12
    baseline_tflops = total_flops / (np.mean(baseline_times) * 1e-3) / 1e12
    print(f"\tnative throughput (TFLOP/s) = {native_tflops:.4f}")
    print(f"\tbaseline throughput (TFLOP/s) = {baseline_tflops:.4f}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the CPU fused MoE kernel.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expert-num", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2880)
    parser.add_argument("--intermediate-size", type=int, default=2880)
    parser.add_argument(
        "--topk-num",
        type=int,
        default=None,
        help="Number of experts to route each token to (default: expert_num // 2)",
    )
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument(
        "--activation",
        type=str,
        choices=["silu", "swigluoai"],
        default="silu",
        help="Activation function",
    )
    parser.add_argument(
        "--isa",
        type=str,
        choices=ISA_CHOICES,
        default="vec",
        help=f"ISA to use (available: {ISA_CHOICES})",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--compare-native-fp8",
        action="store_true",
        help="Compare MXFP4 W4A8 with the W4A16 backend.",
    )
    parser.add_argument(
        "--tokens-per-expert",
        type=int,
        default=None,
        help=(
            "Use top-1 routes with this many tokens per expert; "
            "batch size is derived from expert_num."
        ),
    )

    args = parser.parse_args()

    # Default topk_num to expert_num // 2, minimum 1
    topk_num = (
        args.topk_num if args.topk_num is not None else max(args.expert_num // 2, 1)
    )

    print(args)

    main(
        batch_size=args.batch_size,
        expert_num=args.expert_num,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        topk_num=topk_num,
        use_bias=args.use_bias,
        dtype=torch.bfloat16,  # Following test_cpu_fused_moe.py
        activation=args.activation,
        isa=args.isa,
        seed=args.seed,
        iters=args.iters,
        compare_native_fp8=args.compare_native_fp8,
        tokens_per_expert=args.tokens_per_expert,
    )
