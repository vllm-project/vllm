# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Dimensions supported by the DSV3 specialized kernel
DSV3_SUPPORTED_NUM_EXPERTS = [256, 384]
DSV3_SUPPORTED_HIDDEN_SIZES = [7168]

# Dimensions supported by the gpt-oss specialized kernel
GPT_OSS_SUPPORTED_NUM_EXPERTS = [32, 128]
GPT_OSS_SUPPORTED_HIDDEN_SIZES = [2880]


def get_batch_size_range(max_batch_size):
    return [2**x for x in range(14) if 2**x <= max_batch_size]


def get_model_params(config):
    if config.architectures[0] in (
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
    ):
        num_experts = config.n_routed_experts
        hidden_size = config.hidden_size
    elif config.architectures[0] in ("GptOssForCausalLM",):
        num_experts = config.num_local_experts
        hidden_size = config.hidden_size
    else:
        raise ValueError(f"Unsupported architecture: {config.architectures}")
    return num_experts, hidden_size


def get_benchmark(model, max_batch_size, trust_remote_code):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=get_batch_size_range(max_batch_size),
            x_log=False,
            line_arg="provider",
            line_vals=[
                "torch",
                "vllm",
            ],
            line_names=["PyTorch", "vLLM"],
            styles=([("blue", "-"), ("red", "-")]),
            ylabel="TFLOPs",
            plot_name=f"{model} router gemm throughput",
            args={},
        )
    )
    def benchmark(batch_size, provider):
        config = get_config(model=model, trust_remote_code=trust_remote_code)
        num_experts, hidden_size = get_model_params(config)

        mat_a = torch.randn(
            (batch_size, hidden_size), dtype=torch.bfloat16, device="cuda"
        ).contiguous()
        mat_b = torch.randn(
            (num_experts, hidden_size), dtype=torch.bfloat16, device="cuda"
        ).contiguous()
        bias = torch.randn(
            num_experts, dtype=torch.bfloat16, device="cuda"
        ).contiguous()

        is_hopper_or_blackwell = current_platform.is_device_capability(
            90
        ) or current_platform.is_device_capability_family(100)
        allow_dsv3_router_gemm = (
            is_hopper_or_blackwell
            and num_experts in DSV3_SUPPORTED_NUM_EXPERTS
            and hidden_size in DSV3_SUPPORTED_HIDDEN_SIZES
        )
        allow_gpt_oss_router_gemm = (
            is_hopper_or_blackwell
            and num_experts in GPT_OSS_SUPPORTED_NUM_EXPERTS
            and hidden_size in GPT_OSS_SUPPORTED_HIDDEN_SIZES
        )

        has_bias = False
        if allow_gpt_oss_router_gemm:
            has_bias = True

        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":

            def runner():
                if has_bias:
                    F.linear(mat_a, mat_b, bias)
                else:
                    F.linear(mat_a, mat_b)
        elif provider == "vllm":

            def runner():
                if allow_dsv3_router_gemm:
                    ops.dsv3_router_gemm(mat_a, mat_b, torch.bfloat16)
                elif allow_gpt_oss_router_gemm:
                    ops.gpt_oss_router_gemm(mat_a, mat_b, bias)
                else:
                    raise ValueError("Unsupported router gemm")

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            runner, quantiles=quantiles
        )

        def tflops(t_ms):
            flops = 2 * batch_size * hidden_size * num_experts
            return flops / (t_ms * 1e-3) / 1e12

        return tflops(ms), tflops(max_ms), tflops(min_ms)

    return benchmark


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--max-batch-size", default=16, type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    # Get the benchmark function
    benchmark = get_benchmark(args.model, args.max_batch_size, args.trust_remote_code)
    # Run performance benchmark
    benchmark.run(print_data=True)
