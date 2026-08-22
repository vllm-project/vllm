# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark strided GDN RMSNormGated implementations."""

import argparse

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.ops.gdn_rmsnorm import (
    fused_gdn_rmsnorm_gated,
)
from vllm.triton_utils import triton


def _bench(function) -> list[float]:
    return [
        value * 1000
        for value in triton.testing.do_bench(
            function,
            warmup=100,
            rep=300,
            quantiles=[0.2, 0.5, 0.8],
        )
    ]


def benchmark_case(tokens: int, heads: int, hidden_size: int) -> None:
    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):
        layer = RMSNormGated(
            hidden_size,
            eps=1e-6,
            norm_before_gate=True,
            activation="silu",
            device="cuda",
            dtype=torch.bfloat16,
        )
    x = torch.randn(tokens, heads, hidden_size, device="cuda", dtype=torch.bfloat16)
    projected = torch.randn(
        tokens,
        heads + 96,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    z = projected[:, 96:, :]
    compiled_native = torch.compile(layer.forward_native, fullgraph=True)

    reference = compiled_native(x, z)
    optimized = fused_gdn_rmsnorm_gated(x, z, layer.weight, layer.eps)
    difference = (optimized.float() - reference.float()).double()
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        reference.float().double()
    )

    native_us = _bench(lambda: compiled_native(x, z))
    cuda_us = _bench(lambda: layer.forward_cuda(x, z))
    optimized_us = _bench(
        lambda: fused_gdn_rmsnorm_gated(x, z, layer.weight, layer.eps)
    )
    print(
        f"D={hidden_size:3d} T={tokens:5d} H={heads:2d} "
        f"native={native_us[1]:8.2f} us "
        f"cuda={cuda_us[1]:8.2f} us "
        f"optimized={optimized_us[1]:8.2f} us "
        f"speedup={native_us[1] / optimized_us[1]:5.2f}x "
        f"rel_l2={relative_l2.item():.3g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[64, 96, 128, 192, 256]
    )
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--production-shape", action="store_true")
    args = parser.parse_args()

    for hidden_size in args.hidden_sizes:
        benchmark_case(args.tokens, args.heads, hidden_size)
    if args.production_shape:
        benchmark_case(8192, 64, 128)


if __name__ == "__main__":
    main()
