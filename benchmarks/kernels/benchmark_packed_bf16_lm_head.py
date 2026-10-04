# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark losslessly packed BF16 lm-head projection against PyTorch."""

import argparse

import torch

from vllm.model_executor.kernels.linear.unquantized.packed_bf16_lm_head import (
    _packed_bf16_lm_head_impl,
    choose_launch_config,
    pack_bf16_weight,
)
from vllm.triton_utils import triton


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    weight = torch.randn(
        args.vocab_size,
        args.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = torch.randn(1, args.hidden_size, device="cuda", dtype=torch.bfloat16)
    packed = pack_bf16_weight(weight, max_packed_fraction=1.0)
    if packed is None:
        raise RuntimeError("The generated matrix is not eligible for packing")
    plan, tensors = packed
    launch = choose_launch_config(plan.k)

    def torch_projection():
        return torch.nn.functional.linear(x, weight)

    def packed_projection():
        return _packed_bf16_lm_head_impl(
            x,
            *tensors,
            plan.n,
            plan.k,
            launch.block_n,
            launch.num_warps,
        )

    expected = torch_projection()
    actual = packed_projection()
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.02)

    torch_ms = triton.testing.do_bench(torch_projection)
    packed_ms = triton.testing.do_bench(packed_projection)
    print(f"shape: (1, {plan.k}) x ({plan.n}, {plan.k})^T")
    print(f"packed storage: {plan.packed_fraction:.3f} of dense")
    print(f"torch: {torch_ms:.3f} ms")
    print(f"packed: {packed_ms:.3f} ms")
    print(f"speedup: {torch_ms / packed_ms:.3f}x")


if __name__ == "__main__":
    main()
