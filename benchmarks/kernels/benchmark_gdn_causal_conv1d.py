# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark optimized GDN causal-conv paths against the current operator."""

import argparse
from collections.abc import Callable

import torch

from tests.kernels.mamba.causal_conv1d_contract import ContractCase, build_case
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn as official,
)
from vllm.model_executor.layers.mamba.ops.gdn_causal_conv1d import (
    causal_conv1d_fn as optimized,
)
from vllm.triton_utils import triton

CASES = (
    ContractCase(
        "single_none", (4096,), dim=4096, activation=None, initial_mask=(True,)
    ),
    ContractCase(
        "varlen",
        (1000, 1200, 896, 1000),
        dim=4096,
        initial_mask=(True, False, True, False),
    ),
    ContractCase(
        "fp32",
        (1024,),
        dim=2048,
        dtype=torch.float32,
        initial_mask=(True,),
    ),
    ContractCase("width3", (4096,), dim=4096, width=3, initial_mask=(True,)),
)


def _bench_pair(
    name: str,
    official_call: Callable[[], torch.Tensor],
    optimized_call: Callable[[], torch.Tensor],
) -> None:
    official_call()
    optimized_call()
    torch.accelerator.synchronize()
    official_us = (
        triton.testing.do_bench(official_call, warmup=20, rep=50, return_mode="median")
        * 1000
    )
    optimized_us = (
        triton.testing.do_bench(optimized_call, warmup=20, rep=50, return_mode="median")
        * 1000
    )
    print(
        f"{name:16s} official={official_us:8.2f} us "
        f"optimized={optimized_us:8.2f} us "
        f"speedup={official_us / optimized_us:5.2f}x"
    )


def _call_contract(fn, inputs):
    return fn(
        inputs["x"],
        inputs["weight"],
        inputs["bias"],
        inputs["conv_states"],
        inputs["query_start_loc"],
        cache_indices=inputs["cache_indices"],
        has_initial_state=inputs["has_initial_state"],
        activation=inputs["activation"],
    )


def benchmark_contract_cases() -> None:
    for case in CASES:
        inputs = build_case(case, torch.device("cuda"))
        _bench_pair(
            case.name,
            lambda inputs=inputs: _call_contract(official, inputs),
            lambda inputs=inputs: _call_contract(optimized, inputs),
        )


def benchmark_production_shape() -> None:
    torch.manual_seed(0)
    dim, tokens = 12288, 16384
    projected = torch.randn(tokens, dim + 8192, device="cuda", dtype=torch.bfloat16)
    x = projected[:, :dim].T
    weight = torch.randn(dim, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    states = torch.randn(2, dim, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    query_start_loc = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    cache_indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([False], device="cuda")

    def call(fn):
        return fn(
            x,
            weight,
            None,
            states,
            query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
        )

    reference = call(official)
    actual = call(optimized)
    relative_l2 = torch.linalg.vector_norm(
        (actual.float() - reference.float()).double()
    ) / torch.linalg.vector_norm(reference.float().double())
    print(f"production relative_l2={relative_l2.item():.6g}")
    _bench_pair(
        "sm103_production",
        lambda: call(official),
        lambda: call(optimized),
    )


def benchmark_apc() -> None:
    torch.manual_seed(0)
    dim, tokens, width = 4096, 4096, 4
    projected = torch.randn(tokens, dim + 64, device="cuda", dtype=torch.bfloat16)
    x = projected[:, :dim].T
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    states = torch.randn(
        8, width - 1, dim, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    query_start_loc = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    cache_indices = torch.tensor([[1, 2, 3, 4]], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True], device="cuda")
    first = torch.tensor([1], device="cuda", dtype=torch.int32)
    last = torch.tensor([3], device="cuda", dtype=torch.int32)
    initial = torch.tensor([0], device="cuda", dtype=torch.int32)
    computed = torch.tensor([0], device="cuda", dtype=torch.int32)

    def call(fn):
        return fn(
            x,
            weight,
            bias,
            states,
            query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            block_idx_first_scheduled_token=first,
            block_idx_last_scheduled_token=last,
            initial_state_idx=initial,
            num_computed_tokens=computed,
            block_size_to_align=128,
        )

    _bench_pair("apc", lambda: call(official), lambda: call(optimized))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-shape", action="store_true")
    parser.add_argument("--apc", action="store_true")
    args = parser.parse_args()

    benchmark_contract_cases()
    if args.production_shape:
        benchmark_production_shape()
    if args.apc:
        benchmark_apc()


if __name__ == "__main__":
    main()
