# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from benchmarks.kernels.benchmark_moe import prune_cuda_wna16_search_space


def test_prune_cuda_wna16_search_space_filters_cuda_constraints(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.kernels.benchmark_moe.should_moe_wna16_use_cuda",
        lambda **kwargs: True,
    )
    search_space = [
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 128},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 128},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 384},
    ]

    assert prune_cuda_wna16_search_space(
        search_space,
        num_valid_tokens=8,
        num_experts=8,
        group_size=128,
    ) == [{"BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 128}]


def test_prune_cuda_wna16_search_space_keeps_triton_search_space(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.kernels.benchmark_moe.should_moe_wna16_use_cuda",
        lambda **kwargs: False,
    )
    search_space = [
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 64},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 128},
    ]

    assert (
        prune_cuda_wna16_search_space(
            search_space,
            num_valid_tokens=64,
            num_experts=8,
            group_size=128,
        )
        is search_space
    )
