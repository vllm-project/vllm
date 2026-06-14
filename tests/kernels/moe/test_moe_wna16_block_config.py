# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the moe_wna16 CUDA block-size heuristic.

Run `pytest tests/kernels/moe/test_moe_wna16_block_config.py`.
"""

import pytest

from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_moe_wna16_block_config,
)

GROUP_SIZES = [32, 64, 128]
SIZE_KS = [512, 1024, 2048, 4096]
SIZE_NS = [512, 1024, 2048]
NUM_VALID_TOKENS = [1, 8, 16, 64, 256, 4096]
NUM_EXPERTS = [8, 128, 256]
TOP_K = 8


# Regression test for issue #36008: the heuristic could return a BLOCK_SIZE_K
# with BLOCK_SIZE_K // group_size outside {1, 2, 4, 8} (e.g. 512 // 32 == 16),
# which makes the moe_wna16_gemm CUDA kernel abort.
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("size_k", SIZE_KS)
@pytest.mark.parametrize("size_n", SIZE_NS)
@pytest.mark.parametrize("num_valid_tokens", NUM_VALID_TOKENS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
def test_block_config_groups_per_block_row_is_legal(
    group_size, size_k, size_n, num_valid_tokens, num_experts
):
    block_size_m = min(16, max(1, num_valid_tokens // TOP_K))
    config = get_moe_wna16_block_config(
        config={},
        use_moe_wna16_cuda=True,
        num_valid_tokens=num_valid_tokens,
        size_k=size_k,
        size_n=size_n,
        num_experts=num_experts,
        group_size=group_size,
        real_top_k=TOP_K,
        block_size_m=block_size_m,
    )
    block_size_k = config["BLOCK_SIZE_K"]

    # The moe_wna16_gemm kernel requires all three of these.
    assert block_size_k % group_size == 0
    assert size_k % block_size_k == 0
    assert block_size_k // group_size in (1, 2, 4, 8), (
        f"BLOCK_SIZE_K // group_size = {block_size_k // group_size} "
        f"(BLOCK_SIZE_K={block_size_k}, group_size={group_size})"
    )


def test_block_config_reporter_decode_shape():
    # gate_up gemm of Qwen3.5-35B-A3B-GPTQ-4bit (group_size=32) in single-token
    # decode returned BLOCK_SIZE_K=512 (512 // 32 == 16) before the fix.
    config = get_moe_wna16_block_config(
        config={},
        use_moe_wna16_cuda=True,
        num_valid_tokens=8,
        size_k=2048,
        size_n=1024,
        num_experts=256,
        group_size=32,
        real_top_k=8,
        block_size_m=1,
    )
    assert config["BLOCK_SIZE_K"] // 32 in (1, 2, 4, 8)


def test_block_config_keeps_tuned_config():
    # A tuned config that already pins both block sizes is left untouched.
    config = get_moe_wna16_block_config(
        config={"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        use_moe_wna16_cuda=True,
        num_valid_tokens=8,
        size_k=2048,
        size_n=1024,
        num_experts=256,
        group_size=32,
        real_top_k=8,
        block_size_m=1,
    )
    assert config == {}
