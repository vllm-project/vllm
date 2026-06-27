# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the moe_wna16 CUDA block-size heuristic (issue #36008).

moe_wna16_gemm is only instantiated for BLOCK_SIZE_K // group_size in
{1, 2, 4, 8}, so the heuristic must never return any other ratio. group_size=32
with a size_k that is not a power-of-two multiple of the group is what trips it.
"""

import pytest

from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_wna16_block_config

TOP_K = 8
GROUP_SIZES = [32, 64, 128]
# Mix of power-of-two and ratio-3-inducing shapes (96, 1056, 1344, 2016).
SIZE_KS = [96, 512, 1056, 1344, 2016, 2048, 4096]
NUM_VALID_TOKENS = [1, 8, 16, 64, 4096]


@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("size_k", SIZE_KS)
@pytest.mark.parametrize("num_valid_tokens", NUM_VALID_TOKENS)
@pytest.mark.parametrize("num_experts", [8, 256])
def test_block_config_ratio_is_kernel_legal(
    group_size, size_k, num_valid_tokens, num_experts
):
    if size_k % group_size != 0:
        pytest.skip("size_k is always a multiple of group_size")
    block_size_m = min(16, max(1, num_valid_tokens // TOP_K))
    config = get_moe_wna16_block_config(
        config={},
        use_moe_wna16_cuda=True,
        num_valid_tokens=num_valid_tokens,
        size_k=size_k,
        size_n=1024,
        num_experts=num_experts,
        group_size=group_size,
        real_top_k=TOP_K,
        block_size_m=block_size_m,
    )
    block_size_k = config["BLOCK_SIZE_K"]
    assert size_k % block_size_k == 0
    assert block_size_k % group_size == 0
    assert block_size_k // group_size in (1, 2, 4, 8), (
        f"BLOCK_SIZE_K // group_size = {block_size_k // group_size} "
        f"(BLOCK_SIZE_K={block_size_k}, group_size={group_size}, size_k={size_k})"
    )
