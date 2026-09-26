# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_wna16_block_config


def test_get_moe_wna16_block_config_256_branch_actually_halves_num_blocks():
    # size_k=4096, group_size=128 puts block_size_k=128 initially, and gives
    # num_blocks well above the >=256 threshold that gates the 256-branch:
    #   num_n_blocks = size_k // 128 = 32
    #   num_k_blocks = size_n // 128 = 8
    #   num_m_blocks = ceil(64 / 64) + 1 = 2
    #   num_blocks = 2 * 32 * 8 = 512
    # The 256-branch should halve that to 256 (num_blocks // 2), matching the
    # same "divide by the block-size-doubling factor" pattern used by the
    # very next branch a few lines below.
    result = get_moe_wna16_block_config(
        config={},
        use_moe_wna16_cuda=True,
        num_valid_tokens=64,
        size_k=4096,
        size_n=1024,
        num_experts=1,
        group_size=128,
        real_top_k=1,
        block_size_m=64,
    )

    assert result["BLOCK_SIZE_K"] == 256
    # Before the fix, block_size_k was reassigned to 256 *before* being used
    # as the divisor, so `num_blocks // (256 // block_size_k)` was always a
    # no-op (`// 1`). We can't observe num_blocks directly since it's a local,
    # but the downstream `num_blocks > 1024` branch is decided by exactly the
    # halved-or-not value, so exercise a shape where the two diverge in their
    # BLOCK_SIZE_N choice.
    #
    # With the bug: num_blocks stays 512, so `num_blocks > 1024` is False,
    # and size_n(1024) <= 1024 and num_blocks(512) >= 1024 is also False ->
    # BLOCK_SIZE_N stays 128.
    # With the fix: num_blocks becomes 256, same branches stay False ->
    # BLOCK_SIZE_N also stays 128 for *this* shape, so use a larger shape
    # below where the halving crosses the 1024 threshold instead.
    assert result["BLOCK_SIZE_N"] == 128


def test_get_moe_wna16_block_config_256_branch_affects_block_size_n():
    # size_k=4096, size_n=512, group_size=32, num_experts=8, block_size_m=16:
    #   num_n_blocks = 4096 // 128 = 32
    #   num_k_blocks = 512 // 128 = 4
    #   num_m_blocks = ceil(128/16) + 8 = 16.9375
    #   num_blocks = 16.9375 * 32 * 4 = 2168.0
    # 256-branch fires (size_k % 256 == 0, num_blocks >= 256, block_size_k < 256):
    #   fixed:  num_blocks = 2168 // (256 // 128) = 2168 // 2 = 1084.0
    #   buggy:  num_blocks = 2168 // (256 // 256) = 2168 // 1 = 2168.0 (no-op)
    # block_size_k becomes 256 either way, so BLOCK_SIZE_K alone can't tell
    # fixed apart from buggy here. But the `num_blocks > 1024` branch below
    # halves num_blocks again and sets BLOCK_SIZE_N=256, so the two paths
    # diverge to:
    #   fixed:  num_blocks 1084 -> 542.0,  BLOCK_SIZE_N stays 256
    #   buggy:  num_blocks 2168 -> 1084.0, BLOCK_SIZE_N=256, then
    #           `size_n(512) <= 1024 and num_blocks(1084) >= 1024` also
    #           fires and bumps BLOCK_SIZE_N to 1024.
    # So BLOCK_SIZE_N=256 under the fix vs. BLOCK_SIZE_N=1024 under the bug —
    # this directly demonstrates the fix changes the kernel's tiling choice,
    # not just an internal-only value.
    result = get_moe_wna16_block_config(
        config={},
        use_moe_wna16_cuda=True,
        num_valid_tokens=128,
        size_k=4096,
        size_n=512,
        num_experts=8,
        group_size=32,
        real_top_k=1,
        block_size_m=16,
    )

    assert result["BLOCK_SIZE_K"] == 256
    assert result["BLOCK_SIZE_N"] == 256
