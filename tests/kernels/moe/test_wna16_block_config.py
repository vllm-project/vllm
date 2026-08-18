import pytest

from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_moe_wna16_block_config,
)


def test_wna16_block_config_reduces_num_blocks_correctly():
    # This test reproduces the scenario where size_k is divisible by 256 and
    # the initial block_size_k (set from group_size) is < 256. With a large
    # num_experts this makes num_blocks large enough to trigger the branch.
    # The correct behavior is to reduce num_blocks by the factor
    # (256 // old_block_size_k) before deciding to increase BLOCK_SIZE_N.

    cfg = {}
    use_cuda = True
    num_valid_tokens = 100
    size_k = 512
    size_n = 512
    num_experts = 100
    group_size = 128
    real_top_k = 1
    block_size_m = 1

    res = get_moe_wna16_block_config(
        cfg,
        use_cuda,
        num_valid_tokens,
        size_k,
        size_n,
        num_experts,
        group_size,
        real_top_k,
        block_size_m,
    )

    # After the fix, BLOCK_SIZE_N should remain 128 (not increased to 256)
    # because num_blocks is halved when block_size_k is increased from 128 to
    # 256. Also BLOCK_SIZE_K should be 256 (divides size_k and is divisible
    # by group_size).
    assert res["BLOCK_SIZE_N"] == 128
    assert res["BLOCK_SIZE_K"] == 256
