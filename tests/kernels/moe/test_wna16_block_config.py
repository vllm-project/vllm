from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_moe_wna16_block_config,
)


def test_wna16_block_config_reduces_num_blocks_before_clobbering_block_size_k():
    # Regression test for #52590: with these shapes num_blocks = 3200 triggers
    # the reduction branch. The old code computed the divisor after setting
    # block_size_k = 256 (256 // 256 == 1), so num_blocks stayed inflated and
    # BLOCK_SIZE_N jumped to 1024 instead of 256.
    cfg = get_moe_wna16_block_config(
        {},
        use_moe_wna16_cuda=True,
        num_valid_tokens=100,
        size_k=512,
        size_n=512,
        num_experts=100,
        group_size=128,
        real_top_k=1,
        block_size_m=1,
    )
    assert cfg == {"BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256}
