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

    # Reproduce function's local computations to simulate the buggy vs correct
    # reduction behavior and ensure the public return matches the correct path.
    # Initial local state (as in function):
    init_block_size_k = 128
    if init_block_size_k <= group_size:
        init_block_size_k = group_size

    num_n_blocks = size_k // init_block_size_k
    num_k_blocks = size_n // init_block_size_k
    num_m_blocks = (num_valid_tokens + block_size_m - 1) / block_size_m + num_experts
    if num_valid_tokens // real_top_k <= block_size_m:
        num_m_blocks = min(num_m_blocks, num_valid_tokens)
    num_blocks = num_m_blocks * num_n_blocks * num_k_blocks

    # Buggy path (original code): recompute block_size_k then divide by (256 // block_size_k)
    buggy_block_size_k = init_block_size_k
    buggy_num_blocks = num_blocks
    if size_k % 256 == 0 and buggy_num_blocks >= 256 and buggy_block_size_k < 256:
        buggy_block_size_k = 256
        # divisor uses the new block_size_k (bug): 256 // 256 == 1
        buggy_num_blocks = buggy_num_blocks // (256 // buggy_block_size_k)

    # Correct path: compute divisor before clobbering
    correct_block_size_k = init_block_size_k
    correct_num_blocks = num_blocks
    if size_k % 256 == 0 and correct_num_blocks >= 256 and correct_block_size_k < 256:
        old_bk = correct_block_size_k
        correct_block_size_k = 256
        correct_num_blocks = correct_num_blocks // (256 // old_bk)

    # Simulate further heuristics for BLOCK_SIZE_N on both paths.
    def finalize(block_size_k_val, num_blocks_val, block_size_n_val):
        # follow the same checks as get_moe_wna16_block_config
        # Note: we only need to simulate decisions that affect BLOCK_SIZE_N here.
        if (
            num_m_blocks <= 16
            and size_k % (block_size_k_val * 2) == 0
            and size_k % (block_size_k_val * 2) == 0
            and block_size_k_val <= 512
            and num_blocks_val >= 512
        ):
            block_size_k_val = block_size_k_val * 2
            num_blocks_val = num_blocks_val // 2

        if num_blocks_val > 1024:
            block_size_n_val = 256
            num_blocks_val = num_blocks_val // 2

        if size_n <= 1024 and num_blocks_val >= 1024:
            block_size_n_val = 1024

        # ensure divisibility like the function does at the end for BLOCK_SIZE_K
        # but for this test we only care that BLOCK_SIZE_N follows the correct path
        return block_size_n_val, block_size_k_val

    buggy_bs_n, buggy_bs_k = finalize(buggy_block_size_k, buggy_num_blocks, 128)
    correct_bs_n, correct_bs_k = finalize(correct_block_size_k, correct_num_blocks, 128)

    # The function result should match the correct simulated path, not the buggy one.
    assert res["BLOCK_SIZE_N"] == correct_bs_n
    assert res["BLOCK_SIZE_K"] == correct_bs_k
