# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.gpu.sample.logprob import topk_logprob_block_sizes


def test_topk_logprob_block_sizes_follow_the_launch_rule():
    assert topk_logprob_block_sizes(0) == ()
    # Default max_logprobs is 20. Width is the sampled id plus those ids (21),
    # and the kernel rounds that up to 32. prompt_logprobs=3 needs 4.
    assert topk_logprob_block_sizes(20) == (1, 2, 4, 8, 16, 32)
    assert topk_logprob_block_sizes(1) == (1, 2)
    # -1 allows a vocab-sized request, capped at the kernel's 1024-wide gather.
    assert topk_logprob_block_sizes(-1)[-1] == 1024
    assert topk_logprob_block_sizes(-1)[0] == 1
    assert len(topk_logprob_block_sizes(-1)) == 11
