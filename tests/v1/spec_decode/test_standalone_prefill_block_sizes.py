# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import (
    standalone_prefill_block_sizes,
)


def test_standalone_prefill_block_sizes_follow_the_launch_rule():
    # max_num_batched_tokens=64 covers a 64-token chunk: query + 2 rounds up to 128.
    assert standalone_prefill_block_sizes(64) == (4, 8, 16, 32, 64, 128)
    assert standalone_prefill_block_sizes(32) == (4, 8, 16, 32, 64)
    assert standalone_prefill_block_sizes(8192)[-1] == 512
    assert 4 in standalone_prefill_block_sizes(1)
