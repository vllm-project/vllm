# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft window: sink pages packed in front of the recent tokens."""

import torch

from vllm.v1.attention.backends.rocm_aiter_fa_dspark import (
    DSPARK_DRAFT_SINK,
    DSPARK_DRAFT_WINDOW,
)
from vllm.v1.attention.ops.dspark_draft_window import build_draft_window_sink_tables


def test_short_sequence_keeps_its_real_pages():
    block = 64
    table = torch.arange(8, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([100], dtype=torch.int32)
    pages, packed = build_draft_window_sink_tables(
        table,
        seq_lens,
        block_size=block,
        window=DSPARK_DRAFT_WINDOW,
        sink=DSPARK_DRAFT_SINK,
    )
    assert int(packed[0]) == 100
    assert int(pages[0, 0]) == 0
    assert int(pages[0, 1]) == 1


def test_long_sequence_keeps_the_sink_and_the_recent_window():
    block = 64
    n_blocks = 200
    table = torch.arange(n_blocks, dtype=torch.int32).view(1, -1)
    seq_len = 10_000
    pages, packed = build_draft_window_sink_tables(
        table,
        torch.tensor([seq_len], dtype=torch.int32),
        block_size=block,
        window=DSPARK_DRAFT_WINDOW,
        sink=DSPARK_DRAFT_SINK,
    )
    # One sink page (4 tokens round up to 64) plus the tokens from the window.
    win_start = max(1, (seq_len - DSPARK_DRAFT_WINDOW) // block)
    expect = block + (seq_len - win_start * block)
    assert int(packed[0]) == expect
    assert int(pages[0, 0]) == 0
    assert int(pages[0, 1]) == win_start
    used = (seq_len + block - 1) // block
    assert int(pages[0, used - win_start]) == used - 1


def test_draft_block_stays_under_an_ultraquant_page():
    from vllm.v1.attention.backends.rocm_aiter_fa_dspark import (
        AiterFlashAttentionDSparkBackend,
    )
    from vllm.v1.kv_cache_interface import AttentionSpec

    spec = AttentionSpec(
        block_size=3136,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )
    capped = AiterFlashAttentionDSparkBackend.customize_spec(spec)
    assert capped.block_size == 784


def test_sink_window_reserves_the_leading_block():
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    config = SlidingWindowManager.config(sliding_window=2048, sink_len=4)
    blocks = SlidingWindowManager.max_admission_blocks(
        config,
        block_size=784,
        max_in_flight_tokens=8192,
        max_model_len=262144,
    )
    assert blocks == cdiv(2048 - 1 + 8192, 784) + 1 + 1
    assert SlidingWindowManager.prefix_cacheable(config) is False


def test_sink_group_is_not_masked_by_the_worker():
    """The worker nulls every block before `reach`. That would null the pinned
    sink page, and a large null page of zero keys dilutes the softmax of the
    whole draft. A sink group must opt out of the mask."""
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    plain = SlidingWindowManager.config(sliding_window=DSPARK_DRAFT_WINDOW)
    sink = SlidingWindowManager.config(
        sliding_window=DSPARK_DRAFT_WINDOW, sink_len=DSPARK_DRAFT_SINK
    )
    assert SlidingWindowManager.sliding_window_reach(plain) == DSPARK_DRAFT_WINDOW - 1
    assert SlidingWindowManager.sliding_window_reach(sink) == 0
