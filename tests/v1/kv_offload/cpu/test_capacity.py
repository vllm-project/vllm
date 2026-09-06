# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CPU offloading tier capacity estimate.

The estimate answers one question: how many KV tokens does a full tier serve
at the best request length up to max_model_len? Both functions under test take
plain arguments, so these tests need no spec and no engine.
"""

import pytest

from vllm.v1.kv_offload.config import OffloadingGroupConfig
from vllm.v1.kv_offload.cpu.spec import (
    _capacity_tokens_at_max_len,
    _chunks_per_request,
)

MAX_MODEL_LEN = 36864

# Group shapes as (tokens_per_block, sliding_window_size_in_chunks) pairs.
FULL_16 = ((16, None),)
# One full attention group and 6 sliding window groups of 1024 tokens.
GEMMA = ((16, None),) + ((16, 4),) * 6
# One full attention group and 9 Mamba groups of one state each.
GRANITE = ((528, None),) + ((528, 1),) * 9
# One full attention group and one hidden state group on a smaller block.
FULL_PLUS_HIDDEN = ((16, None), (8, None))
# One group of each kind: full attention, a 1024 token window, and Mamba.
HYBRID = ((16, None), (16, 4), (16, 1))


def make_groups(
    *shapes: tuple[int, int | None],
) -> tuple[OffloadingGroupConfig, ...]:
    """Build a group tuple from (tokens_per_block, window_chunks) pairs.

    Args:
        shapes: One pair for each group of the model.

    Returns:
        The groups, each with a distinct single layer name.
    """
    return tuple(
        OffloadingGroupConfig(
            tokens_per_block=tokens_per_block,
            layer_names=(f"layer.{index}",),
            sliding_window_size_in_chunks=window_chunks,
        )
        for index, (tokens_per_block, window_chunks) in enumerate(shapes)
    )


@pytest.mark.parametrize(
    ("shapes", "blocks_per_chunk", "num_chunks", "expected"),
    [
        pytest.param(FULL_16, 16, 4500, 1152000, id="single-full-group"),
        pytest.param(((16, 16),), 16, 4000, 9216000, id="uniform-swa-4096"),
        pytest.param(((16, 4),), 16, 4000, 36864000, id="uniform-swa-1024"),
        pytest.param(GEMMA, 16, 4000, 877714, id="full-plus-6-swa-1024"),
        pytest.param(GRANITE, 1, 8000, 3736615, id="full-plus-9-mamba"),
        pytest.param(((528, 1),), 1, 8000, 294912000, id="pure-mamba"),
        pytest.param(FULL_PLUS_HIDDEN, 16, 4500, 384000, id="full-plus-hidden"),
        pytest.param(HYBRID, 16, 4000, 989637, id="full-plus-swa-plus-mamba"),
    ],
)
def test_capacity_over_model_shapes(
    shapes: tuple[tuple[int, int | None], ...],
    blocks_per_chunk: int,
    num_chunks: int,
    expected: int,
) -> None:
    """The estimate reports the reviewed capacity of each model shape.

    A window raises the capacity above the byte count of the tier, because a
    windowed group holds a fixed chunk count however long the request grows.
    """
    capacity = _capacity_tokens_at_max_len(
        make_groups(*shapes), blocks_per_chunk, num_chunks, MAX_MODEL_LEN
    )

    assert capacity == expected


@pytest.mark.parametrize(
    ("tokens_per_block", "blocks_per_chunk", "num_chunks"),
    [
        pytest.param(16, 16, 4500, id="qwen3-8b-shape"),
        pytest.param(528, 1, 8000, id="chunk-size-divides-max-len-with-remainder"),
        pytest.param(8, 16, 4500, id="small-block"),
    ],
)
def test_single_uncapped_group_keeps_the_exact_token_count(
    tokens_per_block: int, blocks_per_chunk: int, num_chunks: int
) -> None:
    """One group without a window keeps the exact count of the old metric.

    The old capacity_tokens divided the byte budget by the byte cost of one
    token. The estimate must agree with it on this shape.
    """
    groups = make_groups((tokens_per_block, None))

    capacity = _capacity_tokens_at_max_len(
        groups, blocks_per_chunk, num_chunks, MAX_MODEL_LEN
    )

    assert capacity == num_chunks * blocks_per_chunk * tokens_per_block


def test_empty_tier_reports_an_exact_zero() -> None:
    """A tier of no slots serves no tokens, which is a known value."""
    capacity = _capacity_tokens_at_max_len(make_groups(*FULL_16), 16, 0, MAX_MODEL_LEN)

    assert capacity == 0


@pytest.mark.parametrize(
    ("shapes", "max_model_len"),
    [
        pytest.param(FULL_16, 0, id="max-model-len-not-known"),
        pytest.param((), MAX_MODEL_LEN, id="no-groups"),
        pytest.param(((0, None),), MAX_MODEL_LEN, id="blocks-span-no-tokens"),
    ],
)
def test_capacity_is_none_when_the_token_scale_is_unknown(
    shapes: tuple[tuple[int, int | None], ...], max_model_len: int
) -> None:
    """The estimate reports None when an input leaves the token scale open."""
    capacity = _capacity_tokens_at_max_len(
        make_groups(*shapes), 16, 4500, max_model_len
    )

    assert capacity is None


@pytest.mark.parametrize(
    ("shapes", "blocks_per_chunk", "seq_len", "expected"),
    [
        pytest.param(FULL_16, 16, 36864, 144, id="one-full-group"),
        pytest.param(GEMMA, 16, 36864, 168, id="full-plus-6-swa-1024"),
        pytest.param(GRANITE, 1, 36432, 78, id="full-plus-9-mamba"),
        pytest.param(((16, 4),), 16, 256, 1, id="request-below-the-window"),
        pytest.param(((16, 4),), 16, 36864, 4, id="window-caps-the-request"),
    ],
)
def test_chunks_per_request(
    shapes: tuple[tuple[int, int | None], ...],
    blocks_per_chunk: int,
    seq_len: int,
    expected: int,
) -> None:
    """One request holds one chunk for each chunk-sized span, up to the cap."""
    chunks = _chunks_per_request(make_groups(*shapes), blocks_per_chunk, seq_len)

    assert chunks == expected


@pytest.mark.parametrize(
    ("shapes", "blocks_per_chunk"),
    [
        pytest.param(GEMMA, 16, id="full-plus-6-swa-1024"),
        pytest.param(((16, None), (24, None)), 16, id="chunk-sizes-do-not-nest"),
    ],
)
def test_candidate_lengths_reach_the_capacity_peak(
    shapes: tuple[tuple[int, int | None], ...], blocks_per_chunk: int
) -> None:
    """The estimate matches a sweep of every request length.

    The estimate measures a few candidate lengths and keeps the largest
    result. The sweep shows that the candidate set holds the peak of the
    capacity curve. A short max_model_len keeps the sweep fast.
    """
    max_model_len = 4096
    num_chunks = 1000
    groups = make_groups(*shapes)

    peak = max(
        num_chunks * seq_len // _chunks_per_request(groups, blocks_per_chunk, seq_len)
        for seq_len in range(1, max_model_len + 1)
    )

    capacity = _capacity_tokens_at_max_len(
        groups, blocks_per_chunk, num_chunks, max_model_len
    )

    assert capacity == peak
