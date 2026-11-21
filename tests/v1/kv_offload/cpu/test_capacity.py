# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cpu_kv_cache_size_tokens, the CPU tier capacity at max_model_len."""

import pytest

from tests.v1.kv_offload.conftest import make_offloading_config
from vllm.v1.kv_offload.config import OffloadingGroupConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingInfo

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
    return tuple(
        OffloadingGroupConfig(
            tokens_per_block=tokens_per_block,
            layer_names=(f"layer.{index}",),
            group_id=index,
            sliding_window_size_in_chunks=window_chunks,
        )
        for index, (tokens_per_block, window_chunks) in enumerate(shapes)
    )


def size_tokens(
    shapes: tuple[tuple[int, int | None], ...],
    blocks_per_chunk: int,
    num_chunks: int,
    max_model_len: int = MAX_MODEL_LEN,
) -> int | None:
    config = make_offloading_config(
        groups=make_groups(*shapes),
        blocks_per_chunk=blocks_per_chunk,
        max_model_len=max_model_len,
    )
    info = CPUOffloadingInfo.from_config(config, num_chunks, kv_bytes_per_chunk=1)
    return info.cpu_kv_cache_size_tokens


@pytest.mark.parametrize(
    ("shapes", "blocks_per_chunk", "num_chunks", "expected"),
    [
        # 4500 * 36864 // 144
        pytest.param(FULL_16, 16, 4500, 1152000, id="single-full-group"),
        # 4000 * 36864 // min(144, 16)
        pytest.param(((16, 16),), 16, 4000, 9216000, id="uniform-swa-4096"),
        # 4000 * 36864 // min(144, 4)
        pytest.param(((16, 4),), 16, 4000, 36864000, id="uniform-swa-1024"),
        # 4000 * 36864 // (144 + 6 * 4)
        pytest.param(GEMMA, 16, 4000, 877714, id="full-plus-6-swa-1024"),
        # 8000 * 36864 // (70 + 9 * 1)
        pytest.param(GRANITE, 1, 8000, 3733063, id="full-plus-9-mamba"),
        # 8000 * 36864 // min(70, 1)
        pytest.param(((528, 1),), 1, 8000, 294912000, id="pure-mamba"),
        # 4500 * 36864 // (144 + 288)
        pytest.param(FULL_PLUS_HIDDEN, 16, 4500, 384000, id="full-plus-hidden"),
        # 4000 * 36864 // (144 + 4 + 1)
        pytest.param(HYBRID, 16, 4000, 989637, id="full-plus-swa-plus-mamba"),
        # A tier of no slots holds no tokens, which is a known value.
        pytest.param(FULL_16, 16, 0, 0, id="empty-tier"),
    ],
)
def test_size_tokens_over_model_shapes(
    shapes: tuple[tuple[int, int | None], ...],
    blocks_per_chunk: int,
    num_chunks: int,
    expected: int,
) -> None:
    """A windowed group holds a fixed chunk count, so it raises the capacity."""
    assert size_tokens(shapes, blocks_per_chunk, num_chunks) == expected


@pytest.mark.parametrize(
    ("shapes", "max_model_len"),
    [
        pytest.param(FULL_16, 0, id="max-model-len-not-known"),
        pytest.param((), MAX_MODEL_LEN, id="no-groups"),
    ],
)
def test_size_tokens_is_none_when_unknown(
    shapes: tuple[tuple[int, int | None], ...], max_model_len: int
) -> None:
    assert size_tokens(shapes, 16, 4500, max_model_len) is None
