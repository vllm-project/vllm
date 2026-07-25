# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the --rank-tp-ratio uneven-TP partition helpers."""

import pytest

from vllm.distributed.utils import (
    get_tp_partition_ratios,
    set_tp_partition_ratios,
    tp_partition_offset,
    tp_partition_size,
    tp_partition_sizes,
)


@pytest.fixture(autouse=True)
def clean_ratios():
    set_tp_partition_ratios(None)
    yield
    set_tp_partition_ratios(None)


def test_ratios_unset_by_default():
    assert get_tp_partition_ratios() is None


def test_set_and_get_ratios():
    set_tp_partition_ratios([2, 1, 1])
    assert get_tp_partition_ratios() == [2, 1, 1]
    set_tp_partition_ratios(None)
    assert get_tp_partition_ratios() is None


def test_even_split_without_ratios():
    assert tp_partition_sizes(32, 4) == [8, 8, 8, 8]
    assert tp_partition_size(32, 4, 2) == 8
    assert tp_partition_offset(32, 4, 2) == 16


def test_even_split_requires_divisibility():
    with pytest.raises(AssertionError):
        tp_partition_sizes(10, 4)


def test_ratio_split():
    set_tp_partition_ratios([2, 1, 1])
    assert tp_partition_sizes(32, 3) == [16, 8, 8]
    assert [tp_partition_size(32, 3, r) for r in range(3)] == [16, 8, 8]
    # Offsets are prefix sums of the per-rank sizes.
    assert [tp_partition_offset(32, 3, r) for r in range(3)] == [0, 16, 24]


def test_ratio_split_covers_dimension_exactly():
    set_tp_partition_ratios([3, 2, 1])
    sizes = tp_partition_sizes(48, 3)
    assert sizes == [24, 16, 8]
    assert sum(sizes) == 48
    assert tp_partition_offset(48, 3, 2) + sizes[2] == 48


def test_ratio_split_rejects_non_divisible():
    set_tp_partition_ratios([2, 1, 1])
    with pytest.raises(ValueError, match="not divisible by"):
        tp_partition_sizes(30, 3)


def test_ratio_length_mismatch_falls_back_to_even():
    # Layers running with their own tp_size (e.g. disable_tp -> tp_size=1)
    # must not be affected by an installed ratio vector.
    set_tp_partition_ratios([2, 1, 1])
    assert tp_partition_sizes(32, 1) == [32]
    assert tp_partition_sizes(32, 4) == [8, 8, 8, 8]


def test_arg_validation_ratio_rules():
    from vllm.engine.arg_utils import EngineArgs

    class _FakePlatform:
        @staticmethod
        def device_count() -> int:
            return 3

    def validate(**kwargs):
        args = EngineArgs(model="facebook/opt-125m", **kwargs)
        args._validate_rank_gpu_config(_FakePlatform())

    # --rank-tp-ratio requires --rank-gpu-id.
    with pytest.raises(ValueError, match="requires --rank-gpu-id"):
        validate(tensor_parallel_size=3, rank_tp_ratio=[2, 1, 1])

    common = dict(
        tensor_parallel_size=3,
        rank_gpu_id=[0, 1, 2],
        rank_gpu_memory_mib=15000,
    )
    # Length must equal tensor_parallel_size.
    with pytest.raises(ValueError, match="--rank-tp-ratio length"):
        validate(rank_tp_ratio=[2, 1], **common)
    # Entries must be positive.
    with pytest.raises(ValueError, match="positive integers"):
        validate(rank_tp_ratio=[2, 0, 1], **common)
