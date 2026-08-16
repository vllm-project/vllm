# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALLREDUCE_MAX_SIZE_MB_LIMIT,
    MiB,
    resolve_custom_allreduce_max_size,
)

DEFAULT = 8 * MiB


def test_unset_keeps_default():
    size, applied = resolve_custom_allreduce_max_size(DEFAULT, 2, True, None)
    assert size == DEFAULT
    assert applied is False


def test_tp2_same_node_override():
    size, applied = resolve_custom_allreduce_max_size(DEFAULT, 2, True, 128)
    assert size == 128 * MiB
    assert applied is True


def test_tp4_ignored():
    size, applied = resolve_custom_allreduce_max_size(DEFAULT, 4, True, 128)
    assert size == DEFAULT
    assert applied is False


def test_multi_node_ignored():
    size, applied = resolve_custom_allreduce_max_size(DEFAULT, 2, False, 128)
    assert size == DEFAULT
    assert applied is False


@pytest.mark.parametrize("bad", [0, -1, CUSTOM_ALLREDUCE_MAX_SIZE_MB_LIMIT + 1])
def test_out_of_range_raises(bad: int):
    with pytest.raises(ValueError, match="VLLM_CUSTOM_ALLREDUCE_MAX_SIZE_MB"):
        resolve_custom_allreduce_max_size(DEFAULT, 2, True, bad)


def test_upper_bound_accepted():
    size, applied = resolve_custom_allreduce_max_size(
        DEFAULT, 2, True, CUSTOM_ALLREDUCE_MAX_SIZE_MB_LIMIT
    )
    assert applied is True
    assert size == CUSTOM_ALLREDUCE_MAX_SIZE_MB_LIMIT * MiB
