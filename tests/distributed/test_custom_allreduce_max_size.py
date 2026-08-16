# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALLREDUCE_MAX_SIZE_MB_LIMIT,
    MiB,
    resolve_custom_allreduce_max_size,
)
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

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


def test_override_supersedes_smaller_table_adjusted_size():
    """Constructor applies the env after SM-table min(); 128 must still win."""
    table_adjusted = 4 * MiB
    size, applied = resolve_custom_allreduce_max_size(table_adjusted, 2, True, 128)
    assert applied is True
    assert size == 128 * MiB


def test_value_below_default_lowers_ceiling():
    size, applied = resolve_custom_allreduce_max_size(DEFAULT, 2, True, 4)
    assert applied is True
    assert size == 4 * MiB


def _fake_ar(
    *,
    world_size: int = 2,
    max_size: int = 128 * MiB,
    disabled: bool = False,
    fully_connected: bool = False,
) -> CustomAllreduce:
    ar = object.__new__(CustomAllreduce)
    ar.disabled = disabled
    ar._ptr = 0
    ar.world_size = world_size
    ar.max_size = max_size
    ar.fully_connected = fully_connected
    return ar


def test_should_custom_ar_uses_strict_less_than():
    # Analogous to an 80 MiB prefill under a 128 MiB ceiling: CUSTOM wins
    # only when inp_size < max_size (existing semantics, not <=).
    ar = _fake_ar(max_size=1024)
    below = torch.empty(1008, dtype=torch.uint8)
    exact = torch.empty(1024, dtype=torch.uint8)
    assert ar.should_custom_ar(below) is True
    assert ar.should_custom_ar(exact) is False


def test_should_custom_ar_below_raised_ceiling():
    # Same relation as the 80 MiB residual vs a 128 MiB ceiling.
    ar = _fake_ar(max_size=128 * 1024)
    eighty = torch.empty(80 * 1024, dtype=torch.uint8)
    assert ar.should_custom_ar(eighty) is True
    ar.max_size = 80 * 1024
    assert ar.should_custom_ar(eighty) is False
