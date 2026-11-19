# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offload_limits import (
    RequestOffloadLimits,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("key", ["max_offload_tokens", "max_load_tokens"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, 0), (100, 100), (-1, None), (True, None), (1.5, None), ("8", None)],
)
def test_parses_non_negative_int_limits(key, raw, expected):
    limits = RequestOffloadLimits.from_kv_transfer_params({key: raw})
    assert getattr(limits, key) == expected


@pytest.mark.parametrize("params", [None, {}, {"max_offload_tokens": None}])
def test_missing_limits_are_unbounded(params):
    limits = RequestOffloadLimits.from_kv_transfer_params(params)
    assert limits == RequestOffloadLimits()
    assert not limits.store_disabled
    assert limits.cap_store(128) == 128
    assert limits.can_store(128)
    assert limits.cap_load(128, 16) == 128


def test_store_limit():
    assert RequestOffloadLimits(max_offload_tokens=0).store_disabled
    limits = RequestOffloadLimits(max_offload_tokens=100)
    assert not limits.store_disabled
    assert limits.cap_store(128) == 100
    assert limits.cap_store(64) == 64
    assert limits.can_store(100)
    assert not limits.can_store(101)


def test_load_limit_is_relative_to_computed_tokens():
    limits = RequestOffloadLimits(max_load_tokens=32)
    assert limits.cap_load(128, 16) == 48
    assert limits.cap_load(40, 16) == 40
    assert RequestOffloadLimits(max_load_tokens=0).cap_load(128, 16) == 16
