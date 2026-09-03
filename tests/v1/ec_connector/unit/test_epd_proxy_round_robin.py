# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder fan-out fairness in the disaggregated EPD proxy.

Regression target: the fan-out cursor used to restart at the first encoder on
every incoming request, so single-item requests -- the common case -- always
hit the first instance and left the rest idle. The registry threads one
cursor across calls instead.

A dynamic roster adds a second way to lose that cursor: rebuilding an
`itertools.cycle` whenever an instance registers or leaves resets the
position, which is the same hot spot arriving on every registration.
"""

from collections import Counter

import pytest

from vllm.distributed.ec_transfer.proxy.registry import (
    InstanceRecord,
    InstanceRegistry,
    InstanceRole,
)

ENCODE = InstanceRole.ENCODE


@pytest.fixture
def registry():
    return InstanceRegistry(probe_interval=0)


def _with_encoders(registry, count):
    for index in range(count):
        registry.register(InstanceRecord(ENCODE, f"E{index}"))
    return registry


def _drive(registry, counts):
    """Feed a sequence of per-request item counts through the cursor."""
    return [
        [record.url for record in registry.pick_many(ENCODE, count)] for count in counts
    ]


@pytest.mark.parametrize("n_urls", [1, 2, 3, 5])
def test_full_url_space_is_covered_uniformly(registry, n_urls):
    _with_encoders(registry, n_urls)
    routed = _drive(registry, counts=[1] * (n_urls * 3))
    hits = Counter(url for request in routed for url in request)
    assert set(hits) == {f"E{index}" for index in range(n_urls)}
    assert max(hits.values()) == min(hits.values())


def test_single_item_requests_rotate_through_all_encoders(registry):
    _with_encoders(registry, 3)
    routed = _drive(registry, counts=[1] * 6)
    assert [request[0] for request in routed] == ["E0", "E1", "E2"] * 2


def test_cursor_stays_contiguous_across_varying_item_counts(registry):
    _with_encoders(registry, 3)
    assert _drive(registry, counts=[2, 1, 3, 1]) == [
        ["E0", "E1"],
        ["E2"],
        ["E0", "E1", "E2"],
        ["E0"],
    ]


def test_single_encoder_always_resolves_to_it(registry):
    _with_encoders(registry, 1)
    assert _drive(registry, counts=[4]) == [["E0"] * 4]


def test_a_registration_does_not_restart_the_rotation(registry):
    _with_encoders(registry, 3)
    assert _drive(registry, counts=[2]) == [["E0", "E1"]]
    registry.register(InstanceRecord(ENCODE, "E3"))
    assert _drive(registry, counts=[2]) == [["E2", "E3"]]


def test_an_evicted_encoder_drops_out_of_the_rotation(registry):
    _with_encoders(registry, 3)
    registry.unregister("E1")
    routed = _drive(registry, counts=[1] * 4)
    assert [request[0] for request in routed] == ["E0", "E2", "E0", "E2"]


def test_no_encoder_yields_nothing(registry):
    assert registry.pick_many(ENCODE, 2) == []
    assert registry.pick(ENCODE) is None
