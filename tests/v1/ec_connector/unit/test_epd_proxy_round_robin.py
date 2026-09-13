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

import importlib.util
from collections import Counter
from pathlib import Path

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


@pytest.fixture(scope="module")
def proxy():
    path = Path(__file__).parents[4] / (
        "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_epd_proxy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_generic_e_p_d_routing_without_mooncake_is_supported(proxy):
    proxy.validate_ec_consumer_routing(["http://prefill"], [])


def test_mooncake_e_pd_routing_is_supported(proxy):
    proxy.validate_ec_consumer_routing([], ["tcp://decode:19019"])


def test_mooncake_independent_prefill_routing_fails_fast(proxy):
    with pytest.raises(ValueError, match=r"supports E\+PD only"):
        proxy.validate_ec_consumer_routing(["http://prefill"], ["tcp://decode:19019"])


def test_decode_rewrite_preserves_engine_reported_ec_hash(proxy):
    request = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "image"}}],
            }
        ]
    }

    rewritten = proxy.rewrite_for_decode(
        request,
        {
            0: {
                "mm_hash": "proxy-uuid",
                "ec_mm_hash": "engine-derived-hash",
                "transfer_id": "transfer",
                "image_grid_thw": [1, 2, 3],
            }
        },
    )

    assert rewritten["messages"][0]["content"][0]["uuid"] == "proxy-uuid"
    assert rewritten["messages"][0]["content"][0]["image_embeds"] == {
        "image_grid_thw": [1, 2, 3]
    }
    assert rewritten["ec_transfer_params"]["ec_items"] == [
        {"mm_hash": "engine-derived-hash", "transfer_id": "transfer"}
    ]
