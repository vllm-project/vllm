# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import msgspec
import pytest

from vllm.v1.kv_hints.actions import KvHintResult, SetPriority

VALID_SET = {
    "target": "request_cached",
    "location": "local_g1",
    "claim_id": "router/claim",
    "revision": 1,
    "value": 10,
    "ttl_seconds": 60,
}


def make_set_priority(**overrides) -> SetPriority:
    return SetPriority(**{**VALID_SET, **overrides})


def test_set_priority_constructs_and_is_frozen():
    action = make_set_priority()
    assert (action.target, action.location, action.claim_id) == (
        "request_cached",
        "local_g1",
        "router/claim",
    )
    with pytest.raises((AttributeError, TypeError)):
        action.value = 5


def test_clear_claim_omits_ttl():
    action = make_set_priority(value=None, ttl_seconds=None)
    assert action.value is None and action.ttl_seconds is None


def test_clear_claim_rejects_ttl():
    with pytest.raises(ValueError, match="omit ttl_seconds"):
        make_set_priority(value=None)


def test_set_claim_requires_ttl():
    with pytest.raises(ValueError, match="finite positive TTL"):
        make_set_priority(ttl_seconds=None)


@pytest.mark.parametrize("ttl", [0, -1.0, math.inf, math.nan])
def test_set_claim_rejects_non_positive_or_non_finite_ttl(ttl):
    with pytest.raises(ValueError, match="finite positive TTL"):
        make_set_priority(ttl_seconds=ttl)


@pytest.mark.parametrize("claim_id", ["", "x" * 257])
def test_claim_id_length_bounds(claim_id):
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({**VALID_SET, "claim_id": claim_id}, type=SetPriority)


def test_claim_id_upper_bound_accepted():
    action = msgspec.convert({**VALID_SET, "claim_id": "x" * 256}, type=SetPriority)
    assert action.claim_id == "x" * 256


def test_negative_revision_rejected():
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({**VALID_SET, "revision": -1}, type=SetPriority)


def test_unknown_field_rejected_on_decode():
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({**VALID_SET, "extra": 1}, type=SetPriority, strict=True)


def test_wrong_field_type_rejected_on_decode():
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({**VALID_SET, "revision": "1"}, type=SetPriority, strict=True)


@pytest.mark.parametrize(
    "field, value",
    [("target", "remote_hashes"), ("location", "remote_g2")],
)
def test_unsupported_target_or_location_literal(field, value):
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({**VALID_SET, field: value}, type=SetPriority, strict=True)


def test_missing_required_field_rejected_on_decode():
    payload = {k: v for k, v in VALID_SET.items() if k != "claim_id"}
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(payload, type=SetPriority, strict=True)


def test_hint_result_defaults():
    result = KvHintResult("applied")
    assert result.block_ids == () and result.reason is None
