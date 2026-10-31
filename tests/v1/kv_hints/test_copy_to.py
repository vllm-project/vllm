# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.prefetch import (
    PrefetchSubmitOutcome,
    PrefetchSubmitResult,
)
from vllm.v1.kv_hints.actions import CopyTo
from vllm.v1.kv_hints.copy_to import _map_outcome, apply_copy_to

HASH_HEX = ["aa" * 32, "bb" * 32]


def make_copy_to(**overrides) -> CopyTo:
    payload: dict[str, Any] = {
        "target": list(HASH_HEX),
        "src": {"tier": "cpu"},
        "dst": {"tier": "gpu"},
    }
    for key, value in overrides.items():
        if key in ("src", "dst"):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return msgspec.convert(payload, type=CopyTo)


class FakePrefetchConnector:
    def __init__(self, result=None):
        self.result = result or PrefetchSubmitResult(PrefetchSubmitOutcome.ACCEPTED)
        self.calls: list[list[bytes]] = []

    def request_free_prefetch(self, block_hashes):
        self.calls.append(block_hashes)
        return self.result


class FakeNonOffloadingConnector:
    """Any connector without request-free prefetch (e.g. NIXL)."""


def test_copy_to_payload_rejects_unknown_fields_and_bad_tiers():
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(
            {
                "target": [],
                "src": {"tier": "gpu"},
                "dst": {"tier": "gpu"},
                "extra": 1,
            },
            type=CopyTo,
        )
    with pytest.raises(msgspec.ValidationError):
        make_copy_to(src={"tier": "tape"})
    with pytest.raises(msgspec.ValidationError):
        make_copy_to(target="not-a-list")


def test_copy_to_requires_local_cpu_to_gpu():
    connector = FakePrefetchConnector()
    assert apply_copy_to(connector, make_copy_to(dst={"tier": "cpu"}), None).status == (
        "unsupported"
    )
    assert apply_copy_to(connector, make_copy_to(src={"tier": "gpu"}), None).status == (
        "unsupported"
    )
    assert apply_copy_to(connector, make_copy_to(), None).status == "applied"


def test_copy_to_rejects_cross_engine():
    connector = FakePrefetchConnector()
    result = apply_copy_to(connector, make_copy_to(src={"engine_id": "engine-2"}), None)
    assert result.status == "unsupported"
    assert connector.calls == []


def test_copy_to_rejects_non_hex_target():
    connector = FakePrefetchConnector()
    result = apply_copy_to(connector, make_copy_to(target=["zz"]), None)
    assert result.status == "rejected"
    assert "non-hex" in result.reason
    assert connector.calls == []


def test_copy_to_passes_engine_block_hashes():
    connector = FakePrefetchConnector()
    apply_copy_to(connector, make_copy_to(), None)
    assert connector.calls == [[b"\xaa" * 32, b"\xbb" * 32]]


def test_copy_to_without_offloading_connector():
    assert (
        apply_copy_to(FakeNonOffloadingConnector(), make_copy_to(), None).status
        == "unsupported"
    )
    assert apply_copy_to(None, make_copy_to(), None).status == "unsupported"


@pytest.mark.parametrize(
    "outcome, status",
    [
        (PrefetchSubmitOutcome.ACCEPTED, "applied"),
        (PrefetchSubmitOutcome.ALREADY_AVAILABLE, "applied"),
        (PrefetchSubmitOutcome.DEFERRED, "deferred"),
        (PrefetchSubmitOutcome.MISSING, "rejected"),
        (PrefetchSubmitOutcome.UNSUPPORTED, "unsupported"),
    ],
)
def test_outcome_mapping(outcome, status):
    result = PrefetchSubmitResult(outcome, prefetch_id="p1")
    assert _map_outcome(result).status == status
