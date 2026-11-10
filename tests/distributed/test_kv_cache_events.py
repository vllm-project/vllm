# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter
from typing import Any

import msgspec
import pytest

from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored

# Minimal ExternalBlockHash for testing (bytes are a valid ExternalBlockHash).
_FAKE_HASH: bytes = b"\xab" * 32


class _LegacyAllBlocksCleared(
    msgspec.Struct,
    tag="AllBlocksCleared",  # type: ignore[call-arg]
):
    pass


class _LegacyBlockStored(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag="BlockStored",  # type: ignore[call-arg]
):
    """BlockStored wire schema before locality was added."""

    block_hashes: list[bytes]
    parent_block_hash: bytes | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None


class _LegacyBlockRemoved(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag="BlockRemoved",  # type: ignore[call-arg]
):
    """BlockRemoved wire schema before locality was added."""

    block_hashes: list[bytes]
    medium: str | None
    group_idx: int | None = None


class _PreSessionBlockStored(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag="BlockStored",  # type: ignore[call-arg]
):
    """BlockStored wire schema before session_id was added."""

    block_hashes: list[bytes]
    parent_block_hash: bytes | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None


def _make_block_stored(
    group_idx: int | None = None,
    kv_cache_spec_sliding_window: int | None = None,
    locality: str | None = None,
    session_id: str | None = None,
) -> BlockStored:
    return BlockStored(
        block_hashes=[_FAKE_HASH],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
        lora_id=None,
        medium="GPU",
        lora_name=None,
        group_idx=group_idx,
        kv_cache_spec_sliding_window=kv_cache_spec_sliding_window,
        locality=locality,
        session_id=session_id,
    )


def _make_block_removed(
    group_idx: int | None = None,
    locality: str | None = None,
) -> BlockRemoved:
    return BlockRemoved(
        block_hashes=[_FAKE_HASH],
        medium="GPU",
        group_idx=group_idx,
        locality=locality,
    )


def test_block_stored_default_group_idx_is_none():
    """group_idx defaults to None when not provided."""
    event = _make_block_stored()
    assert event.group_idx is None


def test_block_removed_default_group_idx_is_none():
    """group_idx defaults to None when not provided."""
    event = _make_block_removed()
    assert event.group_idx is None


@pytest.mark.parametrize("group_idx", [1, 2, 3])
def test_block_stored_hash_differs_by_group_idx(group_idx: int):
    """BlockStored events that differ only in group_idx must hash differently."""
    other_group_idx = group_idx + 1
    event_a = _make_block_stored(group_idx=group_idx)
    event_b = _make_block_stored(group_idx=other_group_idx)
    assert hash(event_a) != hash(event_b)


def test_block_stored_hash_same_for_equal_group_idx():
    """Two BlockStored events with identical fields produce the same hash."""
    event_a = _make_block_stored(group_idx=1)
    event_b = _make_block_stored(group_idx=1)
    assert hash(event_a) == hash(event_b)


def test_block_stored_hash_differs_by_session_id():
    event_a = _make_block_stored(session_id="session-a")
    event_b = _make_block_stored(session_id="session-b")
    assert hash(event_a) != hash(event_b)


@pytest.mark.parametrize("group_idx", [1, 2, 3])
def test_block_removed_hash_differs_by_group_idx(group_idx: int):
    """BlockRemoved events that differ only in group_idx must hash differently."""
    other_group_idx = group_idx + 1
    event_a = _make_block_removed(group_idx=group_idx)
    event_b = _make_block_removed(group_idx=other_group_idx)
    assert hash(event_a) != hash(event_b)


def test_block_removed_hash_same_for_equal_group_idx():
    """Two BlockRemoved events with identical fields produce the same hash."""
    event_a = _make_block_removed(group_idx=1)
    event_b = _make_block_removed(group_idx=1)
    assert hash(event_a) == hash(event_b)


def test_block_stored_hash_differs_by_sliding_window():
    event_a = _make_block_stored(group_idx=1, kv_cache_spec_sliding_window=128)
    event_b = _make_block_stored(group_idx=1, kv_cache_spec_sliding_window=256)
    assert hash(event_a) != hash(event_b)


@pytest.mark.parametrize(
    ("event_a", "event_b"),
    [
        (
            _make_block_stored(locality="LOCAL"),
            _make_block_stored(locality="REMOTE"),
        ),
        (
            _make_block_removed(locality="LOCAL"),
            _make_block_removed(locality="REMOTE"),
        ),
    ],
)
def test_event_hash_differs_by_locality(
    event_a: BlockStored | BlockRemoved,
    event_b: BlockStored | BlockRemoved,
):
    assert hash(event_a) != hash(event_b)


def test_block_stored_locality_is_wire_compatible():
    legacy = _LegacyBlockStored(
        block_hashes=[_FAKE_HASH],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
        lora_id=None,
        medium="GPU",
        lora_name=None,
        group_idx=2,
        kv_cache_spec_sliding_window=128,
    )
    legacy_payload = msgspec.msgpack.encode(legacy)
    # More fields can widen the MessagePack map header, not change its contents.
    assert msgspec.msgpack.decode(
        msgspec.msgpack.encode(
            _make_block_stored(
                group_idx=2,
                kv_cache_spec_sliding_window=128,
            )
        )
    ) == msgspec.msgpack.decode(legacy_payload)
    assert msgspec.msgpack.decode(legacy_payload, type=BlockStored).locality is None
    new_payload = msgspec.msgpack.encode(_make_block_stored(locality="LOCAL"))
    assert msgspec.msgpack.decode(new_payload)["locality"] == "LOCAL"
    assert msgspec.msgpack.decode(new_payload, type=_LegacyBlockStored).medium == "GPU"


def test_block_removed_locality_is_wire_compatible():
    legacy = _LegacyBlockRemoved(block_hashes=[_FAKE_HASH], medium="GPU")
    legacy_payload = msgspec.msgpack.encode(legacy)
    assert msgspec.msgpack.encode(_make_block_removed()) == legacy_payload
    assert msgspec.msgpack.decode(legacy_payload, type=BlockRemoved).locality is None
    new_payload = msgspec.msgpack.encode(_make_block_removed(locality="REMOTE"))
    assert msgspec.msgpack.decode(new_payload)["locality"] == "REMOTE"
    assert msgspec.msgpack.decode(new_payload, type=_LegacyBlockRemoved).medium == "GPU"


def test_block_stored_session_id_is_wire_compatible():
    pre_session = _PreSessionBlockStored(
        block_hashes=[_FAKE_HASH],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
        lora_id=None,
        medium="GPU",
        lora_name=None,
        group_idx=2,
        kv_cache_spec_sliding_window=128,
        locality="LOCAL",
    )
    pre_session_payload = msgspec.msgpack.encode(pre_session)
    assert msgspec.msgpack.decode(
        msgspec.msgpack.encode(
            _make_block_stored(
                group_idx=2,
                kv_cache_spec_sliding_window=128,
                locality="LOCAL",
            )
        )
    ) == msgspec.msgpack.decode(pre_session_payload)
    assert (
        msgspec.msgpack.decode(pre_session_payload, type=BlockStored).session_id is None
    )

    new_payload = msgspec.msgpack.encode(_make_block_stored(session_id="session-1"))
    assert msgspec.msgpack.decode(new_payload)["session_id"] == "session-1"
    assert (
        msgspec.msgpack.decode(new_payload, type=_PreSessionBlockStored).medium == "GPU"
    )


@pytest.mark.parametrize(
    ("make_event", "legacy_type"),
    [
        (_make_block_stored, _LegacyBlockStored),
        (_make_block_removed, _LegacyBlockRemoved),
        (AllBlocksCleared, _LegacyAllBlocksCleared),
    ],
)
def test_event_metadata_round_trip_and_legacy_decoder(make_event, legacy_type):
    """New metadata survives the wire while tolerant old consumers keep payloads."""
    event = make_event()
    original = msgspec.msgpack.encode(event)
    payload = msgspec.msgpack.decode(original)
    payload["source"] = {
        "source_id": "connector/1/0",
        "connector_name": "ConfiguredConnector",
    }
    if not isinstance(event, AllBlocksCleared):
        payload["sharing_scope"] = "shared"
    encoded = msgspec.msgpack.encode(payload)
    decoded = msgspec.msgpack.decode(encoded, type=type(event))
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(decoded)) == payload
    assert msgspec.msgpack.decode(encoded, type=legacy_type) == (
        msgspec.msgpack.decode(original, type=legacy_type)
    )


@pytest.mark.parametrize(
    "make_event", [_make_block_stored, _make_block_removed, AllBlocksCleared]
)
@pytest.mark.parametrize("explicit_null", [False, True])
def test_missing_event_metadata_stays_unknown(make_event, explicit_null):
    event = make_event()
    payload = msgspec.msgpack.decode(msgspec.msgpack.encode(event))
    assert "source" not in payload
    assert "sharing_scope" not in payload
    if explicit_null:
        payload["source"] = None
        if not isinstance(event, AllBlocksCleared):
            payload["sharing_scope"] = None
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(payload), type=type(event))
    assert decoded.source is None
    if isinstance(event, AllBlocksCleared):
        assert "sharing_scope" not in decoded.__struct_fields__
    else:
        assert decoded.sharing_scope is None
    assert decoded == event


@pytest.mark.parametrize(
    "make_event", [_make_block_stored, _make_block_removed, AllBlocksCleared]
)
def test_event_sources_remain_distinct_in_counters(make_event):
    """Equal payloads from separate configured producers must not collapse."""
    event = make_event()
    payload = msgspec.msgpack.decode(msgspec.msgpack.encode(event))
    events = []
    for source_id in ("connector/0", "connector/1", "connector/0"):
        payload["source"] = {
            "source_id": source_id,
            "connector_name": "SameConnector",
        }
        events.append(
            msgspec.msgpack.decode(msgspec.msgpack.encode(payload), type=type(event))
        )
    assert events[0] != events[1]
    assert events[0] == events[2]
    assert hash(events[0]) == hash(events[2])
    assert Counter(events) == {events[0]: 2, events[1]: 1}


@pytest.mark.parametrize("make_event", [_make_block_stored, _make_block_removed])
def test_event_scopes_remain_distinct_in_counters(make_event):
    event = make_event()
    payload = msgspec.msgpack.decode(msgspec.msgpack.encode(event))
    events = []
    for scope in (None, "instance", "shared", "shared"):
        payload["sharing_scope"] = scope
        events.append(
            msgspec.msgpack.decode(msgspec.msgpack.encode(payload), type=type(event))
        )
    assert len(Counter(events)) == 3
    assert Counter(events)[events[-1]] == 2


@pytest.mark.parametrize("source", [{}, {"source_id": ""}, {"source_id": 1}])
def test_invalid_event_source_rejected_on_decode(source):
    payload = {"type": "AllBlocksCleared", "source": source}
    with pytest.raises(msgspec.ValidationError):
        msgspec.msgpack.decode(msgspec.msgpack.encode(payload), type=AllBlocksCleared)


@pytest.mark.parametrize("scope", ["local", "", "SHARED", True, 1])
def test_invalid_event_scope_rejected_on_decode(scope):
    payload = msgspec.msgpack.decode(msgspec.msgpack.encode(_make_block_removed()))
    payload["sharing_scope"] = scope
    with pytest.raises(msgspec.ValidationError):
        msgspec.msgpack.decode(msgspec.msgpack.encode(payload), type=BlockRemoved)
