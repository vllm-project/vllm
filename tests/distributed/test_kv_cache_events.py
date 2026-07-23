# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import msgspec
import pytest

from vllm.distributed.kv_events import BlockRemoved, BlockStored

# Minimal ExternalBlockHash for testing (bytes are a valid ExternalBlockHash).
_FAKE_HASH: bytes = b"\xab" * 32


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


def _make_block_stored(
    group_idx: int | None = None,
    kv_cache_spec_sliding_window: int | None = None,
    locality: str | None = None,
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
    assert (
        msgspec.msgpack.encode(
            _make_block_stored(
                group_idx=2,
                kv_cache_spec_sliding_window=128,
            )
        )
        == legacy_payload
    )
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
