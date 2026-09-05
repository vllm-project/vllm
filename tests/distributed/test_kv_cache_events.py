# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import msgspec
import pytest

from vllm.distributed.kv_events import (
    BlockRemoved,
    BlockStored,
    CacheSaltKey,
    ExtraKeyUnion,
    LegacyExtraKey,
    LoRAKey,
    MultiModalKey,
    PromptEmbedsKey,
    extra_keys_to_typed,
)

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


class _TypedBlockStored(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag="BlockStored",  # type: ignore[call-arg]
):
    """BlockStored wire schema with typed ``ExtraKey`` entries (event_version=1)."""

    block_hashes: list[bytes]
    parent_block_hash: bytes | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    event_version: int = 1
    extra_keys: list[tuple[ExtraKeyUnion, ...] | None] | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None
    ownership: str | None = None


def _make_block_stored_typed(
    extra_keys: list[tuple[ExtraKeyUnion, ...] | None] | None,
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
        extra_keys=extra_keys,
        group_idx=1,
        locality=locality,
    )


def test_block_stored_event_version_defaults_to_one():
    """The typed extra_keys schema is event_version=1."""
    assert _make_block_stored_typed(None).event_version == 1


def test_block_stored_typed_extra_keys_roundtrip():
    """Typed extra_keys survive msgpack round-trips with their concrete types."""
    extra_keys = [
        (
            MultiModalKey(modality="image", hash="abc123", block_offset=5),
            LoRAKey(name="lora1"),
        ),
        (CacheSaltKey(salt="saltval"),),
        (PromptEmbedsKey(hash=b"\x01\x02\x03"),),
        (LegacyExtraKey(value=("unknown", "shape")),),
    ]
    event = _make_block_stored_typed(extra_keys)
    payload = msgspec.msgpack.encode(event)
    decoded = msgspec.msgpack.decode(payload, type=BlockStored)
    assert decoded.extra_keys == extra_keys
    assert all(
        type(key) is type(orig)
        for keys, orig_keys in zip(decoded.extra_keys, extra_keys)
        for key, orig in zip(keys, orig_keys)
    )


def test_block_stored_typed_extra_keys_decode_via_union():
    """Entries decode to concrete types via the explicit ExtraKeyUnion."""
    payload = msgspec.msgpack.encode(
        MultiModalKey(modality="image", hash="h1", block_offset=0)
    )
    decoded = msgspec.msgpack.decode(payload, type=ExtraKeyUnion)
    assert isinstance(decoded, MultiModalKey)


def test_block_stored_typed_extra_keys_hash_stable_after_roundtrip():
    """BlockStored hash is stable across msgpack round-trips (aggregator use)."""
    event = _make_block_stored_typed(
        [(MultiModalKey(modality="image", hash="h1", block_offset=0),)]
    )
    payload = msgspec.msgpack.encode(event)
    decoded = msgspec.msgpack.decode(payload, type=BlockStored)
    assert hash(event) == hash(decoded)


def test_block_stored_typed_vs_untyped_payload_differs_for_mm():
    """MM events carry the typed schema; the payload is no longer raw tuples."""
    typed = _make_block_stored_typed(
        [(MultiModalKey(modality="image", hash="h1", block_offset=0),)]
    )
    untyped = _make_block_stored_typed(None)
    assert msgspec.msgpack.encode(typed) != msgspec.msgpack.encode(untyped)


def test_block_stored_untyped_payload_matches_legacy_wire():
    """Events without extra_keys keep the legacy wire shape (backward compat)."""
    legacy = _LegacyBlockStored(
        block_hashes=[_FAKE_HASH],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
        lora_id=None,
        medium="GPU",
        lora_name=None,
        group_idx=1,
    )
    assert (
        msgspec.msgpack.encode(_make_block_stored_typed(None, locality=None))
        == msgspec.msgpack.encode(legacy)
    )


def test_block_stored_typed_extra_keys_distinct_hashes():
    """Different extra_keys produce different event hashes."""
    event_a = _make_block_stored_typed(
        [(MultiModalKey(modality="image", hash="h1", block_offset=0),)]
    )
    event_b = _make_block_stored_typed(
        [(MultiModalKey(modality="image", hash="h2", block_offset=0),)]
    )
    assert hash(event_a) != hash(event_b)


class _FakeFeature:
    def __init__(self, identifier: str, modality: str, mm_hash: str | None):
        self.identifier = identifier
        self.modality = modality
        self.mm_hash = mm_hash


class _FakeLora:
    def __init__(self, name: str):
        self.name = name


def _fake_request(
    mm_features: list[_FakeFeature] | None = None,
    lora_name: str | None = None,
    cache_salt: str | None = None,
) -> Any:
    return SimpleNamespace(
        mm_features=mm_features or [],
        lora_request=_FakeLora(lora_name) if lora_name else None,
        cache_salt=cache_salt,
    )


def test_extra_keys_to_typed_none_passthrough():
    assert extra_keys_to_typed(None, _fake_request()) is None


def test_extra_keys_to_typed_all_none_blocks():
    converted = extra_keys_to_typed([None, None], _fake_request())
    assert converted == [None, None]


def test_extra_keys_to_typed_mm_key_with_modality():
    request = _fake_request(
        mm_features=[_FakeFeature(identifier="abc", modality="image", mm_hash="abc")]
    )
    converted = extra_keys_to_typed([(("abc", 5),)], request)
    assert converted == [
        (MultiModalKey(modality="image", hash="abc", block_offset=5),)
    ]


def test_extra_keys_to_typed_mm_key_unprefixed_hash():
    """Identifier prefixed by LoRA resolves to the unprefixed mm_hash."""
    request = _fake_request(
        mm_features=[
            _FakeFeature(identifier="lora1:abc", modality="audio", mm_hash="abc")
        ]
    )
    converted = extra_keys_to_typed([(("lora1:abc", 3),)], request)
    key = converted[0][0]
    assert isinstance(key, MultiModalKey)
    assert key.modality == "audio"
    assert key.hash == "abc"
    assert key.block_offset == 3


def test_extra_keys_to_typed_lora_and_cache_salt():
    request = _fake_request(lora_name="lora1", cache_salt="saltval")
    converted = extra_keys_to_typed([("lora1", "saltval")], request)
    assert converted == [
        (LoRAKey(name="lora1"), CacheSaltKey(salt="saltval")),
    ]


def test_extra_keys_to_typed_prompt_embeds():
    converted = extra_keys_to_typed([(b"\x01\x02\x03",)], _fake_request())
    assert converted == [(PromptEmbedsKey(hash=b"\x01\x02\x03"),)]


def test_extra_keys_to_typed_unknown_shape_wrapped_legacy():
    converted = extra_keys_to_typed([("unknown",)], _fake_request())
    assert converted == [(LegacyExtraKey(value="unknown"),)]


def test_extra_keys_to_typed_unknown_pair_wrapped_legacy():
    """An unrecognised 2-tuple key is preserved as a single LegacyExtraKey."""
    converted = extra_keys_to_typed([(("future", "key"),)], _fake_request())
    assert converted == [(LegacyExtraKey(value=("future", "key")),)]
