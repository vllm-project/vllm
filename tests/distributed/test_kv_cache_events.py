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
    KVEventBatch,
    LegacyExtraKey,
    LoRAKey,
    MultiModalKey,
    PromptEmbedsKey,
    extra_keys_to_typed,
)

# Pure msgspec unit tests: no GPU/distributed state to clean up.
pytestmark = pytest.mark.skip_global_cleanup

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
    """locality is optional and backward-decode compatible.

    The current schema (15 declared fields + tag) encodes with a ``map16``
    header while the pre-locality schema uses ``fixmap``, so payloads are not
    byte-identical. Compatibility is preserved at decode time.
    """
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
    assert msgspec.msgpack.decode(legacy_payload, type=BlockStored).locality is None
    no_locality_payload = msgspec.msgpack.encode(
        _make_block_stored(
            group_idx=2,
            kv_cache_spec_sliding_window=128,
        )
    )
    assert (
        msgspec.msgpack.decode(no_locality_payload, type=_LegacyBlockStored).medium
        == "GPU"
    )
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


def _make_block_stored_typed(
    extra_keys: list[tuple[ExtraKeyUnion, ...] | None] | None,
    locality: str | None = None,
) -> BlockStored:
    event = BlockStored(
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
    return event.to_typed() if extra_keys else event


def test_block_stored_typed_events_carry_event_version_on_wire():
    """Typed events (extra_keys set) publish event_version=1 in the payload.

    ``KVCacheEvent`` uses ``omit_defaults=True``, which drops fields left at
    their default. ``event_version`` therefore cannot default to 1; it is
    stored as ``None`` and stamped to 1 by ``to_typed()`` only when the event
    carries typed extra keys. Legacy events (extra_keys=None) keep the
    version absent from the payload.
    """
    # Typed event: event_version present and equal to 1 after decode.
    typed = _make_block_stored_typed(
        [(MultiModalKey(modality="image", hash="h1", block_offset=0),)]
    )
    batch = KVEventBatch(
        ts=0.0,
        events=[typed],
        data_parallel_rank=0,
    )
    decoded_batch = msgspec.msgpack.decode(
        msgspec.msgpack.encode(batch), type=KVEventBatch
    )
    decoded_typed = decoded_batch.events[0]
    assert isinstance(decoded_typed, BlockStored)
    assert decoded_typed.event_version == 1

    # Legacy event: no event_version on the wire; decodes to None.
    legacy = _make_block_stored_typed(None)
    decoded_batch = msgspec.msgpack.decode(
        msgspec.msgpack.encode(
            KVEventBatch(ts=0.0, events=[legacy], data_parallel_rank=0)
        ),
        type=KVEventBatch,
    )
    decoded_legacy = decoded_batch.events[0]
    assert isinstance(decoded_legacy, BlockStored)
    assert decoded_legacy.event_version is None

    # A legacy consumer (schema without event_version) decodes both payloads.
    legacy_payload = msgspec.msgpack.encode(legacy)
    assert (
        msgspec.msgpack.decode(legacy_payload, type=_LegacyBlockStored).medium == "GPU"
    )


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
    """Events without extra_keys keep backward-compatible decoding.

    The merged schema (15 fields + tag) encodes with a ``map16`` header, while
    the legacy 12-field schema uses ``fixmap``; the payloads are not
    byte-identical. Backward compatibility is preserved at decode time: a
    legacy consumer schema decodes the untyped payload with all fields intact.
    """
    payload = msgspec.msgpack.encode(_make_block_stored_typed(None, locality=None))
    legacy = msgspec.msgpack.decode(payload, type=_LegacyBlockStored)
    assert legacy.medium == "GPU"
    assert legacy.group_idx == 1
    assert legacy.block_hashes == [_FAKE_HASH]
    # Typed consumer sees no event_version on the untyped payload.
    assert msgspec.msgpack.decode(payload, type=BlockStored).event_version is None


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
    assert converted == [(MultiModalKey(modality="image", hash="abc", block_offset=5),)]


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


def test_extra_keys_to_typed_cache_salt_equals_lora_name():
    """Equal cache_salt/lora_name values keep the cache-salt entry.

    The producer emits the LoRA name first, then `request.cache_salt` (present
    on the first block only) on block 0. When both values collide, the second
    occurrence must still classify as CacheSaltKey so consumers can identify
    the salt entry, instead of consuming it as a duplicate LoRAKey.
    """
    request = _fake_request(lora_name="same", cache_salt="same")
    converted = extra_keys_to_typed([("same", "same")], request)
    assert converted == [
        (LoRAKey(name="same"), CacheSaltKey(salt="same")),
    ]


def test_extra_keys_to_typed_lora_consumed_once_across_blocks():
    """Each known request value is consumed only once per block sequence.

    A LoRA request repeats its name in the extra keys of every block; only
    the first block also carries `cache_salt`. If the LoRA name were matched
    against the salt position, or if consumption were not tracked, equal
    strings in later blocks could be misclassified.
    """
    request = _fake_request(lora_name="lora1", cache_salt="saltval")
    # Block 0: lora + salt, block 1: lora only.
    converted = extra_keys_to_typed([("lora1", "saltval"), ("lora1",)], request)
    assert converted == [
        (LoRAKey(name="lora1"), CacheSaltKey(salt="saltval")),
        (LoRAKey(name="lora1"),),
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


def test_block_stored_session_id_is_wire_compatible():
    """session_id is optional and backward-decode compatible.

    The merged schema (15 declared fields + tag) encodes with a ``map16``
    header while the pre-session schema uses ``fixmap``, so payloads are not
    byte-identical. Compatibility is preserved at decode time: events without
    ``session_id`` decode to ``None`` under the current schema, and pre-session
    consumers decode the new payload with their fields intact.
    """
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

    # Current schema decodes the pre-session payload, session_id defaulting
    # to None; a pre-session consumer decodes the current no-session payload.
    assert (
        msgspec.msgpack.decode(pre_session_payload, type=BlockStored).session_id is None
    )
    no_session_payload = msgspec.msgpack.encode(
        _make_block_stored(
            group_idx=2,
            kv_cache_spec_sliding_window=128,
            locality="LOCAL",
        )
    )
    assert (
        msgspec.msgpack.decode(no_session_payload, type=_PreSessionBlockStored).medium
        == "GPU"
    )

    new_payload = msgspec.msgpack.encode(_make_block_stored(session_id="session-1"))
    assert msgspec.msgpack.decode(new_payload)["session_id"] == "session-1"
    assert (
        msgspec.msgpack.decode(new_payload, type=_PreSessionBlockStored).medium == "GPU"
    )
