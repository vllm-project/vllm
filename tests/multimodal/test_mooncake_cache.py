# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Mooncake-store backend of the multi-modal processor cache.

The Mooncake client is replaced by an in-memory stand-in that implements the
handful of methods the backend uses, so these run without a cluster.
"""

import pytest
import torch

from vllm.multimodal.cache import MultiModalCacheMissError
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
)
from vllm.multimodal.mooncake_cache import (
    MooncakeProcessorCacheOptions,
    MooncakeProcessorReceiverCache,
    MooncakeProcessorSenderCache,
    MooncakeProcessorStore,
    _decode_prompt_updates,
    _encode_prompt_updates,
    _Unshareable,
)
from vllm.multimodal.processing import PromptUpdateDetails
from vllm.multimodal.processing.processor import (
    PromptIndexTargets,
    ResolvedPromptUpdate,
    UpdateMode,
)

pytestmark = pytest.mark.cpu_test


class FakeMooncakeStore:
    """In-memory stand-in for `MooncakeDistributedStore`."""

    def __init__(self):
        self.data: dict[str, bytes] = {}

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [1 if key in self.data else 0 for key in keys]

    def get_batch(self, keys: list[str]) -> list[bytes]:
        return [self.data.get(key, b"") for key in keys]

    def get(self, key: str) -> bytes:
        return self.data.get(key, b"")

    def put(self, key: str, value) -> int:
        # Mooncake's `put` declines to overwrite an existing key and still
        # reports success. Modelled so tests catch a regression to `put`.
        self.data.setdefault(key, bytes(value))
        return 0

    def put_parts(self, key: str, *parts) -> int:
        self.data.setdefault(key, b"".join(bytes(part) for part in parts))
        return 0

    def upsert(self, key: str, value) -> int:
        self.data[key] = bytes(value)
        return 0

    def upsert_parts(self, key: str, *parts) -> int:
        self.data[key] = b"".join(bytes(part) for part in parts)
        return 0

    def remove(self, key: str, force: bool = False) -> int:
        return 0 if self.data.pop(key, None) is not None else -1

    def close(self) -> None:
        pass


class _ModelConfigStub:
    """Just enough of `ModelConfig` for the caches to size themselves."""

    class _MMConfig:
        mm_processor_cache_gb = 1.0

    def get_multimodal_config(self):
        return self._MMConfig()


def _item(size_by_key: dict[str, int]) -> MultiModalKwargsItem:
    return MultiModalKwargsItem(
        {
            key: MultiModalFieldElem(
                data=torch.arange(size, dtype=torch.int8),
                field=MultiModalSharedField(batch_size=1),
            )
            for key, size in size_by_key.items()
        }
    )


def _updates(seq: list[int]) -> list[ResolvedPromptUpdate]:
    return [
        ResolvedPromptUpdate(
            modality="image",
            item_idx=0,
            mode=UpdateMode.REPLACE,
            target=[7, 8],
            content=PromptUpdateDetails.from_seq(seq),
        )
    ]


def _make_store(shadow_ttl_s: float = 30.0):
    backend = FakeMooncakeStore()
    store = MooncakeProcessorStore(
        store=backend,
        options=MooncakeProcessorCacheOptions(shadow_ttl_s=shadow_ttl_s),
    )
    return backend, store


def _sender(store, ttl_store=None):
    return MooncakeProcessorSenderCache(_ModelConfigStub(), store=store)


def _receiver(store):
    return MooncakeProcessorReceiverCache(_ModelConfigStub(), store=store)


def test_store_roundtrip_preserves_tensors():
    backend, store = _make_store()
    item = _item({"a": 8, "b": 16})

    store.put("h", item, [], item_size=24)
    store.flush()

    loaded = store.get_kwargs("h")
    assert loaded is not None
    assert loaded.keys() == item.keys()
    for key in item:
        assert torch.equal(loaded[key].data, item[key].data)


def test_store_reports_only_fully_written_items():
    backend, store = _make_store()
    store.put("h", _item({"a": 8}), [], item_size=8)
    store.flush()

    assert set(store.probe(["h", "missing"])) == {"h"}

    # A hash whose payload was evicted must not be reported, even though its
    # metadata is still around.
    del backend.data[store._kwargs_key("h")]
    assert store.probe(["h"]) == {}


def test_sender_publishes_on_miss_and_keeps_the_data():
    backend, store = _make_store()
    sender = _sender(store)
    item = _item({"a": 8})
    updates = _updates([1, 2, 3])

    assert sender.is_cached(["h"]) == [False]

    out_item, out_updates = sender.get_and_update_item((item, updates), "h")
    store.flush()

    # The cold path still ships the data over IPC rather than making P1 read
    # back what P0 just wrote.
    assert out_item is item
    assert out_updates is updates
    assert store._kwargs_key("h") in backend.data
    assert store._meta_key("h") in backend.data


def test_sender_reuses_an_item_published_by_another_process():
    backend, store = _make_store()
    updates = _updates([1, 2, 3])

    _sender(store).get_and_update_item((_item({"a": 8}), updates), "h")
    store.flush()

    # A second process starts with an empty shadow and must still see the hit.
    fresh = _sender(store)
    assert fresh.is_cached(["h"]) == [True]

    out_item, out_updates = fresh.get_and_update_item(None, "h")
    assert out_item is None
    assert [u.content.full for u in out_updates] == [[1, 2, 3]]


def test_sender_reverifies_a_stale_shadow_entry():
    backend, store = _make_store(shadow_ttl_s=0.0)
    sender = _sender(store)

    sender.get_and_update_item((_item({"a": 8}), []), "h")
    store.flush()
    assert sender.is_cached(["h"]) == [True]

    # Evicted remotely: with the TTL elapsed the shadow entry must be dropped
    # rather than promising P1 an item it cannot fetch.
    backend.data.clear()
    assert sender.is_cached(["h"]) == [False]
    assert sender.is_cached_item("h") is False


def test_sender_trusts_a_fresh_shadow_entry_without_the_store():
    backend, store = _make_store(shadow_ttl_s=1e6)
    sender = _sender(store)

    sender.get_and_update_item((_item({"a": 8}), []), "h")
    store.flush()
    assert sender.is_cached(["h"]) == [True]

    backend.data.clear()
    assert sender.is_cached(["h"]) == [True]

    sender.invalidate("h")
    assert sender.is_cached(["h"]) == [False]


def test_sender_skips_items_whose_prompt_updates_cannot_be_encoded():
    backend, store = _make_store()
    sender = _sender(store)

    opaque = [
        ResolvedPromptUpdate(
            modality="image",
            item_idx=0,
            mode=UpdateMode.REPLACE,
            target=[7],
            content=PromptUpdateDetails(full=[1], is_embed=lambda full: full),
        )
    ]
    sender.get_and_update_item((_item({"a": 8}), opaque), "h")
    store.flush()

    assert store._meta_key("h") not in backend.data
    assert _sender(store).is_cached(["h"]) == [False]


@pytest.mark.parametrize(
    "content",
    [
        PromptUpdateDetails.from_seq([1, 2, 3]),
        PromptUpdateDetails.select_token_id([1, 2, 3], 2),
        PromptUpdateDetails.select_token_ids([1, 2, 3], [2, 3]),
    ],
)
def test_prompt_updates_survive_the_wire_schema(content):
    """The store keeps prompt updates, so every built-in `is_embed` helper must
    round-trip through the encoding."""
    update = ResolvedPromptUpdate(
        modality="image",
        item_idx=0,
        mode=UpdateMode.REPLACE,
        target=[7, 8],
        content=content,
    )

    (loaded,) = _decode_prompt_updates(_encode_prompt_updates([update]))

    assert loaded.modality == update.modality
    assert loaded.item_idx == update.item_idx
    assert loaded.mode is update.mode
    assert loaded.target == update.target
    assert loaded.content.full == content.full

    if content.is_embed is None:
        assert loaded.content.is_embed is None
    else:
        assert torch.equal(
            loaded.content.is_embed(content.full),
            content.is_embed(content.full),
        )


def test_prompt_index_targets_are_not_shareable():
    """A `PromptIndex` target holds callables, so it cannot be published."""
    update = ResolvedPromptUpdate(
        modality="image",
        item_idx=0,
        mode=UpdateMode.REPLACE,
        target=PromptIndexTargets.start(),
        content=PromptUpdateDetails.from_seq([1]),
    )

    with pytest.raises(_Unshareable):
        _encode_prompt_updates([update])


def test_receiver_reads_through_to_the_store():
    backend, store = _make_store()
    item = _item({"a": 8})
    store.put("h", item, [], item_size=8)
    store.flush()

    receiver = _receiver(store)
    loaded = receiver.get_and_update_item(None, "h")
    assert torch.equal(loaded["a"].data, item["a"].data)

    # Served from the local cache once populated.
    backend.data.clear()
    assert receiver.get_and_update_item(None, "h") is loaded


def test_receiver_caches_the_cold_path_item():
    backend, store = _make_store()
    receiver = _receiver(store)
    item = _item({"a": 8})

    assert receiver.get_and_update_item(item, "h") is item
    assert receiver.get_and_update_item(None, "h") is item


def test_receiver_raises_a_retryable_error_on_drift():
    _, store = _make_store()
    receiver = _receiver(store)

    with pytest.raises(MultiModalCacheMissError) as exc_info:
        receiver.get_and_update_item(None, "h")

    assert exc_info.value.mm_hashes == ["h"]


def test_store_errors_degrade_to_a_miss():
    backend, store = _make_store()
    sender = _sender(store)

    def boom(keys):
        raise RuntimeError("store is down")

    backend.batch_is_exist = boom
    assert sender.is_cached(["h"]) == [False]


def test_a_damaged_metadata_object_does_not_become_permanent():
    """Regression: `put` silently declines to overwrite, so a damaged object
    used to fail every probe for as long as it lived. Publishing must repair
    it, and a failed decode must drop it."""
    backend, store = _make_store()
    item, updates = _item({"a": 8}), _updates([1, 2, 3])

    store.put("h", item, updates, item_size=8)
    store.flush()
    good = backend.data[store._meta_key("h")]

    # Truncate the metadata the way the observed failure looked.
    backend.data[store._meta_key("h")] = good[:2]

    sender = _sender(store)
    assert sender.is_cached(["h"]) == [False]  # reported as a miss, no raise
    store.flush()
    assert store._meta_key("h") not in backend.data  # and dropped, not left to rot

    # Republishing repairs it rather than no-op'ing.
    store.put("h", item, updates, item_size=8)
    store.flush()
    assert backend.data[store._meta_key("h")] == good
    assert _sender(store).is_cached(["h"]) == [True]


def test_receiver_drops_an_unretrievable_pair():
    """A key can exist while its data cannot be read. Publishing uses `put`,
    which declines to overwrite, so the receiver must remove the pair or every
    later request for that item fails the same way."""
    backend, store = _make_store()
    receiver = _receiver(store)

    # Key present, payload unreadable -- the state a failed write leaves behind.
    backend.data[store._kwargs_key("h")] = b"\x00\x01"

    with pytest.raises(MultiModalCacheMissError):
        receiver.get_and_update_item(None, "h")
    store.flush()
    assert store._kwargs_key("h") not in backend.data
    assert store._meta_key("h") not in backend.data

    # With the keys gone, a fresh publish repopulates it.
    item = _item({"a": 8})
    store.put("h", item, _updates([1]), item_size=8)
    store.flush()
    loaded = store.get_kwargs("h")
    assert loaded is not None and torch.equal(loaded["a"].data, item["a"].data)


def test_probe_tolerates_a_repeated_hash():
    """The same image can appear twice in one prompt, so `is_cached` may be
    handed the same hash twice. A repeated key inside a batch lookup comes back
    correct for one occurrence and shifted for the other, so the keys must be
    deduplicated before the batch call."""
    backend, store = _make_store()
    store.put("h", _item({"a": 8}), _updates([1, 2, 3]), item_size=8)
    store.flush()

    seen = []
    inner = backend.get_batch

    def recording_get_batch(keys):
        seen.append(list(keys))
        return inner(keys)

    backend.get_batch = recording_get_batch
    assert set(store.probe(["h", "h", "h"])) == {"h"}
    assert seen == [[store._meta_key("h")]], seen

    # And the sender reports every occurrence as cached.
    assert _sender(store).is_cached(["h", "h"]) == [True, True]
