# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import multiprocessing
import os
import signal
import tempfile
import time
from unittest import mock

import numpy as np
import pytest
import torch

from vllm.multimodal.cache import (
    LmdbObjectStoreSenderCache,
    LmdbObjectStoreWorkerReceiverCache,
)
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
)
from vllm.multimodal.lmdb_cache import LmdbMultiModalCache
from vllm.multimodal.processing import PromptInsertion
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes


def _dummy_elem(
    modality: str,
    key: str,
    size: int,
    *,
    rng: np.random.RandomState | None = None,
):
    if rng is None:
        data = torch.empty((size,), dtype=torch.int8)
    else:
        data = torch.from_numpy(rng.randint(4, size=(size,), dtype=np.int8))

    return MultiModalFieldElem(
        modality=modality,
        key=key,
        data=data,
        field=MultiModalSharedField(batch_size=1),
    )


def _dummy_item(
    modality: str,
    size_by_key: dict[str, int],
    *,
    rng: np.random.RandomState | None = None,
):
    return MultiModalKwargsItem.from_elems(
        [_dummy_elem(modality, key, size, rng=rng) for key, size in size_by_key.items()]
    )


@pytest.fixture()
def lmdb_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield LmdbMultiModalCache(
            tmpdir,
            cache_size=GiB_bytes,
            min_eviction_age=-1,
            max_object_size=10 * MiB_bytes,
        )


def test_lmdb_insert_get_evict(lmdb_cache: LmdbMultiModalCache):
    sender_cache = LmdbObjectStoreSenderCache(lmdb_cache)

    ITEM_CHUNKS = 150.5
    dummy_item = _dummy_item(
        "modality", {"key": int(lmdb_cache._max_chunk_size * ITEM_CHUNKS)}
    )
    prompt_update = PromptInsertion("dummy", "target", "insertion")

    # Do two rounds to ensure inserting after evicting works correctly.
    for _ in range(2):
        with sender_cache.begin() as txn:
            assert not txn.is_cached_item("fake_hash")
            item_1, update_1 = txn.get_and_update_item(
                "modality", (dummy_item, prompt_update), "fake_hash"
            )
            assert update_1 == prompt_update

        receiver_cache = LmdbObjectStoreWorkerReceiverCache(lmdb_cache)
        with receiver_cache.begin() as txn:
            retrieved_item = txn.get_and_update_item("modality", item_1, "fake_hash")
            assert retrieved_item == dummy_item

        with sender_cache.begin() as txn:
            assert txn.is_cached_item("fake_hash")
            item_2, update_2 = txn.get_and_update_item("modality", None, "fake_hash")
            assert item_1 == item_2
            assert update_2 == prompt_update

        evicted_items, _ = lmdb_cache.evict_once(min_utilization=0.0)
        assert evicted_items == math.ceil(ITEM_CHUNKS) + 1


def test_lmdb_eviction_with_transaction_open(lmdb_cache: LmdbMultiModalCache):
    sender_cache = LmdbObjectStoreSenderCache(lmdb_cache)

    ITEM_CHUNKS = 150.5
    dummy_item = _dummy_item(
        "modality", {"key": int(lmdb_cache._max_chunk_size * ITEM_CHUNKS)}
    )
    prompt_update = PromptInsertion("dummy", "target", "insertion")
    with sender_cache.begin() as txn:
        txn.get_and_update_item("modality", (dummy_item, prompt_update), "fake_hash")

    with sender_cache.begin() as txn:
        assert txn.is_cached_item("fake_hash")
        evicted, _ = lmdb_cache.evict_once(min_utilization=0.0)
        assert evicted == math.ceil(ITEM_CHUNKS) + 1

        # The item should be cached since the transaction is still open.
        assert txn.is_cached_item("fake_hash")

        # But a separate transaction should not see the item.
        with sender_cache.begin() as other_txn:
            assert not other_txn.is_cached_item("fake_hash")

        hash_item, update = txn.get_and_update_item("modality", None, "fake_hash")
        assert update == prompt_update

    # After the transaction commits, the item should be there.
    with sender_cache.begin() as txn:
        assert txn.is_cached_item("fake_hash")
        new_hash_item, update = txn.get_and_update_item("modality", None, "fake_hash")
        assert update == prompt_update
        assert new_hash_item == hash_item

    # And the receiver cache should see the item as well.
    receiver_cache = LmdbObjectStoreWorkerReceiverCache(lmdb_cache)
    with receiver_cache.begin() as txn:
        retrieved_item = txn.get_and_update_item("modality", new_hash_item, "fake_hash")
        assert retrieved_item == dummy_item

        evicted, _ = lmdb_cache.evict_once(min_utilization=0.0)
        assert evicted == math.ceil(ITEM_CHUNKS) + 1

        # The item should be cached since the transaction is still open.
        retrieved_item = txn.get_and_update_item("modality", new_hash_item, "fake_hash")
        assert retrieved_item == dummy_item

    # But now it should be gone.
    with receiver_cache.begin() as txn, pytest.raises(ValueError):
        txn.get_and_update_item("modality", new_hash_item, "fake_hash")


def test_lmdb_concurrent_inserts(lmdb_cache: LmdbMultiModalCache):
    sender_cache = LmdbObjectStoreSenderCache(lmdb_cache)

    ITEM_CHUNKS = 150.5
    dummy_item = _dummy_item(
        "modality", {"key": int(lmdb_cache._max_chunk_size * ITEM_CHUNKS)}
    )
    prompt_update = PromptInsertion("dummy", "target", "insertion")
    with sender_cache.begin() as txn:
        assert not txn.is_cached_item("fake_hash")
        item_1, update_1 = txn.get_and_update_item(
            "modality", (dummy_item, prompt_update), "fake_hash"
        )

        with sender_cache.begin() as other_txn:
            assert not other_txn.is_cached_item("fake_hash")
            item2, update_2 = other_txn.get_and_update_item(
                "modality", (dummy_item, prompt_update), "fake_hash"
            )

    # Both transactions should return the same item and update
    assert item_1 == item2
    assert update_1 == update_2

    # And the item should be present in a new transaction.
    with sender_cache.begin() as new_txn:
        assert new_txn.is_cached_item("fake_hash")

    # And the receiver cache should see the item as well.
    receiver_cache = LmdbObjectStoreWorkerReceiverCache(lmdb_cache)
    with receiver_cache.begin() as txn:
        retrieved_item = txn.get_and_update_item("modality", item_1, "fake_hash")
        assert retrieved_item == dummy_item

    evicted_items, _ = lmdb_cache.evict_once(min_utilization=0.0)
    assert evicted_items == math.ceil(ITEM_CHUNKS) + 1

    # Now it should be gone.
    with sender_cache.begin() as txn:
        assert not txn.is_cached_item("fake_hash")

    evicted_items, _ = lmdb_cache.evict_once(min_utilization=0.0)
    assert evicted_items == 0


# On macOS, set_process_title crashes in a forked subprocess.
@mock.patch("vllm.envs.VLLM_WORKER_MULTIPROC_METHOD", "spawn")
def test_lmdb_evictor_process(lmdb_cache: LmdbMultiModalCache):
    event = multiprocessing.Event()

    with lmdb_cache.begin_write() as txn:
        ITEM_CHUNKS = 150.5
        dummy_item = _dummy_item(
            "modality", {"key": int(lmdb_cache._max_chunk_size * ITEM_CHUNKS)}
        )

        prompt_update = PromptInsertion("dummy", "target", "insertion")
        txn.get_and_update_item("modality", (dummy_item, prompt_update), "test_hash")

    with lmdb_cache.begin_read() as txn:
        assert txn.is_cached_item("test_hash")

    evictor_process = lmdb_cache.start_evictor(event)
    event.wait()
    os.kill(evictor_process.pid, signal.SIGUSR1)

    for _ in range(5):
        assert evictor_process.is_alive()
        with lmdb_cache.begin_read() as txn:
            if not txn.is_cached_item("test_hash"):
                break

        time.sleep(0.1)
    else:
        raise AssertionError("Evictor process did not evict the item in time.")
