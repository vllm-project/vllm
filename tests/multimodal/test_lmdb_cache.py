# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import multiprocessing
import os
import signal
import tempfile
import time

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
from vllm.multimodal.lmdb_cache import LmdbMultiModalCache, ensure_lmdb_env
from vllm.multimodal.processing import PromptInsertion
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes
from vllm.utils.system_utils import get_mp_context


def _dummy_elem(
    size: int,
    *,
    rng: np.random.RandomState | None = None,
):
    if rng is None:
        data = torch.empty((size,), dtype=torch.int8)
    else:
        data = torch.from_numpy(rng.randint(4, size=(size,), dtype=np.int8))

    return MultiModalFieldElem(
        data=data,
        field=MultiModalSharedField(batch_size=1),
    )


def _dummy_item(
    size_by_key: dict[str, int],
    *,
    rng: np.random.RandomState | None = None,
):
    return MultiModalKwargsItem(
        {key: _dummy_elem(size, rng=rng) for key, size in size_by_key.items()}
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


def test_ensure_lmdb_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        env_1 = ensure_lmdb_env(tmpdir)
        env_2 = ensure_lmdb_env(tmpdir)

        assert env_1 is env_2

        def _child():
            env_3 = ensure_lmdb_env(tmpdir)
            assert env_3 is not env_1

        p = multiprocessing.get_context("fork").Process(target=_child)
        p.start()
        p.join()
        assert p.exitcode == 0


def test_lmdb_insert_get_evict(lmdb_cache: LmdbMultiModalCache):
    sender_cache = LmdbObjectStoreSenderCache(lmdb_cache)

    MM_HASH = "fake_hash"
    ITEM_CHUNKS = 150.5
    dummy_item = _dummy_item({"key": int(lmdb_cache._max_chunk_size * ITEM_CHUNKS)})
    prompt_update = PromptInsertion("dummy", "target", "insertion")

    # Do two rounds to ensure inserting after evicting works correctly.
    for _ in range(2):
        with sender_cache.begin() as txn:
            assert not txn.is_cached_item(MM_HASH)
            should_be_none, update_1 = txn.get_and_update_item(
                (dummy_item, prompt_update), MM_HASH
            )
            assert should_be_none is None
            assert update_1 == prompt_update

        receiver_cache = LmdbObjectStoreWorkerReceiverCache(lmdb_cache)
        with receiver_cache.begin() as txn:
            retrieved_item = txn.get_and_update_item(None, MM_HASH)
            assert retrieved_item == dummy_item

        with sender_cache.begin() as txn:
            assert txn.is_cached_item(MM_HASH)
            should_be_none, update_2 = txn.get_and_update_item(None, MM_HASH)
            assert should_be_none is None
            assert update_2 == prompt_update

        evicted_items, _ = lmdb_cache.evict_once(min_utilization=0.0)
        assert evicted_items == math.ceil(ITEM_CHUNKS) + 1


def test_lmdb_eviction_with_transaction_open(lmdb_cache: LmdbMultiModalCache):
    sender_cache = LmdbObjectStoreSenderCache(lmdb_cache)

    MM_HASH = "fake_hash"
    ITEM_CHUNKS = 150.5
    dummy_item = _dummy_item({"key": int(lmdb_cache._max_chunk_size * ITEM_CHUNKS)})
    prompt_update = PromptInsertion("dummy", "target", "insertion")
    with sender_cache.begin() as txn:
        txn.get_and_update_item((dummy_item, prompt_update), MM_HASH)

    with sender_cache.begin() as txn:
        assert txn.is_cached_item(MM_HASH)
        evicted, _ = lmdb_cache.evict_once(min_utilization=0.0)
        assert evicted == math.ceil(ITEM_CHUNKS) + 1

        # The item should be cached since the transaction is still open.
        assert txn.is_cached_item(MM_HASH)

        # But a separate transaction should not see the item.
        with sender_cache.begin() as other_txn:
            assert not other_txn.is_cached_item(MM_HASH)

        should_be_none, update = txn.get_and_update_item(None, MM_HASH)
        assert should_be_none is None
        assert update == prompt_update

    # After the transaction commits, the item should be there.
    with sender_cache.begin() as txn:
        assert txn.is_cached_item(MM_HASH)
        should_be_none, update = txn.get_and_update_item(None, MM_HASH)
        assert should_be_none is None
        assert update == prompt_update

    # And the receiver cache should see the item as well.
    receiver_cache = LmdbObjectStoreWorkerReceiverCache(lmdb_cache)
    with receiver_cache.begin() as txn:
        retrieved_item = txn.get_and_update_item(None, MM_HASH)
        assert retrieved_item == dummy_item

        evicted, _ = lmdb_cache.evict_once(min_utilization=0.0)
        assert evicted == math.ceil(ITEM_CHUNKS) + 1

        # The item should be cached since the transaction is still open.
        retrieved_item = txn.get_and_update_item(None, MM_HASH)
        assert retrieved_item == dummy_item

    # But now it should be gone.
    with receiver_cache.begin() as txn, pytest.raises(ValueError):
        txn.get_and_update_item(None, MM_HASH)


def test_lmdb_concurrent_inserts(lmdb_cache: LmdbMultiModalCache):
    sender_cache = LmdbObjectStoreSenderCache(lmdb_cache)

    ITEM_CHUNKS = 150.5
    MM_HASH = "fake_hash"
    dummy_item = _dummy_item({"key": int(lmdb_cache._max_chunk_size * ITEM_CHUNKS)})
    prompt_update = PromptInsertion("dummy", "target", "insertion")
    with sender_cache.begin() as txn:
        assert not txn.is_cached_item(MM_HASH)
        item_1, update_1 = txn.get_and_update_item((dummy_item, prompt_update), MM_HASH)
        assert item_1 is None

        with sender_cache.begin() as other_txn:
            assert not other_txn.is_cached_item(MM_HASH)
            item2, update_2 = other_txn.get_and_update_item(
                (dummy_item, prompt_update), MM_HASH
            )
            assert item2 is None

    # Both transactions should return the same prompt update.
    assert update_1 == update_2

    # And the item should be present in a new transaction.
    with sender_cache.begin() as new_txn:
        assert new_txn.is_cached_item(MM_HASH)

    # And the receiver cache should see the item as well.
    receiver_cache = LmdbObjectStoreWorkerReceiverCache(lmdb_cache)
    with receiver_cache.begin() as txn:
        retrieved_item = txn.get_and_update_item(None, MM_HASH)
        assert retrieved_item == dummy_item

    evicted_items, _ = lmdb_cache.evict_once(min_utilization=0.0)
    assert evicted_items == math.ceil(ITEM_CHUNKS) + 1

    # Now it should be gone.
    with sender_cache.begin() as txn:
        assert not txn.is_cached_item(MM_HASH)

    evicted_items, _ = lmdb_cache.evict_once(min_utilization=0.0)
    assert evicted_items == 0


def test_lmdb_evictor_process(lmdb_cache: LmdbMultiModalCache):
    event = get_mp_context().Event()

    MM_HASH = "fake_hash"
    ITEM_CHUNKS = 150.5

    with lmdb_cache.begin_write() as txn:
        dummy_item = _dummy_item({"key": int(lmdb_cache._max_chunk_size * ITEM_CHUNKS)})

        prompt_update = PromptInsertion("dummy", "target", "insertion")
        txn.get_and_update_item((dummy_item, prompt_update), MM_HASH)

    with lmdb_cache.begin_read() as txn:
        assert txn.is_cached_item(MM_HASH)

    evictor_process = lmdb_cache.start_evictor(event)
    event.wait()
    os.kill(evictor_process.pid, signal.SIGUSR1)

    for _ in range(5):
        assert evictor_process.is_alive()
        with lmdb_cache.begin_read() as txn:
            if not txn.is_cached_item(MM_HASH):
                break

        time.sleep(0.1)
    else:
        raise AssertionError("Evictor process did not evict the item in time.")
