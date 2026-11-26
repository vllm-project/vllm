# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage floors preserve backing blocks without extending attention reach."""

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_request
from tests.v1.core.utils import new_kv_cache_group
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test


def new_manager(num_blocks=32):
    spec = AttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
    )
    cache = KVCacheConfig(
        num_blocks,
        [],
        [
            new_kv_cache_group(
                ["layer"], SlidingWindowManager.layer_config(spec, sliding_window=32)
            )
        ],
        hash_block_size=16,
        prefix_cache_retention_interval=0,
    )
    init_none_hash(sha256)
    return KVCacheManager(
        cache, max_model_len=512, scheduler_block_size=16, hash_block_size=16
    )


def request_of(name):
    return make_request(name, list(range(256)), 16, sha256)


class FixedStorageRetention:
    def __init__(self, manager):
        self.manager = manager

    def prepare(self, request: Request, local_tokens: int, external_tokens: int):
        group = self.manager.coordinator.single_type_managers[0]
        created = request.request_id not in group.storage_retention_starts
        self.manager.set_storage_retention(request.request_id, [1])
        return created

    def forget(self, request_id: str):
        pass


def test_storage_floor_keeps_local_hits_and_external_blocks_until_cleared():
    manager = new_manager()
    warm = request_of("warm")
    assert manager.allocate_slots(warm, 256) is not None
    warm_ids = manager.get_block_ids(warm.request_id)[0]
    manager.free(warm)

    request = request_of("reuse")
    hits, local, _ = manager.get_computed_blocks(request)
    assert local == 240
    manager.set_storage_retention(request.request_id, [13])
    assert (
        manager.allocate_slots(
            request,
            1,
            num_new_computed_tokens=local,
            new_computed_blocks=hits,
            num_external_computed_tokens=15,
        )
        is not None
    )
    ids = manager.get_block_ids(request.request_id)[0]
    assert ids[:13] == [0] * 13
    assert ids[13:15] == warm_ids[13:15] and all(ids[13:])

    manager.remove_skipped_blocks(request.request_id, 256)
    assert manager.get_block_ids(request.request_id)[0][13] == warm_ids[13]
    manager.clear_storage_retention(request.request_id)
    manager.remove_skipped_blocks(request.request_id, 256)
    assert manager.get_block_ids(request.request_id)[0][13] == 0
    manager.free(request)


@pytest.mark.parametrize("full_sequence", [False, True])
@pytest.mark.parametrize("existing_floor", [False, True])
def test_failed_allocation_rolls_back_only_a_new_storage_floor(
    full_sequence, existing_floor
):
    manager = new_manager(num_blocks=15 if full_sequence else 10)
    request = request_of("retry")
    manager.storage_retention_provider = FixedStorageRetention(manager)
    if existing_floor:
        manager.set_storage_retention(request.request_id, [1])
    assert (
        manager.allocate_slots(
            request,
            1,
            num_external_computed_tokens=224,
            full_sequence_must_fit=full_sequence,
        )
        is None
    )
    manager.storage_retention_provider = None
    retried = manager.allocate_slots(
        request,
        1,
        num_external_computed_tokens=224,
        full_sequence_must_fit=full_sequence,
    )
    assert (retried is None) == existing_floor
    if existing_floor:
        manager.clear_storage_retention(request.request_id)
        assert (
            manager.allocate_slots(
                request,
                1,
                num_external_computed_tokens=224,
                full_sequence_must_fit=full_sequence,
            )
            is not None
        )
    assert manager.get_block_ids(request.request_id)[0][:12] == [0] * 12
    manager.free(request)


@pytest.mark.parametrize("pop", [False, True])
def test_free_and_preemption_clear_the_storage_floor(pop):
    manager = new_manager()
    request = request_of("lifetime")
    manager.set_storage_retention(request.request_id, [1])
    assert (
        manager.allocate_slots(request, 1, num_external_computed_tokens=224) is not None
    )
    assert manager.get_block_ids(request.request_id)[0][1] > 0
    if pop:
        blocks = manager.pop_blocks_for_free(request)
        manager.block_pool.free_blocks(reversed(blocks))
    else:
        manager.free(request)
    request.status = RequestStatus.PREEMPTED
    assert (
        manager.allocate_slots(request, 1, num_external_computed_tokens=224) is not None
    )
    assert manager.get_block_ids(request.request_id)[0][:12] == [0] * 12
    manager.free(request)
