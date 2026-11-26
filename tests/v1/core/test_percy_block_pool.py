# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percy A5: BlockPool changes block ownership only through its GrantTable."""

import ast
import inspect
import pathlib
import re
import textwrap

import pytest
import torch

import vllm.v1.core.kv_cache_manager as kv_cache_manager_module
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.grant_table import BlockState, GrantError
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.request import Request

from .utils import new_kv_cache_group, percy_ack_planned_steps  # noqa: F401

BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _request(request_id: str, num_tokens: int, cache_salt: str | None = None):
    params = SamplingParams(max_tokens=4)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=params,
        pooling_params=None,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _manager(num_blocks: int = 16) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            new_kv_cache_group(
                ["layer"],
                AttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=1024,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
        scheduler_block_size=BLOCK_SIZE,
    )


def _pool(num_blocks: int = 8) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=BLOCK_SIZE
    )


def _publish(pool: BlockPool, block, tag: bytes = b"h") -> None:
    pool.grant_table.confirm_valid([block.block_id], BLOCK_SIZE)
    assert pool.grant_table.publish([block.block_id])
    pool._insert_block_hash(
        make_block_hash_with_group_id(tag, 0), block, num_tokens=BLOCK_SIZE
    )


def _assert_ref_cnt_mirrors_readers(pool: BlockPool) -> None:
    readers = pool.grant_table.readers_of([b.block_id for b in pool.blocks[1:]])
    assert [b.ref_cnt for b in pool.blocks[1:]] == readers.tolist()


_QUEUE_WRITERS = ("append", "append_n", "prepend_n", "insert_before_cached")


def test_only_the_drain_path_feeds_the_free_queue():
    """Structural: no code outside BlockPool touches the free queue's writers,
    and inside it only the drain path (plus LRU re-ordering) does."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(BlockPool)))
    writers: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in _QUEUE_WRITERS
                and "free_block_queue" in ast.unparse(node.value)
            ):
                writers.add(fn.name)
    # _queue_drained: drained (FREE) and released cached-idle blocks;
    # make_room: re-queues the blocks it just drained at the head;
    # deprioritize_blocks: re-orders blocks already in the queue.
    assert writers == {"_queue_drained", "make_room", "deprioritize_blocks"}

    root = pathlib.Path(kv_cache_manager_module.__file__).parents[2]
    pattern = re.compile(r"free_block_queue\.(" + "|".join(_QUEUE_WRITERS) + r")\(")
    offenders = [
        str(path)
        for path in root.rglob("*.py")
        if path.name != "block_pool.py" and pattern.search(path.read_text())
    ]
    assert offenders == []


def test_freed_block_reaches_queue_only_after_drain():
    pool = _pool()
    [block] = pool.get_new_blocks(1, owner=1)
    num_free = pool.get_num_free_blocks()
    pool.grant_table.begin_step(5)
    pool.free_blocks([block], step_seq=5)  # a step up to 5 may still use it
    assert pool.grant_table.state(block.block_id) == BlockState.DRAINING
    assert block.prev_free_block is None
    assert pool.get_num_free_blocks() == num_free
    assert pool.drain(4) == 0
    assert pool.drain(5) == 1
    assert pool.grant_table.state(block.block_id) == BlockState.FREE
    assert pool.free_block_queue.get_all_free_blocks()[0] is block
    _assert_ref_cnt_mirrors_readers(pool)


def test_cached_block_is_never_regranted_in_the_evicting_call():
    pool = _pool(num_blocks=3)  # blocks 1 and 2 are usable
    a, b = pool.get_new_blocks(2, owner=1)
    _publish(pool, a, b"a")
    pool.free_blocks([b, a])  # b drains to the head, a stays cached-idle
    pool.make_room(1)
    [b2] = pool.get_new_blocks(1, owner=2)
    assert b2 is b
    # a now leads the queue: get_new_blocks refuses instead of evicting.
    gen = pool.grant_table.gen(a.block_id)
    with pytest.raises(ValueError, match="make_room"):
        pool.get_new_blocks(1, owner=2)
    assert pool.grant_table.state(a.block_id) == BlockState.CACHED_IDLE
    assert a.block_hash is not None and a.prev_free_block is not None
    # The separate eviction call revokes (gen + 1) and drains it.
    assert pool.make_room(1)
    assert a.block_hash is None
    assert pool.grant_table.state(a.block_id) == BlockState.FREE
    assert pool.grant_table.gen(a.block_id) == gen + 1
    [a2] = pool.get_new_blocks(1, owner=2)
    assert a2 is a
    _assert_ref_cnt_mirrors_readers(pool)


def test_make_room_skips_cached_blocks_a_step_may_still_read():
    pool = _pool(num_blocks=4)
    a, b, c = pool.get_new_blocks(3, owner=1)
    _publish(pool, a, b"a")
    pool.grant_table.begin_step(2)
    pool.free_blocks([a], step_seq=2)  # cached, but step 2 may still read it
    pool.free_blocks([b, c], step_seq=0)
    pool.drain(1)
    assert pool.make_room(2)
    assert pool.grant_table.state(a.block_id) == BlockState.CACHED_IDLE
    granted = pool.get_new_blocks(2, owner=2)
    assert {x.block_id for x in granted} == {b.block_id, c.block_id}
    assert not pool.make_room(1)  # only the in-flight cached block is left
    pool.drain(2)
    assert pool.make_room(1)
    assert pool.get_new_blocks(1, owner=2) == [a]


def test_cache_full_blocks_publishes_only_confirmed_blocks():
    pool = _pool()
    req = _request("r", 2 * BLOCK_SIZE)
    owner = pool.grant_table.owner_for("r")
    blocks = pool.get_new_blocks(2, owner=owner)
    pool.grant_table.confirm_valid([blocks[0].block_id], BLOCK_SIZE)
    pool.grant_table.confirm_valid([blocks[1].block_id], BLOCK_SIZE - 1)
    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=BLOCK_SIZE,
        kv_cache_group_id=0,
    )
    assert blocks[0].block_hash is not None
    assert pool.grant_table.state(blocks[0].block_id) == BlockState.SHARED
    assert blocks[1].block_hash is None  # I3: valid < block_size
    assert pool.grant_table.state(blocks[1].block_id) == BlockState.EXCLUSIVE
    assert pool.grant_table.check_writes(
        owner, [blocks[1].block_id], pool.grant_table.gens_of([blocks[1].block_id])
    )


@pytest.mark.usefixtures("percy_ack_planned_steps")
def test_domain_mismatch_is_a_miss(monkeypatch):
    manager = _manager()
    first = _request("first", 3 * BLOCK_SIZE)
    manager.allocate_slots(first, 3 * BLOCK_SIZE)
    manager.free(first)

    hit = _request("hit", 3 * BLOCK_SIZE)
    _, num_hit_tokens, _ = manager.get_computed_blocks(hit)
    assert num_hit_tokens == 2 * BLOCK_SIZE

    # Same hashes, foreign domain: the lookup treats the hit as a miss.
    monkeypatch.setattr(kv_cache_manager_module, "request_domain", lambda r: 7)
    other = _request("other", 3 * BLOCK_SIZE)
    blocks, num_hit_tokens, _ = manager.get_computed_blocks(other)
    assert num_hit_tokens == 0 and not any(blocks.blocks)


@pytest.mark.usefixtures("percy_ack_planned_steps")
def test_salted_requests_never_share_blocks():
    manager = _manager()
    salted = _request("salted", 3 * BLOCK_SIZE, cache_salt="tenant-a")
    manager.allocate_slots(salted, 3 * BLOCK_SIZE)
    manager.free(salted)
    plain = _request("plain", 3 * BLOCK_SIZE)
    assert manager.get_computed_blocks(plain)[1] == 0
    other_salt = _request("b", 3 * BLOCK_SIZE, cache_salt="tenant-b")
    assert manager.get_computed_blocks(other_salt)[1] == 0
    same_salt = _request("a2", 3 * BLOCK_SIZE, cache_salt="tenant-a")
    assert manager.get_computed_blocks(same_salt)[1] == 2 * BLOCK_SIZE


def test_touch_refuses_foreign_domain_without_mutation():
    pool = _pool()
    [block] = pool.get_new_blocks(1, owner=1, domain=3)
    _publish(pool, block)
    pool.free_blocks([block])
    with pytest.raises(GrantError):
        pool.touch([block], group=0, domain=4)
    assert block.ref_cnt == 0 and block.prev_free_block is not None
    pool.touch([block], group=0, domain=3)
    assert block.ref_cnt == 1
    _assert_ref_cnt_mirrors_readers(pool)


def test_partial_entry_is_sealed_only_over_confirmed_slots():
    # Hash granularity 2 inside 4-token blocks allows a partial entry at 2.
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=2)
    pool.grant_table.set_group_block_size(0, BLOCK_SIZE)
    params = SamplingParams(max_tokens=4)
    params.update_from_generation_config({}, eos_token_id=100)
    req = Request(
        request_id="r",
        prompt_token_ids=list(range(2 * BLOCK_SIZE)),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(2, sha256),
    )
    sealed, unconfirmed = pool.get_new_blocks(2, owner=1)
    pool.grant_table.confirm_valid([sealed.block_id], 2)
    assert pool.cache_partial_block(req, sealed, 2, 0, BLOCK_SIZE) is not None
    # Unconfirmed slots never get a published partial entry (Percy M2: the
    # entry is registered and waits for the slots' ack).
    assert pool.cache_partial_block(req, unconfirmed, 2, 0, BLOCK_SIZE) is not None
    assert unconfirmed.block_hash is None
    assert pool.is_publication_pending(unconfirmed.block_id)
    assert pool.grant_table.state(sealed.block_id) == BlockState.EXCLUSIVE
    pool.free_blocks([sealed, unconfirmed])
    assert not pool.is_publication_pending(unconfirmed.block_id)
    # The last holder left: the entry's slots are frozen and stay reachable.
    assert pool.grant_table.state(sealed.block_id) == BlockState.CACHED_IDLE
    assert sealed.block_hash is not None
    assert pool.grant_table.state(unconfirmed.block_id) == BlockState.FREE


@pytest.mark.usefixtures("percy_ack_planned_steps")
def test_ref_cnt_mirrors_readers_through_hits_and_pins():
    manager = _manager()
    pool = manager.block_pool
    reqs = [_request(f"r{i}", 3 * BLOCK_SIZE) for i in range(3)]
    for req in reqs:
        blocks, num_hit, _ = manager.get_computed_blocks(req)
        manager.allocate_slots(req, 3 * BLOCK_SIZE - num_hit, num_hit, blocks)
        _assert_ref_cnt_mirrors_readers(pool)
    held = manager.coordinator.single_type_managers[0].req_to_blocks["r0"][0]
    pool.touch([held])  # an internal pin (e.g. a connector read)
    _assert_ref_cnt_mirrors_readers(pool)
    for req in reqs:
        manager.free(req)
        _assert_ref_cnt_mirrors_readers(pool)
    pool.free_blocks([held])
    _assert_ref_cnt_mirrors_readers(pool)
    assert pool.grant_table.state(held.block_id) == BlockState.CACHED_IDLE
