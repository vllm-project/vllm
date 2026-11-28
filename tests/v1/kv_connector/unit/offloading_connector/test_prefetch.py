# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for prefetch hints: what to pin, and which key owns each chunk."""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.prefetch import (
    PREFETCH_ACTION_TYPE,
    PREFETCH_ACTION_VERSION,
    decode_prefetch_hashes,
    plan_prefetch,
    truncate_plan,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_offload.base import get_offload_block_hash, get_offload_group_idx

HASHES = tuple(bytes([i]) * 32 for i in range(20))


def envelope(*actions: KvHintAction) -> KvHintsEnvelope:
    return KvHintsEnvelope(
        protocol_version="0.1", message_id="m-1", actions=list(actions)
    )


def prefetch_action(hashes=HASHES, version=PREFETCH_ACTION_VERSION) -> KvHintAction:
    return KvHintAction(
        action_id="a-1",
        action_type=PREFETCH_ACTION_TYPE,
        action_version=version,
        payload={"block_hashes": [h.hex() for h in hashes]},
    )


def other_action() -> KvHintAction:
    return KvHintAction(
        action_id="a-2",
        action_type="agenthint.session",
        action_version="1.0",
        payload={},
    )


# ---------------------------------------------------------------- decoding


def test_no_envelope_means_no_prefetch():
    assert decode_prefetch_hashes(None) == ()


def test_envelope_without_a_prefetch_action():
    assert decode_prefetch_hashes(envelope(other_action())) == ()


def test_prefetch_action_round_trips_its_hashes():
    decoded = decode_prefetch_hashes(envelope(other_action(), prefetch_action()))
    assert decoded == HASHES


def test_unknown_action_version_is_ignored():
    action = prefetch_action(version="9.9")
    assert decode_prefetch_hashes(envelope(action)) == ()


def test_payload_without_hashes_is_ignored():
    action = KvHintAction(
        action_id="a-3",
        action_type=PREFETCH_ACTION_TYPE,
        action_version=PREFETCH_ACTION_VERSION,
        payload={},
    )
    assert decode_prefetch_hashes(envelope(action)) == ()


def test_malformed_hash_is_ignored():
    action = KvHintAction(
        action_id="a-4",
        action_type=PREFETCH_ACTION_TYPE,
        action_version=PREFETCH_ACTION_VERSION,
        payload={"block_hashes": ["not-hex"]},
    )
    assert decode_prefetch_hashes(envelope(action)) == ()


# ----------------------------------------------------------------- planning


def test_plan_keeps_whole_chunks_and_keys_each_by_its_last_hash():
    plan = plan_prefetch(HASHES, hashes_per_chunk=8, blocks_per_chunk=8, group_ids=(0,))
    assert plan is not None
    assert plan.num_chunks == 2
    # 20 hashes make two whole chunks; the partial tail is dropped.
    assert plan.block_hashes == HASHES[:16]
    keys = plan.keys_by_group[0]
    assert [get_offload_block_hash(k) for k in keys] == [HASHES[7], HASHES[15]]
    assert [get_offload_group_idx(k) for k in keys] == [0, 0]


def test_plan_carries_one_key_per_chunk_for_every_group():
    plan = plan_prefetch(
        HASHES, hashes_per_chunk=8, blocks_per_chunk=8, group_ids=(0, 1)
    )
    assert plan is not None
    assert sorted(plan.keys_by_group) == [0, 1]
    for group_id, keys in plan.keys_by_group.items():
        assert len(keys) == 2
        assert [get_offload_group_idx(k) for k in keys] == [group_id, group_id]
        assert [get_offload_block_hash(k) for k in keys] == [HASHES[7], HASHES[15]]


def test_plan_declines_when_granularities_disagree():
    assert (
        plan_prefetch(HASHES, hashes_per_chunk=8, blocks_per_chunk=1, group_ids=(0,))
        is None
    )


def test_plan_declines_without_a_whole_chunk():
    assert (
        plan_prefetch(
            HASHES[:5], hashes_per_chunk=8, blocks_per_chunk=8, group_ids=(0,)
        )
        is None
    )
    assert (
        plan_prefetch(HASHES, hashes_per_chunk=8, blocks_per_chunk=8, group_ids=())
        is None
    )


# -------------------------------------------------------------- truncation


def test_truncate_keeps_the_leading_chunks_only():
    plan = plan_prefetch(
        HASHES, hashes_per_chunk=8, blocks_per_chunk=8, group_ids=(0, 1)
    )
    assert plan is not None
    shorter = truncate_plan(plan, 1)
    assert shorter is not None
    assert shorter.num_chunks == 1
    assert shorter.block_hashes == HASHES[:8]
    assert all(len(keys) == 1 for keys in shorter.keys_by_group.values())


def test_truncate_to_nothing_declines():
    plan = plan_prefetch(HASHES, hashes_per_chunk=8, blocks_per_chunk=8, group_ids=(0,))
    assert plan is not None
    assert truncate_plan(plan, 0) is None


def test_truncate_beyond_the_plan_is_a_no_op():
    plan = plan_prefetch(HASHES, hashes_per_chunk=8, blocks_per_chunk=8, group_ids=(0,))
    assert plan is not None
    assert truncate_plan(plan, 5) is plan


# ------------------------------------------------ reservation becomes a spec


def test_a_plan_describes_the_spec_of_its_reserved_blocks():
    pool = BlockPool(num_gpu_blocks=200, enable_caching=True, hash_block_size=16)
    plan = plan_prefetch(
        HASHES, hashes_per_chunk=8, blocks_per_chunk=8, group_ids=(0, 1)
    )
    assert plan is not None
    pinned = pool.pin_blocks_by_hash(plan.block_hashes, (0, 1), block_size=16)
    assert pinned is not None
    # One GPU block per hash per group, group-major: exactly what the load spec
    # asserts, since sum(group_sizes) has to be len(block_ids).
    block_ids = plan.dst_block_ids(pinned)
    group_sizes = plan.group_sizes(pinned)
    assert len(block_ids) == 32
    assert group_sizes == [16, 16]
    assert sum(group_sizes) == len(block_ids)
    assert plan.block_indices() == [0, 0]
    # Ids are disjoint across groups and every block is held exactly once.
    assert len(set(block_ids)) == 32
    assert all(
        block.ref_cnt == 1 for pinned_set in pinned for block in pinned_set.blocks
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
