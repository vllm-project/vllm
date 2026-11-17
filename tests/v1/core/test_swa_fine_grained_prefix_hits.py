# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fine-grained prefix-cache hits with a sliding-window KV cache group.

Hybrid full attention + Mamba ("align") + sliding window, scaled from a model
whose draft adds an SWA group: hash unit 16, attention blocks 32, window 56,
Mamba block 256 (= scheduler block).
"""

from dataclasses import replace

import pytest
import torch

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

from .test_prefix_caching import make_kv_cache_manager, make_request

UNIT = 16
BLOCK = 32
WINDOW = 56
MAMBA_BLOCK = 256

init_none_hash(sha256)


def _config(num_blocks: int = 400) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=MAMBA_BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
            KVCacheGroupSpec(
                ["swa"],
                SlidingWindowSpec(
                    block_size=BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=WINDOW,
                ),
            ),
        ],
        prefix_cache_retention_interval=0,
    )


def _manager(fine_grained: bool, monkeypatch, num_blocks: int = 400):
    monkeypatch.setattr(
        SlidingWindowManager, "supports_fine_grained_hash_lookup", fine_grained
    )
    return make_kv_cache_manager(
        _config(num_blocks),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=UNIT,
    )


def _run_producer(manager, tokens: list[int], num_output: int):
    req = make_request("producer", tokens, UNIT, sha256)
    # Chunks end at the Mamba blocks and at the prompt's last hash boundary,
    # as the scheduler's align-mode chunk split does.
    boundary = (len(tokens) - 1) // UNIT * UNIT
    ends = sorted({*range(MAMBA_BLOCK, boundary, MAMBA_BLOCK), boundary, len(tokens)})
    for end in ends:
        assert manager.allocate_slots(req, end - req.num_computed_tokens) is not None
        req.num_computed_tokens = end
        manager.new_step_starts()
    # SWA table at the end of prefill; decode later nulls pages that leave
    # the window (they stay cached in the free queue).
    req.prefill_swa_ids = list(manager.get_block_ids(req.request_id)[2])
    for i in range(num_output):
        req.append_output_token_ids(50_000 + i)
        assert manager.allocate_slots(req, 1) is not None
        req.num_computed_tokens += 1
        manager.new_step_starts()
    return req


@pytest.mark.parametrize("fine_grained", [True, False])
def test_partial_hash_hits_gate(fine_grained: bool, monkeypatch):
    manager = _manager(fine_grained, monkeypatch)
    assert manager.coordinator.enable_partial_hash_hits is fine_grained


@pytest.mark.parametrize("num_output", [4, 40])
def test_gpu_partial_hit_at_non_block_aligned_prompt_end(num_output, monkeypatch):
    """A follow-up turn (prompt + output + new tokens) hits at the previous
    prompt's last hash boundary, which ends inside an SWA block. With 4 output
    tokens the SWA tail block is still partial (partial entry); with 40 it is
    full and matched by its block hash."""
    prompt_len = 601
    boundary = (prompt_len - 1) // UNIT * UNIT  # 592: inside SWA block 18
    assert boundary % BLOCK and boundary % MAMBA_BLOCK
    manager = _manager(True, monkeypatch)
    tokens = list(range(prompt_len))
    producer = _run_producer(manager, tokens, num_output)
    producer_swa = producer.prefill_swa_ids
    tail_block_id = producer_swa[boundary // BLOCK]
    manager.free(producer)
    manager.new_step_starts()

    follow_up = tokens + list(producer.output_token_ids) + list(range(70_000, 70_100))
    consumer = make_request("consumer", follow_up, UNIT, sha256)
    blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == boundary

    swa = blocks.blocks[2]
    first = (boundary - (WINDOW - 1)) // BLOCK
    assert len(swa) == -(-boundary // BLOCK)
    assert all(b.is_null for b in swa[:first])
    assert [b.block_id for b in swa[first:]] == producer_swa[first : len(swa)]
    assert swa[-1].block_id == tail_block_id

    # The shared SWA tail block is copied before the consumer writes into it.
    assert (
        manager.allocate_slots(
            consumer,
            len(follow_up) - num_computed,
            num_new_computed_tokens=num_computed,
            new_computed_blocks=blocks,
        )
        is not None
    )
    copies, _ = manager.take_kv_cache_block_copies()
    src_ids = {c.src_block_id for c in copies}
    assert tail_block_id in src_ids
    assert manager.get_block_ids(consumer.request_id)[2][boundary // BLOCK] != (
        tail_block_id
    )


def test_block_aligned_hits_without_fine_grained_swa(monkeypatch):
    """Old behaviour: the SWA group forces scheduler-block-aligned hits."""
    prompt_len = 601
    manager = _manager(False, monkeypatch)
    tokens = list(range(prompt_len))
    producer = _run_producer(manager, tokens, 40)
    manager.free(producer)
    manager.new_step_starts()
    follow_up = tokens + list(producer.output_token_ids) + list(range(70_000, 70_100))
    consumer = make_request("consumer", follow_up, UNIT, sha256)
    _, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == (prompt_len - 1) // MAMBA_BLOCK * MAMBA_BLOCK


def test_swa_window_eviction_limits_hit(monkeypatch):
    """Evicting a window block below the boundary drops the hit to a point
    whose whole window is still cached."""
    prompt_len = 601
    boundary = (prompt_len - 1) // UNIT * UNIT
    manager = _manager(True, monkeypatch)
    tokens = list(range(prompt_len))
    producer = _run_producer(manager, tokens, 4)
    swa_ids = producer.prefill_swa_ids
    manager.free(producer)
    manager.new_step_starts()
    first = (boundary - (WINDOW - 1)) // BLOCK
    manager.block_pool.evict_blocks({swa_ids[first]})
    consumer = make_request("consumer", tokens + [1, 2, 3], UNIT, sha256)
    _, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed < boundary


def test_swa_mask_keeps_only_reachable_window(monkeypatch):
    """Retention 0: only the window ending at the replay boundary is hashed."""
    mask = SlidingWindowManager.reachable_block_mask(
        start_block=0,
        end_block=20,
        alignment_tokens=UNIT,
        kv_cache_spec=_config().kv_cache_groups[2].kv_cache_spec,
        use_eagle=False,
        retention_interval=0,
        reachable_boundaries=(600,),
    )
    assert mask is not None
    boundary = 592
    keep = set(range((boundary - (WINDOW - 1)) // BLOCK, (boundary - 1) // BLOCK + 1))
    assert {i for i, m in enumerate(mask) if m} == keep


def test_eagle_fine_grained_swa_drops_one_unit(monkeypatch):
    monkeypatch.setattr(SlidingWindowManager, "supports_fine_grained_hash_lookup", True)
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    spec = _config().kv_cache_groups[2].kv_cache_spec
    pool = BlockPool(num_gpu_blocks=50, enable_caching=True, hash_block_size=UNIT)
    tokens = list(range(200))
    req = make_request("r", tokens, UNIT, sha256)
    # Cache blocks 2..5 by their block hash (unit index (j + 1) * 2 - 1).
    for j in range(2, 6):
        pool.cached_block_hash_to_block.insert(
            make_block_hash_with_group_id(req.block_hashes[(j + 1) * 2 - 1], 0),
            pool.blocks[10 + j],
        )
    blocks, hit = SlidingWindowManager.find_longest_cache_hit(
        block_hashes=req.block_hashes,
        max_length=176,
        kv_cache_group_ids=[0],
        block_pool=pool,
        kv_cache_spec=replace(spec),
        drop_eagle_block=True,
        alignment_tokens=UNIT,
    )
    # Coverage ends at block 5 (192 tokens) but max_length caps the peek at 176.
    assert hit == 160
    assert [b.block_id for b in blocks[0] if not b.is_null] == [13, 14]
