# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay-start alignment plus the per-group hit ledger a replay group keeps.

A bounded-replay group rebuilds its KV from the first replayed token. Two
halves have to agree on that position: the scheduler floors the start to every
replayed group's block boundary, and the KV cache coordinator truncates that
group's hit ledger to the same position. The ledger decides how many leading
blocks the group pads with null blocks, so an untruncated ledger puts the
rebuilt window inside a null block even when the start itself is aligned.

Everything here runs on plain in-memory groups: no engine, no worker, no
device.
"""

from types import SimpleNamespace
from typing import Any

import torch

from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_utils import (
    partial_hash_hits_enabled,
    resolve_prefix_replay_block_align,
    resolve_prefix_replay_start,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroup, MambaSpec


def _attention_group(block_size: int) -> KVCacheGroup:
    spec = AttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    layer = FullAttentionManager.layer_config(spec)
    return KVCacheGroup.from_layers({"layer": layer})


def _mamba_group(
    block_size: int, tokens_per_state: int, prefix_caching: bool
) -> KVCacheGroup:
    spec = MambaSpec(
        block_size=block_size,
        shapes=((2, 4),),
        dtypes=(torch.float32,),
        tokens_per_state=tokens_per_state,
    )
    layer = MambaManager.layer_config(spec, enable_prefix_caching=prefix_caching)
    return KVCacheGroup.from_layers({"layer": layer})


def _replay_group(block_size: int, sliding_window: int) -> KVCacheGroup:
    spec = AttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    layer = SlidingWindowManager.layer_config(
        spec, sliding_window=sliding_window, bounded_replay=True
    )
    return KVCacheGroup.from_layers({"sliding_window": layer})


def _ring_group(block_size: int) -> KVCacheGroup:
    spec = AttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    layer = CircularBufferManager.layer_config(spec)
    return KVCacheGroup.from_layers({"ring": layer})


def _sliding_window_config(block_size: int, sliding_window: int, bounded_replay: bool):
    spec = AttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    layer = SlidingWindowManager.layer_config(
        spec, sliding_window=sliding_window, bounded_replay=bounded_replay
    )
    return layer.manager_config


class _Coordinator:
    """Binds the real ledger method to a plain in-memory group list."""

    def __init__(self, groups: list[KVCacheGroup], managers=()) -> None:
        self.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
        self.single_type_managers = list(managers)
        self.retention_interval = 0

    def _replay_group_computed_tokens(
        self,
        group_id: int,
        num_local_computed_tokens: int,
        total_computed_tokens: int,
        replay_start: int,
    ) -> tuple[int, int]:
        coordinator: Any = self
        return KVCacheCoordinator._replay_group_computed_tokens(
            coordinator,
            group_id,
            num_local_computed_tokens,
            total_computed_tokens,
            replay_start,
        )


def _ledger(
    groups: list[KVCacheGroup],
    group_id: int,
    num_local_computed_tokens: int,
    total_computed_tokens: int,
    replay_start: int,
) -> tuple[int, int]:
    """The (local, total) pair the coordinator hands this group."""
    return _Coordinator(groups)._replay_group_computed_tokens(
        group_id,
        num_local_computed_tokens,
        total_computed_tokens,
        replay_start,
    )


def test_replay_start_is_block_aligned_for_a_partial_hit() -> None:
    """Block 16, replay window 16, hit 47: the reviewed defect."""
    block_size = 16
    window = 16
    num_hit_tokens = 47
    groups = [_attention_group(block_size), _replay_group(block_size, window)]

    align = resolve_prefix_replay_block_align(groups, 1)
    assert align == block_size

    unaligned = num_hit_tokens - window
    assert unaligned == 31
    assert unaligned % block_size != 0

    replay_start = resolve_prefix_replay_start(num_hit_tokens, window, align)
    assert replay_start == 16
    assert replay_start % block_size == 0

    config = _sliding_window_config(block_size, window, bounded_replay=True)
    retained = SlidingWindowManager.sliding_window_reach(config)
    assert retained == 15
    manager = SimpleNamespace(num_retained_tokens=retained)

    num_skipped_tokens = SlidingWindowManager.get_num_skipped_tokens(
        manager, replay_start
    )
    assert num_skipped_tokens == 1
    assert num_skipped_tokens // block_size == 0
    assert (0, num_skipped_tokens // block_size * block_size) == (0, 0)

    # The ledger stops at the aligned start, so the group pads no null block in
    # front of the replay.
    local, total = _ledger(groups, 1, num_hit_tokens, num_hit_tokens, replay_start)
    assert (local, total) == (replay_start, replay_start)
    skipped_local = SlidingWindowManager.get_num_skipped_tokens(manager, local)
    assert skipped_local // block_size == 0

    # Without the truncation the ledger stops at the raw hit: two null blocks,
    # and the replay start sits inside the second one.
    raw_skipped = SlidingWindowManager.get_num_skipped_tokens(manager, num_hit_tokens)
    assert raw_skipped == 32
    assert raw_skipped // block_size == 2
    assert replay_start // block_size < raw_skipped // block_size


def test_replay_start_alignment_uses_the_lcm_of_replay_blocks() -> None:
    groups = [_attention_group(32), _replay_group(8, 8), _replay_group(16, 16)]
    assert resolve_prefix_replay_block_align(groups, 1) == 16
    assert (
        resolve_prefix_replay_block_align(
            [_replay_group(8, 8), _replay_group(16, 16)], 1
        )
        == 16
    )
    # Nothing replays: the grid collapses to one token.
    assert resolve_prefix_replay_block_align([_attention_group(32)], 1) == 1

    assert resolve_prefix_replay_start(100, 16, 16) == 80
    assert resolve_prefix_replay_start(100, 16, 8) == 80
    assert resolve_prefix_replay_start(100, 16, 64) == 64
    # A start before the first token clamps to 0, never negative.
    assert resolve_prefix_replay_start(10, 16, 16) == 0
    assert resolve_prefix_replay_start(0, 16, 16) == 0


def test_replay_alignment_only_lengthens_the_window() -> None:
    num_hit_tokens = 100
    window = 16
    unaligned = num_hit_tokens - window
    for align in (8, 16, 32, 64):
        start = resolve_prefix_replay_start(num_hit_tokens, window, align)
        assert 0 <= unaligned - start <= align - 1
        assert num_hit_tokens - start >= window


def test_replay_group_ledger_truncates_only_the_replay_group() -> None:
    groups = [_attention_group(32), _replay_group(16, 16)]
    # A group without a replay window keeps the raw hit.
    assert _ledger(groups, 0, 47, 47, 16) == (47, 47)
    # The replay group treats the aligned start as its whole prefix.
    assert _ledger(groups, 1, 47, 47, 16) == (16, 16)
    # The total keeps its offset above the local hit (external tokens).
    assert _ledger(groups, 1, 47, 52, 16) == (16, 21)
    # No replay this step: nothing is truncated.
    assert _ledger(groups, 1, 47, 47, 0) == (47, 47)


class _RecordingManager:
    """Stands in for a single-type manager and records its ledger input."""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.seen: list[tuple[int, int]] = []

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        self.seen.append((total_computed_tokens, num_local_computed_tokens))
        return 0


def test_allocation_sees_the_truncated_ledger() -> None:
    groups = [_attention_group(32), _replay_group(16, 16)]
    managers = [_RecordingManager(32), _RecordingManager(16)]
    coordinator = _Coordinator(groups, managers)
    allocated = KVCacheCoordinator.get_num_blocks_to_allocate(
        coordinator,
        request_id="r",
        num_tokens=47,
        new_computed_blocks=([], []),
        num_encoder_tokens=0,
        total_computed_tokens=47,
        num_local_computed_tokens=47,
        num_tokens_main_model=47,
        replay_start=16,
    )
    assert allocated == 0
    assert managers[0].seen == [(47, 47)]
    assert managers[1].seen == [(16, 16)]


def test_ring_group_without_replay_is_untouched() -> None:
    ring = _ring_group(8)
    assert ring.prefix_cacheable is False
    assert ring.prefix_replay_tokens == 0
    assert CircularBufferManager.supports_fine_grained_hash_lookup is False

    # A ring group adds no replay boundary and takes no truncation.
    assert resolve_prefix_replay_block_align([_replay_group(16, 16), ring], 1) == 16
    assert resolve_prefix_replay_block_align([ring], 1) == 1
    assert _ledger([_attention_group(32), ring], 1, 47, 47, 16) == (47, 47)

    # A sliding-window group without a replay window keeps its plain reach.
    plain = _sliding_window_config(8, 8, bounded_replay=False)
    assert SlidingWindowManager.prefix_replay_tokens(plain) == 0
    assert SlidingWindowManager.sliding_window_reach(plain) == 7


def test_partial_hash_gate_stays_on_the_base_policy() -> None:
    # No Mamba group with prefix caching: the gate stays closed.
    assert not partial_hash_hits_enabled([_attention_group(16)], 16)
    assert not partial_hash_hits_enabled([_attention_group(8)], 8)
    assert not partial_hash_hits_enabled(
        [_mamba_group(16, tokens_per_state=1, prefix_caching=False)], 8
    )
    # A cache block spanning more than one hash block opens it.
    assert partial_hash_hits_enabled(
        [_mamba_group(16, tokens_per_state=1, prefix_caching=True)], 8
    )
