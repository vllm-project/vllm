# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-step ghost-block defer guard (upstream PR #42359, ported).

A block's hash is published to the shared BlockPool at scheduling time, before
the forward pass writes its KV. A request admitted later in the SAME step can
match it and read unwritten values. MambaManager has guarded this since #29387;
every other manager -- including the MLA ones DeepSeek-V4 uses -- did not.

These tests pin down three things that are easy to get silently wrong:

1. the gate is OFF by default, so the port cannot change behaviour unasked;
2. the gate reads ``use_eagle`` *at call time*. This fork assigns it after
   construction (the coordinator does it once attention groups are known), so a
   port that captured it in ``__init__`` would read False forever and the guard
   would never fire while still looking installed;
3. the guard actually defers -- the negative control must also be checked, or a
   guard that can never fire passes as a guard that works.
"""

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    MLAAttentionManager,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


def _pool():
    return BlockPool(num_gpu_blocks=100, enable_caching=True, hash_block_size=BLOCK_SIZE)


def _full_manager(block_pool):
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    return FullAttentionManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK_SIZE,
    )


def _mla_manager(block_pool):
    spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    return MLAAttentionManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK_SIZE,
    )


@pytest.mark.parametrize("make", [_full_manager, _mla_manager])
def test_guard_is_off_by_default(make, monkeypatch):
    """Unset env => the port must be inert, whatever use_eagle says."""
    monkeypatch.setattr(
        "vllm.envs.VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT", False, raising=False
    )
    manager = make(_pool())
    manager.use_eagle = True
    assert manager._ghost_block_guard_enabled is False


@pytest.mark.parametrize("make", [_full_manager, _mla_manager])
@pytest.mark.parametrize("use_eagle,expected", [(True, True), (False, False)])
def test_gate_reads_use_eagle_at_call_time(make, use_eagle, expected, monkeypatch):
    """use_eagle is assigned AFTER construction in this fork.

    Constructing first and only then setting the attribute is exactly the order
    the coordinator uses. A port that captured use_eagle in __init__ fails here.
    """
    monkeypatch.setattr(
        "vllm.envs.VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT", True, raising=False
    )
    manager = make(_pool())
    assert manager._ghost_block_guard_enabled is False  # use_eagle still default
    manager.use_eagle = use_eagle
    assert manager._ghost_block_guard_enabled is expected


@pytest.mark.parametrize("make", [_full_manager, _mla_manager])
def test_mode_2_covers_every_group_regardless_of_use_eagle(make, monkeypatch):
    """Mode 2 is why the env var is tri-state rather than a bool.

    Upstream gates on use_eagle, having framed this as a spec-decode issue. On
    DeepSeek-V4 only the sliding-window groups carry the eagle flag, so mode 1
    measured 2 of 5 managers active and left MLAAttentionManager -- the main
    attention path -- unguarded. The race is in block publication and does not
    care about spec decode, so mode 2 covers every group.
    """
    monkeypatch.setattr(
        "vllm.envs.VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT", 2, raising=False
    )
    manager = make(_pool())
    manager.use_eagle = False  # the case mode 1 leaves unguarded
    assert manager._ghost_block_guard_enabled is True


def test_mamba_guard_is_always_on(monkeypatch):
    """Mamba's guard predates the env var (#29387) and must ignore it."""
    monkeypatch.setattr(
        "vllm.envs.VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT", False, raising=False
    )
    from vllm.v1.kv_cache_interface import MambaSpec

    spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.float32,),
        block_size=BLOCK_SIZE,
        page_size_padded=None,
        mamba_type="mamba2",
        num_speculative_blocks=0,
    )
    manager = MambaManager(
        spec,
        block_pool=_pool(),
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK_SIZE,
    )
    manager.use_eagle = False
    assert manager._ghost_block_guard_enabled is True


@pytest.mark.parametrize("make", [_full_manager, _mla_manager])
def test_defers_only_when_the_tail_was_published_this_step(make, monkeypatch):
    """The behavioural test, with its negative control.

    A tail published in this step must defer (an impossible block count, which
    makes allocate_slots return None). A tail published in an earlier step must
    not. Without the second half, a guard wired to always defer would pass.
    """
    monkeypatch.setattr(
        "vllm.envs.VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT", True, raising=False
    )
    pool = _pool()
    manager = make(pool)
    manager.use_eagle = True

    # block_hash is a read-only property; the group id is folded into the key,
    # which is why the guard's set is typed BlockHashWithGroupId.
    tail = KVCacheBlock(block_id=7)
    tail.set_block_hash(
        make_block_hash_with_group_id(BlockHash(b"tail"), group_id=0),
        num_tokens=BLOCK_SIZE,
    )

    def num_blocks(**kw):
        return manager.get_num_blocks_to_allocate(
            request_id="r1",
            num_tokens=BLOCK_SIZE,
            new_computed_blocks=[tail],
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=BLOCK_SIZE,
            **kw,
        )

    # published in an earlier step -> ordinary allocation, no defer
    baseline = num_blocks()
    assert baseline <= pool.num_gpu_blocks

    # published in THIS step -> defer
    manager.cached_blocks_this_step.add(tail.block_hash)
    assert num_blocks() == pool.num_gpu_blocks + 1

    # a new step clears the set -> back to ordinary allocation
    manager.new_step_starts()
    assert num_blocks() == baseline
