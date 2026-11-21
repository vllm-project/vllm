# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the EAGLE/MTP block drop actually costs a hybrid mamba model.

`prefix_match_unit` exists so the drop is one hash unit instead of one cache
block. It does that for full attention. It does not do it for the reconciled
hit, because the mamba group only materializes state at block boundaries: capped
at the attention candidate, its lookup floors to the previous boundary and the
joint hit gives up a whole block.

Config below: block 16, hash unit 4, owner caches three blocks (48 tokens).

    full attention alone   48 -> 44   one hash unit, as intended
    mamba alone            48         (it ignores drop_eagle_block)
    reconciled                  32    a whole block

Neither group loses a block by itself. The block is lost in the reconciliation.
"""

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

BLOCK = 16
HASH = 4  # prefix_match_unit: partial matching is ON
OWNER = 48  # three whole blocks


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _manager(use_eagle: bool):
    kv_cache_config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=BLOCK,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=4096,
        enable_caching=True,
        hash_block_size=HASH,
        use_eagle=use_eagle,
    )
    assert manager.coordinator.enable_partial_hash_hits, "partial matching must be on"
    return manager


def _seed_owner(manager):
    """Prefill 48 tokens the way the scheduler splits them.

    Stops on each block boundary, and also one hash unit below each -- the
    position EAGLE resumes from. A chunk has to END there for the state to be
    materializable, so the stop is what gives the fix something to register.
    Today nothing is registered at those positions; the assertion below is what
    changes when something is.
    """
    owner = make_request("owner", [7] * OWNER, HASH, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(owner)
    done, first = 0, True
    stops = sorted(
        {b - HASH for b in range(BLOCK, OWNER + 1, BLOCK)}
        | set(range(BLOCK, OWNER + 1, BLOCK))
    )
    for stop in stops:
        step = stop - done
        if step <= 0:
            continue
        ok = (
            manager.allocate_slots(owner, step, num_computed, computed)
            if first
            else manager.allocate_slots(owner, step)
        )
        assert ok is not None
        first, done = False, stop
        owner.num_computed_tokens = done
        manager.new_step_starts()
    return owner


def _follower(manager):
    f = make_request("follower", [7] * OWNER + [9] * 8, HASH, sha256)
    blocks, joint, _ = manager.get_computed_blocks(f)
    _pg_blocks, per_group = manager.coordinator.find_longest_cache_hit_per_group(
        f.block_hashes, OWNER + 7
    )
    return joint, tuple(per_group), blocks


def test_full_attention_gives_up_only_one_hash_unit():
    """The property `prefix_match_unit` is supposed to deliver. It holds."""
    manager = _manager(use_eagle=True)
    _seed_owner(manager)
    _joint, per_group, _ = _follower(manager)
    assert per_group[0] == OWNER - HASH == 44


def test_mamba_alone_reaches_the_full_prefix():
    """Evaluated on its own, mamba ignores drop_eagle_block and reaches 48."""
    manager = _manager(use_eagle=True)
    _seed_owner(manager)
    _joint, per_group, _ = _follower(manager)
    assert per_group[1] == OWNER == 48


def test_without_eagle_nothing_is_given_up():
    """Isolates the drop as the cause: no eagle, no loss."""
    manager = _manager(use_eagle=False)
    _seed_owner(manager)
    joint, _per_group, _ = _follower(manager)
    assert joint == OWNER == 48


@pytest.mark.xfail(
    strict=True,
    reason="mamba has no checkpoint at the position EAGLE resumes from, so the "
    "reconciled hit floors to the previous block boundary",
)
def test_reconciled_hit_should_not_give_up_a_whole_block():
    """The defect. Both groups individually reach at least 44, but the
    reconciled hit is 32: mamba is capped at the attention candidate (44) and
    its state exists only at 16/32/48, so it floors to 32.

    Fails on main (32). Passes once a mamba checkpoint exists at the position
    EAGLE resumes from, i.e. one hash unit below a block boundary.
    """
    manager = _manager(use_eagle=True)
    _seed_owner(manager)
    joint, per_group, blocks = _follower(manager)
    assert min(per_group) >= OWNER - HASH, per_group
    assert joint == OWNER - HASH, (
        f"joint hit {joint} gave up a whole {BLOCK}-token block; full attention "
        f"only gave up {OWNER - per_group[0]} (per-group hits {per_group})"
    )
    assert all(len(g) * BLOCK >= joint for g in blocks.blocks)
