# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EAGLE/MTP drop when the shared prefix ends before the owner's prompt.

The sibling file pins the case where the shared prefix runs to the END of the
owner's prompt. This one pins the other shape, which is what a served system
prompt looks like: every request begins with the same preamble and then diverges
into its own question. The owner's prompt tail therefore sits over tokens no
sibling shares, so nothing cached there can serve them.

Config below: block 16, hash unit 4. Owner prompt is 40 tokens -- a 24-token
shared prefix followed by a 16-token private suffix. The follower shares the
first 24 and then diverges.

    full attention alone   16 -> 12   one hash unit below the shared boundary
    mamba alone            16         (it ignores drop_eagle_block)
    reconciled                   0    everything

The shared prefix ends mid-block (24 % 16 = 8), so full attention's own match
stops at the last whole shared block, 16, and the drop takes it to 12. Mamba has
state only at 0 and 16; capped at 12 it can only fall back to 0, and the joint
hit is nothing at all. Without EAGLE the same follower keeps 16.

This is the shape where a check-point at the *prompt tail* cannot help: 40, and
36 one hash unit below it, are both over the owner's private suffix.
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
SHARED = 24  # the system prompt; ends mid-block on purpose
SUFFIX = 16  # the owner's own question
PROMPT = SHARED + SUFFIX


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
    """Prefill the owner's 40 tokens, stopping everywhere state could be kept.

    Block boundaries, one hash unit below each -- the position EAGLE resumes
    from -- and the prompt tail. Stopping is what makes a position registrable;
    which of them actually get an entry is what these tests report.
    """
    owner = make_request("owner", [7] * SHARED + [8] * SUFFIX, HASH, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(owner)
    done, first = 0, True
    stops = sorted(
        {b - HASH for b in range(BLOCK, PROMPT + 1, BLOCK)}
        | set(range(BLOCK, PROMPT + 1, BLOCK))
        | {PROMPT}
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
    """A sibling that shares the system prompt and then asks something else."""
    f = make_request("follower", [7] * SHARED + [9] * 8, HASH, sha256)
    _blocks, joint, _ = manager.get_computed_blocks(f)
    _pg_blocks, per_group = manager.coordinator.find_longest_cache_hit_per_group(
        f.block_hashes, SHARED + 7
    )
    return joint, tuple(per_group)


def test_full_attention_gives_up_only_one_hash_unit():
    """Full attention matches the last whole shared block and drops one unit."""
    manager = _manager(use_eagle=True)
    _seed_owner(manager)
    _joint, per_group = _follower(manager)
    assert per_group[0] == BLOCK - HASH  # 12


def test_mamba_alone_reaches_the_shared_block_boundary():
    """Mamba never applies the drop, so on its own it keeps the whole block."""
    manager = _manager(use_eagle=True)
    _seed_owner(manager)
    _joint, per_group = _follower(manager)
    assert per_group[1] == BLOCK  # 16


def test_without_eagle_the_hit_survives():
    """No drop, so the two groups agree and the follower keeps the block."""
    manager = _manager(use_eagle=False)
    _seed_owner(manager)
    joint, per_group = _follower(manager)
    assert per_group == (BLOCK, BLOCK)
    assert joint == BLOCK


def test_reconciled_hit_is_currently_nothing():
    """Today the two groups reconcile to zero: the follower recomputes it all.

    Capped at full attention's 12, mamba's only lower position is 0.
    """
    manager = _manager(use_eagle=True)
    _seed_owner(manager)
    joint, _per_group = _follower(manager)
    assert joint == 0


@pytest.mark.xfail(
    strict=True,
    reason="mamba materializes no state one hash unit below a block boundary, "
    "so the reconciled hit falls to 0 even though both groups reached >= 12",
)
def test_reconciled_hit_should_reach_the_resume_point():
    """What the follower should keep: the position EAGLE actually resumes from.

    Both groups reach at least 12 on their own. A check-point at the owner's
    prompt tail cannot supply this -- 40 and 36 are over the private suffix --
    so the state has to exist one hash unit below the shared block boundary.
    """
    manager = _manager(use_eagle=True)
    _seed_owner(manager)
    joint, _per_group = _follower(manager)
    assert joint == BLOCK - HASH  # 12
