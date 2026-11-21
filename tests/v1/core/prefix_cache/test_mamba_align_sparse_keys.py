# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" stores keys only at sparse boundaries; the KV-cache event
stream indexes the hit list as a dense grid.

Interior mamba blocks are null and carry no block hash -- that part is by design
(``reachable_block_mask``). ``emit_cached_block_events`` then walks the hit list
positionally, so for the mamba group it publishes a key per hit block. The keys
it publishes are the wrong ones: the interior block that was never stored, and
one past the end of the hit, while the group's only real key is omitted. When
the consumer's prompt is not mamba-block-aligned the same walk indexes past the
block-hash view and raises IndexError inside ``get_computed_blocks``.

Requires ``kv_cache_report_mode = "full"`` (KV-cache events). CPU only.
"""

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    get_group_id,
    init_none_hash,
    maybe_convert_block_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

# (hash_block_size, full_attn_block_size, mamba_block_size)
SIZES = [
    pytest.param(2, 2, 4, id="toy"),
    pytest.param(16, 64, 64, id="large-block"),
]


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _manager(hash_bs, fa_bs, mamba_bs, events=True, num_blocks=64):
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=fa_bs, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_bs,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    return make_kv_cache_manager(
        kv_cache_config=cfg,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_bs,
        enable_kv_cache_events=events,
    )


def _produce(manager, tokens, hash_bs):
    req = make_request("producer", tokens, hash_bs, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(req)
    assert num_computed == 0
    assert manager.allocate_slots(req, len(tokens), num_computed, computed) is not None
    manager.free(req)
    manager.new_step_starts()
    manager.take_events()
    return req


# ---------------------------------------------------------------- PART 1 ----
@pytest.mark.parametrize("hash_bs,fa_bs,mamba_bs", SIZES)
def test_mamba_group_grid_is_sparse(hash_bs, fa_bs, mamba_bs):
    """The mamba group stores ONE key (the prompt tail). Every interior mamba
    block index of the same request carries no key at all."""
    manager = _manager(hash_bs, fa_bs, mamba_bs, events=False)
    n = mamba_bs + hash_bs  # 6 / 80: just over one mamba block
    req = _produce(manager, list(range(n)), hash_bs)

    key_at = {
        t: req.block_hashes[t // hash_bs - 1] for t in range(hash_bs, n + 1, hash_bs)
    }

    # mamba: only the prompt-tail key exists.
    have = [t for t, h in key_at.items() if manager.block_pool.get_cached_block(h, [1])]
    assert have == [n], have
    hit = manager.block_pool.get_cached_block(key_at[n], [1])[0]
    assert get_group_id(hit.block_hash) == 1
    assert get_block_hash(hit.block_hash) == key_at[n]
    assert hit.block_hash_num_tokens == n

    # mamba block index 0 covers tokens [0, mamba_bs) and would be keyed by the
    # hash at mamba_bs tokens: never written (intermediate state snapshot).
    assert manager.block_pool.get_cached_block(key_at[mamba_bs], [1]) is None


# ---------------------------------------------------------------- PART 2 ----
@pytest.mark.parametrize("hash_bs,fa_bs,mamba_bs", SIZES)
@pytest.mark.xfail(
    strict=True,
    reason="emit_cached_block_events walks the hit list positionally: it publishes "
    "mamba keys that were never stored and omits the one that was",
)
def test_emit_cached_block_events_invents_mamba_keys(hash_bs, fa_bs, mamba_bs):
    """kv_cache_report_mode='full' publishes mamba keys that were never stored,
    over a token range that was never hit."""
    manager = _manager(hash_bs, fa_bs, mamba_bs)
    n = mamba_bs + hash_bs
    _produce(manager, list(range(n)), hash_bs)

    req1 = make_request("consumer", list(range(2 * mamba_bs)), hash_bs, sha256)
    req1.kv_cache_report_mode = "full"
    blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == n, num_computed
    assert len(blocks.blocks[1]) == 2  # [null, hit] -- 2 blocks for an n-token hit

    stored = [e for e in manager.take_events() if type(e).__name__ == "BlockStored"]
    mamba_event = {e.group_idx: e for e in stored}[1]
    emitted = list(mamba_event.block_hashes)

    real_key = maybe_convert_block_hash(req1.block_hashes[n // hash_bs - 1])
    beyond_the_hit = maybe_convert_block_hash(
        req1.block_hashes[2 * mamba_bs // hash_bs - 1]
    )
    assert manager.block_pool.get_cached_block(req1.block_hashes[n // hash_bs - 1], [1])
    assert (
        manager.block_pool.get_cached_block(
            req1.block_hashes[mamba_bs // hash_bs - 1], [1]
        )
        is None
    )
    assert (
        manager.block_pool.get_cached_block(
            req1.block_hashes[2 * mamba_bs // hash_bs - 1], [1]
        )
        is None
    )

    assert real_key in emitted, (
        f"mamba group's only stored key (at {n} tokens) was not published"
    )
    assert beyond_the_hit not in emitted, (
        f"published a key at {2 * mamba_bs} tokens; hit was only {n}"
    )


@pytest.mark.parametrize("hash_bs,fa_bs,mamba_bs", SIZES)
def test_emit_cached_block_events_current_shape(hash_bs, fa_bs, mamba_bs):
    """What main emits today. Unmarked, so it goes red when the shape changes."""
    manager = _manager(hash_bs, fa_bs, mamba_bs)
    n = mamba_bs + hash_bs
    _produce(manager, list(range(n)), hash_bs)

    req1 = make_request("consumer", list(range(2 * mamba_bs)), hash_bs, sha256)
    req1.kv_cache_report_mode = "full"
    manager.get_computed_blocks(req1)
    stored = [e for e in manager.take_events() if type(e).__name__ == "BlockStored"]
    mamba_event = {e.group_idx: e for e in stored}[1]

    never_stored_interior = maybe_convert_block_hash(
        req1.block_hashes[mamba_bs // hash_bs - 1]
    )
    beyond_the_hit = maybe_convert_block_hash(
        req1.block_hashes[2 * mamba_bs // hash_bs - 1]
    )
    assert list(mamba_event.block_hashes) == [never_stored_interior, beyond_the_hit]
    assert len(mamba_event.token_ids) == 2 * mamba_bs


@pytest.mark.parametrize("hash_bs,fa_bs,mamba_bs", SIZES)
@pytest.mark.xfail(
    strict=True,
    raises=IndexError,
    reason="the same positional walk indexes past the block-hash view and raises "
    "IndexError inside get_computed_blocks",
)
def test_emit_cached_block_events_indexerror_on_unaligned_prompt(
    hash_bs, fa_bs, mamba_bs
):
    """The dense-grid index walks off the end of the block-hash view -> crash
    inside Scheduler.schedule()."""
    manager = _manager(hash_bs, fa_bs, mamba_bs)
    n = mamba_bs + hash_bs
    _produce(manager, list(range(n)), hash_bs)

    # Consumer prompt = n + 1 tokens: floor(len/mamba_bs) == 1 mamba-granular
    # hash, but the mamba hit list is 2 long.
    req1 = make_request("consumer", list(range(n + 1)), hash_bs, sha256)
    req1.kv_cache_report_mode = "full"
    blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == n, num_computed


# ---------------------------------------------------------------- PART 3 ----
@pytest.mark.xfail(
    strict=True,
    reason="invented keys are not specific to partial matching; reproduces with "
    "no prefix_match_unit set",
)
def test_plain_mamba_align_no_partial_hash_also_invents_keys():
    """No prefix_match_unit at all (hash_block_size == every block_size):
    align mode still leaves interior mamba blocks null/unkeyed, and the event
    stream still publishes a key for each of them."""
    bs = 4
    manager = _manager(bs, bs, bs, events=True)
    _produce(manager, list(range(2 * bs)), bs)  # 8 tokens

    assert not manager.coordinator.enable_partial_hash_hits

    req1 = make_request("consumer", list(range(3 * bs)), bs, sha256)
    req1.kv_cache_report_mode = "full"
    blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 2 * bs
    assert len(blocks.blocks[1]) == 2
    assert blocks.blocks[1][0].is_null  # interior mamba block: null, no key

    h4, h8 = req1.block_hashes[0], req1.block_hashes[1]
    assert manager.block_pool.get_cached_block(h8, [1]) is not None
    assert manager.block_pool.get_cached_block(h4, [1]) is None  # never stored

    stored = [e for e in manager.take_events() if type(e).__name__ == "BlockStored"]
    emitted = list({e.group_idx: e for e in stored}[1].block_hashes)
    assert maybe_convert_block_hash(h4) not in emitted, (
        "published a mamba key for a null (never-written) block"
    )
