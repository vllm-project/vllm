# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E7v2 (LMCache #4674) - the connector local-prefix over-reach.

Pure-CPU regression: with a capable KV connector and a hybrid
full-attention + Mamba model, a Mamba group hit that lags the
full-attention hit must never produce a connector-local prefix deeper
than a valid Mamba state. Fails on main, passes with the connector
path reconciliation.
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

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _manager(hash_bs, fa_bs, mamba_bs, use_eagle):
    cfg = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=fa_bs,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
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
    manager = make_kv_cache_manager(
        cfg,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_bs,
        use_eagle=use_eagle,
    )
    assert manager.coordinator.enable_partial_hash_hits, "partial matching must be on"
    return manager


def _seed_owner(manager, n, hash_bs):
    owner = make_request("owner", [7] * n, hash_bs, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(owner)
    assert num_computed == 0
    assert manager.allocate_slots(owner, n, 0, computed) is not None
    manager.free(owner)
    manager.new_step_starts()


def test_connector_divergent_local_hit_reports_mamba_lag():
    """The connector local hit must not over-reach the Mamba state boundary.

    E7v2: with a capable KV connector (spec decode on) and a hybrid
    full-attention + Mamba model, ``get_computed_blocks_for_connector``
    used to report the **full-attention** hit as ``num_local`` whenever no
    group ran deeper - even when the Mamba group lagged, so that boundary
    had no valid Mamba state (align mode materializes state only at sparse
    mamba-block positions, and under a sparse grid the Mamba hit can be 0
    while full attention is deep). With no external tokens supplied,
    serving that boundary writes a Mamba state that does not match the
    full-attention prefix: silent corruption. The scheduler's
    ``hit_diverged and num_external == 0`` fallback to
    ``get_computed_blocks`` only fires when the connector supplies nothing,
    and the reconciled path it lands on is itself incomplete for
    align-mode boundaries (open PRs #53479/#53802), so the guard alone
    does not close the defect.

    Setup: a sparse mamba grid (state every 4 tokens, full attention every
    2) makes the Mamba independent hit lag full attention; an owner request
    seeds the shared prefix and a follower diverges after 10 tokens.

    Invariant pinned: a servable connector-local prefix hit is a boundary
    at which *every* group has a valid cached state, so ``num_local`` must
    not run deeper than the Mamba group's hit, and the divergence must be
    flagged.
    """
    # Sparse grid: mamba state every 4 tokens, full attention every 2. The
    # Mamba independent hit lags (or is 0) while full attention goes deeper.
    hash_bs, fa_bs, mamba_bs = 2, 2, 4
    manager = _manager(hash_bs, fa_bs, mamba_bs, use_eagle=True)
    _seed_owner(manager, 12, hash_bs)

    # Follower shares the first 10 tokens, then diverges.
    follower = make_request("follower", [7] * 10 + [9] * 2, hash_bs, sha256)
    _blocks_per_group, hit_lengths = (
        manager.coordinator.find_longest_cache_hit_per_group(
            follower.block_hashes, follower.num_tokens - 1
        )
    )
    fa_hit, mamba_hit = hit_lengths[0], hit_lengths[1]

    # Premise of the bug: the Mamba group's independent hit lags full attention.
    assert mamba_hit < fa_hit, (
        f"need the Mamba hit lagging full attention to reach the over-reach; "
        f"got FA={fa_hit}, mamba={mamba_hit}"
    )

    # The connector path, as the scheduler calls it for a capable connector.
    blocks, num_local, _shared, hit_diverged = (
        manager.get_computed_blocks_for_connector(follower)
    )
    assert hit_diverged, (
        f"expected a diverged hit (mamba {mamba_hit} < fa {fa_hit}); "
        f"got hit_diverged={hit_diverged}"
    )

    # The invariant (see docstring): never deeper than a valid Mamba state.
    assert num_local <= mamba_hit, (
        f"connector local prefix {num_local} runs deeper than the Mamba state "
        f"boundary {mamba_hit} (full-attention hit {fa_hit}, hit_diverged="
        f"{hit_diverged}); with no external tokens the Mamba state at {num_local} "
        f"is absent, so the prefix is served misaligned."
    )
    assert blocks is not None


@pytest.mark.parametrize("fa_bs,mamba_bs", [(2, 4), (4, 2)])
@pytest.mark.parametrize(
    "max_cache_hit_length,expected_hit", [(0, 0), (1, 0), (6, 4), (8, 8), (None, 12)]
)
def test_connector_reconciliation_preserves_hit_length_cap(
    fa_bs, mamba_bs, max_cache_hit_length, expected_hit
):
    """Both divergent fallback paths must honor the caller's prefix limit."""
    manager = _manager(1, fa_bs, mamba_bs, use_eagle=False)
    request = make_request("capped", [7] * 14, 1, sha256)
    # Cache valid states at every complete block boundary through token 12.
    # Different group block sizes make a cap of 6 diverge in either direction.
    for group_id, block_size in enumerate((fa_bs, mamba_bs)):
        num_blocks = 12 // block_size
        blocks = manager.block_pool.get_new_blocks(num_blocks)
        manager.block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=0,
            num_full_blocks=num_blocks,
            block_size=block_size,
            kv_cache_group_id=group_id,
        )
        manager.block_pool.free_blocks(blocks)

    assert manager.get_computed_blocks(request)[1] == 12
    if max_cache_hit_length == 6:
        _, hits = manager.coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, max_cache_hit_length
        )
        assert hits == ((6, 4) if fa_bs == 2 else (4, 6))

    blocks, num_local, shared, hit_diverged = manager.get_computed_blocks_for_connector(
        request, max_cache_hit_length
    )
    expected_blocks, expected_local, expected_shared = manager.get_computed_blocks(
        request, max_cache_hit_length
    )
    assert num_local == expected_local == expected_hit
    assert blocks.get_block_ids() == expected_blocks.get_block_ids()
    assert shared == expected_shared
    assert hit_diverged == (max_cache_hit_length == 6 and fa_bs == 2)
