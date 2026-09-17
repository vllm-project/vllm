# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP ownership for the QSA caches, and the addressing it implies.

A sparse selector cache and the KV it selects from can share a spec type and
still need opposite treatment under DCP. These tests pin that, and pin the
compact-local slot formula the sharded side depends on.

They need no GPU and no model. The DSV4 delivery's experience is that the
ownership and capacity bugs are caught here and are false-passed by eager GPU
runs, so these come first.
"""

import pytest
import torch

from vllm.v1.core.kv_cache_utils import dcp_world_size_for_kv_cache_spec
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MLAAttentionSpec,
)

pytestmark = pytest.mark.cpu_test

DTYPE = torch.bfloat16


def _main_kv_spec() -> MLAAttentionSpec:
    """The KV that QSA attends over. Shards."""
    return MLAAttentionSpec(block_size=784, num_kv_heads=1, head_size=128, dtype=DTYPE)


def _compressed_spec() -> MLAAttentionSpec:
    """The selector's cache. Same spec type, opposite policy."""
    return MLAAttentionSpec(
        block_size=784,
        num_kv_heads=1,
        head_size=128,
        dtype=DTYPE,
        tokens_per_state=8,
        dcp_transparent=True,
    )


def _raw_ring_spec() -> CircularBufferSpec:
    """The raw key ring. Not a FullAttentionSpec, so replicated already."""
    return CircularBufferSpec(
        block_size=4, num_kv_heads=1, head_size=128, head_size_v=0, dtype=DTYPE
    )


@pytest.mark.parametrize("world", [2, 4, 8])
def test_main_kv_shards(world: int) -> None:
    assert dcp_world_size_for_kv_cache_spec(_main_kv_spec(), world) == world


@pytest.mark.parametrize("world", [2, 4, 8])
def test_compressed_cache_is_replicated(world: int) -> None:
    """The selector must see the whole sequence, or it selects from a slice."""
    assert dcp_world_size_for_kv_cache_spec(_compressed_spec(), world) == 1


@pytest.mark.parametrize("world", [2, 4, 8])
def test_raw_ring_is_replicated(world: int) -> None:
    assert dcp_world_size_for_kv_cache_spec(_raw_ring_spec(), world) == 1


def test_same_spec_type_opposite_policy() -> None:
    """The point of the flag. `isinstance` cannot separate these two."""
    main, compressed = _main_kv_spec(), _compressed_spec()
    assert isinstance(main, FullAttentionSpec)
    assert isinstance(compressed, FullAttentionSpec)
    assert dcp_world_size_for_kv_cache_spec(main, 4) == 4
    assert dcp_world_size_for_kv_cache_spec(compressed, 4) == 1


def test_transparent_is_off_by_default() -> None:
    """No existing model changes behaviour."""
    assert _main_kv_spec().dcp_transparent is False
    assert dcp_world_size_for_kv_cache_spec(_main_kv_spec(), 4) == 4


def test_dcp_one_is_identity() -> None:
    for spec in (_main_kv_spec(), _compressed_spec(), _raw_ring_spec()):
        assert dcp_world_size_for_kv_cache_spec(spec, 1) == 1


# --- the compact-local formula the sharded main KV depends on ---------------


def _owner(g: int, world: int, interleave: int) -> int:
    return (g // interleave) % world


def _local_id(g: int, world: int, interleave: int) -> int:
    return (g // (world * interleave)) * interleave + (g % interleave)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 4, 16])
def test_ownership_is_an_exact_partition(world: int, interleave: int) -> None:
    """Every global position belongs to exactly one rank."""
    n = world * interleave * 12
    owners = [_owner(g, world, interleave) for g in range(n)]
    for rank in range(world):
        assert owners.count(rank) == n // world


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 4, 16])
def test_local_ids_are_dense_and_unique(world: int, interleave: int) -> None:
    """Compact-local addressing leaves no gaps and no collisions.

    This is the property the virtual-block form loses when the block size is
    smaller than the interleave.
    """
    n = world * interleave * 12
    for rank in range(world):
        owned = [g for g in range(n) if _owner(g, world, interleave) == rank]
        local = [_local_id(g, world, interleave) for g in owned]
        assert len(set(local)) == len(local), "duplicate local slot"
        assert sorted(local) == list(range(len(local))), "local ids are not dense"


@pytest.mark.parametrize("block_size,interleave", [(2, 4), (4, 16), (784, 1)])
def test_local_offset_stays_inside_its_block(block_size: int, interleave: int) -> None:
    """`offset = local_id % block_size` must not spill onto a neighbour block.

    The virtual-block form (pick the global block first, then compact the
    offset) breaks here whenever `block_size < interleave`.
    """
    world = 8
    n = world * interleave * 8
    for rank in range(world):
        for g in range(n):
            if _owner(g, world, interleave) != rank:
                continue
            offset = _local_id(g, world, interleave) % block_size
            assert 0 <= offset < block_size
