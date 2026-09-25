# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP ownership for the QSA caches, and the addressing it implies.

A sparse selector cache and the KV it selects from share a spec type and need
opposite treatment under DCP. These tests pin that, the grouping rule that lets
both share one block table, and the compact-local slot formula the sharded side
depends on. They need no GPU and no model.
"""

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test

DTYPE = torch.bfloat16
BLOCK = 784
MAX_LEN = 262144


def _main_kv(dcp: int = 1) -> MLAAttentionSpec:
    """The KV that QSA attends over. Sharded, so its block counts slots."""
    del dcp
    return MLAAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=128, dtype=DTYPE
    )


def _selector(dcp: int) -> MLAAttentionSpec:
    """The selector cache, as QSACompressedKeyCache declares it."""
    return MLAAttentionSpec(
        block_size=BLOCK * dcp,
        num_kv_heads=1,
        head_size=128,
        dtype=DTYPE,
        tokens_per_state=8,
        dcp_sharded=False,
        storage_block_size=BLOCK * dcp,
    )


def _narrow_selector() -> MLAAttentionSpec:
    """A selector left at the unscaled block, which cannot share a group."""
    return MLAAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=128,
        dtype=DTYPE,
        tokens_per_state=8,
        dcp_sharded=False,
    )


def _raw_ring() -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=DTYPE,
        dcp_sharded=False,
    )


def _span(spec, dcp: int) -> int:
    return spec.block_size * (dcp if spec.dcp_sharded else 1)


# --- ownership ---------------------------------------------------------------


def test_main_kv_is_sharded_by_default() -> None:
    assert _main_kv().dcp_sharded


@pytest.mark.parametrize("spec", [_selector(2), _raw_ring()])
def test_replicated_caches_opt_out(spec) -> None:
    assert not spec.dcp_sharded


def test_same_spec_type_carries_opposite_ownership() -> None:
    """The spec type cannot decide this; both are MLAAttentionSpec."""
    assert type(_main_kv()) is type(_selector(2))
    assert _main_kv().dcp_sharded != _selector(2).dcp_sharded


# --- grouping ----------------------------------------------------------------


@pytest.mark.parametrize("dcp", [2, 4])
def test_sharded_and_replicated_group_when_spans_match(dcp: int) -> None:
    """The point of the change: one block table for both."""
    specs = {"main": _main_kv(), "selector": _selector(dcp)}
    assert _span(specs["main"], dcp) == _span(specs["selector"], dcp)
    assert UniformTypeKVCacheSpecs.is_uniform_type(specs, dcp)


def test_specs_do_not_group_when_spans_differ() -> None:
    """A selector at 784 against a main KV spanning 1568 needs its own table."""
    specs = {"main": _main_kv(), "selector": _narrow_selector()}
    assert not UniformTypeKVCacheSpecs.is_uniform_type(specs, 2)


def test_default_world_size_refuses_mixed_ownership() -> None:
    """Callers without a config keep the block-size comparison they had."""
    specs = {"main": _main_kv(), "selector": _selector(2)}
    assert not UniformTypeKVCacheSpecs.is_uniform_type(specs)


def test_one_rank_compares_block_size_exactly() -> None:
    """At world 1 the span is the block size, so nothing moves."""
    specs = {"main": _main_kv(), "selector": _selector(1)}
    assert UniformTypeKVCacheSpecs.is_uniform_type(specs, 1)
    assert {s.block_size for s in specs.values()} == {BLOCK}


# --- the group's block size --------------------------------------------------


def test_group_block_size_is_the_sharded_members() -> None:
    """The slot mapper scales this by the world size, so it must be 784.

    The span counts the world size twice. A gcd equals 784 only for this member
    set; a third member with a smaller block drags it below.
    """
    group = UniformTypeKVCacheSpecs.from_specs(
        {"main": _main_kv(), "selector": _selector(2)}, 2
    )
    assert group is not None
    assert group.block_size == BLOCK


def test_group_block_size_divides_every_member() -> None:
    group = UniformTypeKVCacheSpecs.from_specs(
        {"main": _main_kv(), "selector": _selector(2)}, 2
    )
    assert all(
        spec.block_size % group.block_size == 0
        for spec in group.kv_cache_specs.values()
    )


def test_all_replicated_group_keeps_its_block_size() -> None:
    group = UniformTypeKVCacheSpecs.from_specs(
        {"a": _selector(2), "b": _selector(2)}, 2
    )
    assert group is not None and group.block_size == BLOCK * 2


# --- block-table width and sizing --------------------------------------------


@pytest.mark.parametrize("dcp", [2, 4])
def test_both_members_need_the_same_block_table_width(dcp: int) -> None:
    assert cdiv(MAX_LEN, _span(_main_kv(), dcp)) == cdiv(
        MAX_LEN, _span(_selector(dcp), dcp)
    )


def test_a_narrow_selector_needs_a_wider_block_table() -> None:
    """168 against 335, which is what forced the split."""
    assert cdiv(MAX_LEN, _span(_main_kv(), 2)) == 168
    assert cdiv(MAX_LEN, _span(_narrow_selector(), 2)) == 335


@pytest.mark.parametrize("dcp", [2, 4])
def test_a_replicated_cache_is_budgeted_for_the_whole_sequence(dcp: int) -> None:
    """Sizing follows the flag, or the pool is short by the world size."""
    assert _span(_selector(dcp), dcp) == _selector(dcp).block_size


def test_raw_ring_span_is_not_scaled() -> None:
    """A replicated AttentionSpec: the old type rule scaled this to 8."""
    assert _span(_raw_ring(), 2) == 4


# --- storage_block_size ------------------------------------------------------


def test_storage_block_size_must_divide_block_size() -> None:
    with pytest.raises(AssertionError, match="storage_block_size"):
        MLAAttentionSpec(
            block_size=1000,
            num_kv_heads=1,
            head_size=128,
            dtype=DTYPE,
            dcp_sharded=False,
            storage_block_size=BLOCK,
        )


# --- compact-local addressing ------------------------------------------------
#
# Must agree with the DCP branch of the slot-mapping kernel in
# vllm/v1/worker/block_table.py, or a rank attends over keys it does not hold.


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
    """Compact-local addressing leaves no gaps and no collisions."""
    n = world * interleave * 12
    for rank in range(world):
        owned = [g for g in range(n) if _owner(g, world, interleave) == rank]
        local = [_local_id(g, world, interleave) for g in owned]
        assert len(set(local)) == len(local), "duplicate local slot"
        assert sorted(local) == list(range(len(local))), "local ids are not dense"


@pytest.mark.parametrize("block_size,interleave", [(2, 4), (4, 16), (784, 1)])
def test_local_offset_stays_inside_its_block(block_size: int, interleave: int) -> None:
    """`offset = local_id % block_size` must not spill onto a neighbour block."""
    world = 8
    n = world * interleave * 8
    for rank in range(world):
        for g in range(n):
            if _owner(g, world, interleave) != rank:
                continue
            offset = _local_id(g, world, interleave) % block_size
            assert 0 <= offset < block_size


# --- merge -------------------------------------------------------------------


@pytest.mark.parametrize("spec_fn", [lambda: _main_kv(), lambda: _selector(2)])
def test_merge_keeps_the_ownership_flag(spec_fn) -> None:
    spec = spec_fn()
    assert MLAAttentionSpec.merge([spec, spec]).dcp_sharded == spec.dcp_sharded
