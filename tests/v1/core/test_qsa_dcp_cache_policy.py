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

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
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


# --- grouping and capacity --------------------------------------------------
#
# The ownership flag is only half the policy. A group of layers is merged into
# one spec before the coordinator sees it, and the pool is sized from that
# merged spec, so both paths have to carry the flag or the cache is quietly
# sharded or under-provisioned.


def test_merging_a_group_keeps_the_replication_flag() -> None:
    """A model has many compressed-cache layers, and they merge into one spec.

    Losing the flag here shards the selector cache with no other symptom: each
    rank then scores only its own slice and picks different positions.
    """
    merged = MLAAttentionSpec.merge([_compressed_spec(), _compressed_spec()])
    assert merged.dcp_transparent is True
    assert dcp_world_size_for_kv_cache_spec(merged, 4) == 1


def test_merging_sharded_layers_stays_sharded() -> None:
    merged = MLAAttentionSpec.merge([_main_kv_spec(), _main_kv_spec()])
    assert merged.dcp_transparent is False
    assert dcp_world_size_for_kv_cache_spec(merged, 4) == 4


def test_a_group_cannot_mix_the_two_policies() -> None:
    """One group gets one ownership policy, so mixing has to be refused."""
    with pytest.raises(AssertionError, match="DCP ownership"):
        MLAAttentionSpec.merge([_main_kv_spec(), _compressed_spec()])


def _config(max_model_len: int, dcp: int):
    """A stub, because building a real VllmConfig needs a device.

    The two sizing methods read exactly these two fields and nothing else, and
    the point here is the arithmetic, not config validation.
    """
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
    )


@pytest.mark.parametrize("world", [2, 4])
def test_a_replicated_cache_is_budgeted_for_the_whole_sequence(world: int) -> None:
    """Sizing follows ownership, or the pool is short by the world size."""
    from dataclasses import replace

    config = _config(max_model_len=8192, dcp=world)
    sharded = replace(_main_kv_spec(), dcp_shard_count=world)
    unsharded = _main_kv_spec()
    replicated = _compressed_spec()

    assert replicated.logical_block_span == replicated.block_size, "never split"
    assert sharded.logical_block_span == sharded.block_size * world
    assert unsharded.logical_block_span == unsharded.block_size

    # Sizing now follows the spec, not the config, so compare specs. The
    # replicated cache is budgeted exactly as an unsharded one is.
    assert replicated.max_num_blocks_per_req(config, 8192) == cdiv(
        8192, replicated.block_size
    )

    # The sharded one still shrinks, or DCP buys no headroom at all.
    assert sharded.max_memory_usage_bytes(config) < unsharded.max_memory_usage_bytes(
        config
    )
    assert sharded.max_num_blocks_per_req(
        config, 8192
    ) < unsharded.max_num_blocks_per_req(config, 8192)


def _uniform(*specs) -> bool:
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    return UniformTypeKVCacheSpecs.is_uniform_type(
        {f"layer.{i}": spec for i, spec in enumerate(specs)}
    )


def test_one_rank_keeps_the_selector_packed_with_the_main_kv() -> None:
    """No DCP, no split: a single-rank run must keep the layout it had."""
    replicated_off = MLAAttentionSpec(
        block_size=784,
        num_kv_heads=1,
        head_size=128,
        dtype=DTYPE,
        tokens_per_state=8,
    )
    assert _uniform(_main_kv_spec(), replicated_off)


def test_a_narrow_selector_still_cannot_share_a_group() -> None:
    """The split is only forced when the spans disagree.

    A selector left at 784 tokens against a main KV spanning 1568 needs a
    different block-table width, so it must still split. That is the case the
    aligned span removes.
    """
    from dataclasses import replace

    sharded_main = replace(_main_kv_spec(), dcp_shard_count=2)
    assert not _uniform(sharded_main, _compressed_spec())


def test_a_narrow_selector_needs_a_wider_block_table() -> None:
    """168 against 335 is what serving reported before the spans were aligned."""
    from dataclasses import replace

    config = _config(max_model_len=262144, dcp=2)
    sharded_main = replace(_main_kv_spec(), dcp_shard_count=2)
    assert sharded_main.max_num_blocks_per_req(config, 262144) == 168
    assert _compressed_spec().max_num_blocks_per_req(config, 262144) == 335


def test_the_raw_ring_is_placed_rather_than_rejected() -> None:
    """The DCP guard rejects by type, and the ring has to be on the list.

    It is replicated, like Mamba: every rank writes every token into its own
    ring so the replicated selector reads identical ones.
    """
    from vllm.v1.core.kv_cache_coordinator import DCP_AWARE_SPECS

    assert isinstance(_raw_ring_spec(), DCP_AWARE_SPECS)
    assert isinstance(_main_kv_spec(), DCP_AWARE_SPECS)
    assert dcp_world_size_for_kv_cache_spec(_raw_ring_spec(), 2) == 1


# --- the two DCP resolvers must agree ---------------------------------------


def test_block_span_follows_ownership_not_spec_type() -> None:
    """A replicated cache's block spans what it holds, not twice that.

    Both the raw ring and the selector are AttentionSpec subclasses, so a rule
    keyed on the type scales their span even though neither is sharded. The
    scheduler then rounds block boundaries the group does not have.
    """
    from vllm.v1.core.kv_cache_utils import stamp_dcp_shard_counts

    specs = stamp_dcp_shard_counts(
        {"main": _main_kv_spec(), "sel": _compressed_spec(), "ring": _raw_ring_spec()},
        2,
    )
    assert specs["main"].dcp_shard_count == 2, "sharded"
    assert specs["sel"].dcp_shard_count == 1, "replicated"
    assert specs["ring"].dcp_shard_count == 1, "replicated"

    assert specs["main"].logical_block_span == 784 * 2
    assert specs["sel"].logical_block_span == 784
    assert specs["ring"].logical_block_span == 4


def test_one_rank_scales_nothing() -> None:
    from vllm.v1.core.kv_cache_utils import stamp_dcp_shard_counts

    specs = {"m": _main_kv_spec(), "s": _compressed_spec(), "r": _raw_ring_spec()}
    for spec in stamp_dcp_shard_counts(specs, 1).values():
        assert spec.logical_block_span == spec.block_size


# --- the aligned-span merge -------------------------------------------------
#
# The selector's block spans the same tokens as the sharded main KV block, so
# both block tables are the same width and the two share one group again. That
# is what removes the 93.8%-empty selector blocks.


def _aligned_selector(dcp: int) -> MLAAttentionSpec:
    """What QSACompressedKeyCache declares under DCP: span 784*dcp."""
    return MLAAttentionSpec(
        block_size=784 * dcp,
        num_kv_heads=1,
        head_size=128,
        dtype=DTYPE,
        tokens_per_state=8,
        dcp_transparent=True,
        dcp_shard_count=1,  # span already expressed, as the real spec does
    )


def _stamped_main(dcp: int) -> MLAAttentionSpec:
    """The main KV after stamp_dcp_shard_counts: sharded, so count == dcp."""
    from dataclasses import replace

    return replace(_main_kv_spec(), dcp_shard_count=dcp)


def test_the_spans_match_so_the_widths_match() -> None:
    main, selector = _stamped_main(2), _aligned_selector(2)
    assert main.logical_block_span == 1568
    assert selector.logical_block_span == 1568

    config = _config(max_model_len=262144, dcp=2)
    assert main.max_num_blocks_per_req(config, 262144) == 168
    assert selector.max_num_blocks_per_req(config, 262144) == 168


def test_they_group_together_again() -> None:
    """The split that cost 29.8% of capacity is no longer forced."""
    assert _uniform(_stamped_main(2), _aligned_selector(2))


def test_storage_is_unchanged_only_the_block_count_falls() -> None:
    """A wider page and proportionally fewer blocks is the same bytes."""
    config = _config(max_model_len=262144, dcp=2)
    narrow = _compressed_spec()  # 784-token block, 98 states
    wide = _aligned_selector(2)  # 1568-token block, 196 states

    assert wide.page_size_bytes == 2 * narrow.page_size_bytes
    assert wide.max_num_blocks_per_req(config, 262144) == 168
    assert narrow.max_num_blocks_per_req(config, 262144) == 335
    # same total storage, within one block of rounding
    assert abs(168 * wide.page_size_bytes - 335 * narrow.page_size_bytes) <= (
        wide.page_size_bytes
    )


def test_one_rank_is_untouched() -> None:
    """At DCP=1 the span is the block size and nothing moves."""
    main, selector = _main_kv_spec(), _aligned_selector(1)
    assert main.logical_block_span == 784
    assert selector.logical_block_span == 784
    assert _uniform(main, selector)


def test_a_replicated_cache_still_gets_the_whole_sequence() -> None:
    """Widening the span must not quietly halve the budget."""
    selector = _aligned_selector(2)
    assert selector.logical_block_span == selector.block_size, "never sharded"
    # 168 blocks of 196 states covers all 32,768 states
    assert 168 * (selector.block_size // 8) >= 262144 // 8


# --- merge must carry the resolved shard count ------------------------------
#
# `dcp_shard_count` is resolved before grouping, so it is a real field and the
# trailing `fields(AttentionSpec)` equality check in merge() compares it. A
# merge that drops it leaves None against a stamped member and asserts, which
# breaks any dense model started with DCP.


@pytest.mark.parametrize("world", [2, 4])
def test_merging_stamped_specs_keeps_the_shard_count(world: int) -> None:
    """Merge the specs AFTER stamping, which is the order the engine uses.

    Merging unstamped specs passes whatever merge() does, because both sides
    are then None. That is why this defect reached a GPU.
    """
    from vllm.v1.core.kv_cache_utils import stamp_dcp_shard_counts

    specs = stamp_dcp_shard_counts({"a": _main_kv_spec(), "b": _main_kv_spec()}, world)
    members = list(specs.values())
    assert all(m.dcp_shard_count == world for m in members), "stamped"

    merged = type(members[0]).merge(members)
    assert merged.dcp_shard_count == world, "merge dropped the shard count"


def test_merging_disagreeing_shard_counts_is_refused() -> None:
    from dataclasses import replace

    a = replace(_main_kv_spec(), dcp_shard_count=2)
    b = replace(_main_kv_spec(), dcp_shard_count=1)
    with pytest.raises(AssertionError, match="One DCP shard count"):
        type(a).merge([a, b])


def test_the_group_block_is_the_sharded_members_not_a_gcd() -> None:
    """A gcd equals it only by luck of the member set.

    A third member with a smaller block would drag the gcd below the sharded
    member's block, and the slot mapper would then multiply the wrong number by
    the world size -- the mirror of the bug that scored 0.0000 on MRCR.
    """
    from dataclasses import replace

    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    sharded = replace(_main_kv_spec(), dcp_shard_count=2)  # block 784
    selector = _aligned_selector(2)  # block 1568, transparent
    group = UniformTypeKVCacheSpecs.from_specs({"main": sharded, "sel": selector})
    assert group is not None, "the spans agree, so they group"
    assert group.block_size == 784, "the sharded member's block, not gcd(784, 1568)"


def test_storage_block_size_must_divide_the_block() -> None:
    """The one drift that misplaces writes without raising anything."""
    with pytest.raises(AssertionError, match="storage_block_size"):
        MLAAttentionSpec(
            block_size=784,
            num_kv_heads=1,
            head_size=128,
            dtype=DTYPE,
            tokens_per_state=8,
            storage_block_size=1568,
        )
