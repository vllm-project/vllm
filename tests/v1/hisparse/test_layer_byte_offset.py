# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Each HiSparse layer must get its own region of the shared KV slab.

``HiSparseHotSpec`` and ``HiSparseResidentSpec`` both report
``has_layer_views = False``, so ``allocate_kv_cache`` hands every layer in a
group the whole backing tensor instead of a per-layer view. The
``l * layer_stride`` term of the documented layout
(``offset + l * layer_stride + b * block_stride``) therefore has to be applied
where the caches are bound.

Dropping that term is silent: every layer still gets a well-formed, in-bounds
view, and the resolver, the DMA and the attention kernel all behave correctly
on the rows they are handed. The layers simply share one region, so each one
reads whichever neighbour wrote last. The e2e suite cannot see it either --
it shrinks the model to 8 layers, which fits in a single hot group, and a
one-layer group has nothing to collide with.

These tests are pure Python on purpose: no GPU, no AITER, no model. The fault
is in platform-neutral layout arithmetic, so the guard has to run everywhere.
"""

import pytest

from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheTensor,
)

PAGE = 18432
NUM_LAYERS = 8
# One block holds a page for each layer plus the indexer, matching the GLM
# layout this was found on: block_stride == 9 * page for 8 layers.
BLOCK_STRIDE = PAGE * (NUM_LAYERS + 1)


def _hot_tensor(offset: int = 0) -> KVCacheTensor:
    return KVCacheTensor(
        size=BLOCK_STRIDE * 4,
        layers=[f"layer.{i}.hot" for i in range(NUM_LAYERS)],
        layer_stride=PAGE,
        block_stride=BLOCK_STRIDE,
        offset=offset,
    )


def _layer_byte_offset(tensor: KVCacheTensor, name: str) -> int:
    from vllm.v1.hisparse.binding import _hisparse_layer_byte_offset

    return _hisparse_layer_byte_offset(tensor, name)


@pytest.mark.parametrize("base", [0, PAGE * 32])
def test_each_layer_gets_a_distinct_offset(base: int) -> None:
    """The whole point: no two layers may start at the same byte."""
    tensor = _hot_tensor(base)
    offsets = [_layer_byte_offset(tensor, name) for name in tensor.layers]

    assert offsets == [base + i * PAGE for i in range(NUM_LAYERS)]
    assert len(set(offsets)) == NUM_LAYERS


def test_layer_regions_do_not_overlap_within_a_block() -> None:
    """Consecutive layers are exactly one page apart and stay inside a block."""
    tensor = _hot_tensor()
    offsets = [_layer_byte_offset(tensor, name) for name in tensor.layers]

    for previous, current in zip(offsets, offsets[1:]):
        assert current - previous == PAGE
    assert offsets[-1] + PAGE <= BLOCK_STRIDE


def test_single_layer_group_is_unchanged() -> None:
    """A one-layer group must keep the tensor offset verbatim.

    This is the shape the e2e test produces, and why it passes either way.
    """
    tensor = KVCacheTensor(
        size=PAGE * 4,
        layers=["only.hot"],
        layer_stride=PAGE,
        block_stride=PAGE,
        offset=PAGE * 7,
    )
    assert _layer_byte_offset(tensor, "only.hot") == PAGE * 7


def test_hisparse_specs_have_no_layer_views() -> None:
    """The precondition that makes the manual offset necessary.

    If either spec ever gains per-layer views, ``allocate_kv_cache`` applies
    the layer stride itself and applying it again here would double-count.
    """
    hot = HiSparseHotSpec(block_size=16, page_size=PAGE, blocks_per_request=4)
    resident = HiSparseResidentSpec(block_size=16, page_size=PAGE)

    assert not hot.has_layer_views
    assert not resident.has_layer_views
