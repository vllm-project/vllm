# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extensible KV cache over the standardized KV cache layout.

Covers the layout-derived buffer segmentation, the committed-prefix
narrowing used for KV-connector registration views, and VMM-backed
allocation through the shared allocate+reshape path.
"""

from dataclasses import dataclass

import pytest
import torch

from vllm.utils.extensible_tensor import (
    ExtensibleKVCacheBuffers,
    ExtensibleKVCacheBuilder,
    granule_size,
)
from vllm.utils.vmm_driver import HipVmmDriver, vmm_unavailable_reason
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    compute_layout_strides,
)
from vllm.v1.worker.gpu.attn_utils import narrow_kv_caches_to_num_blocks
from vllm.v1.worker.utils import allocate_kv_cache, kv_cache_num_segments

BLOCK_SIZE = 16
NUM_BLOCKS = 8
NUM_HEADS = 2
HEAD_SIZE = 32

requires_vmm = pytest.mark.skipif(
    vmm_unavailable_reason() is not None,
    reason=f"VMM unavailable: {vmm_unavailable_reason()}",
)


def _attn_spec(**kwargs) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
        **kwargs,
    )


@pytest.mark.parametrize(
    "layout,num_layers,expected",
    [
        (KVCacheLayout.LBHNC, 1, 1),
        (KVCacheLayout.LBHNC, 4, 4),
        (KVCacheLayout.LBNHC, 4, 4),
        (KVCacheLayout.BLHNC, 4, 1),
        (KVCacheLayout.BLNHC, 4, 1),
        (KVCacheLayout.BHLNC, 4, 1),
    ],
)
def test_kv_cache_num_segments_layouts(layout, num_layers, expected):
    """Segments = contiguous runs of blocks `block_stride` apart: one per
    layer region under layer-compact layouts, one for the whole buffer under
    block-outermost layouts."""
    config = _single_group_config(_attn_spec(), num_layers, layout)
    assert kv_cache_num_segments(config) == expected


def test_kv_cache_num_segments_separate_kv_head_groups():
    """The layout, not the spec packing, decides whether the K/V head groups
    sit inside the block (LBHNC, fused-AITER) or outside it as two planes
    (LHBNC, AITER sparse PA)."""
    spec = _attn_spec(
        num_head_slots=2,
        state_content_bytes=NUM_HEADS * HEAD_SIZE * 2,  # fp16 K (or V) per slot
    )
    assert spec.num_heads == 2
    assert kv_cache_num_segments(_single_group_config(spec, 3)) == 3
    assert (
        kv_cache_num_segments(_single_group_config(spec, 3, KVCacheLayout.LHBNC))
        == 3 * 2
    )


def test_kv_cache_num_segments_mamba():
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((NUM_HEADS, HEAD_SIZE),),
        dtypes=(torch.float32,),
    )
    assert kv_cache_num_segments(_single_group_config(spec, 2)) == 2
    assert (
        kv_cache_num_segments(_single_group_config(spec, 2, KVCacheLayout.BLHNC)) == 1
    )


def test_extensible_narrows_layouts_to_block_outermost():
    """Layer-compact regions of different page sizes have no uniform
    bytes-per-block stride, which the extensible commit model needs; the
    engine narrows the layout candidates to block-outermost ones, or the
    caller disables the extensible KV cache when none is supported."""
    from vllm.v1.attention.backends.utils import narrow_layouts_to_block_outermost

    assert narrow_layouts_to_block_outermost([["LBNHC", "BLHNC"]]) == [["BLHNC"]]
    # Legacy names alias layer-compact layouts and are dropped too.
    assert narrow_layouts_to_block_outermost([["NHD", "BLHNC"], ["BLHNC"]]) == [
        ["BLHNC"],
        ["BLHNC"],
    ]
    # A rank without any block-outermost candidate disables the feature.
    assert narrow_layouts_to_block_outermost([["LBNHC"], ["BLHNC"]]) is None


def _single_group_config(
    spec,
    num_layers: int = 1,
    layout: KVCacheLayout = KVCacheLayout.LBHNC,
) -> KVCacheConfig:
    layer_names = [f"layer.{i}" for i in range(num_layers)]
    layer_stride, block_stride, _, _, _ = compute_layout_strides(
        spec, NUM_BLOCKS, num_layers, layout
    )
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * NUM_BLOCKS * num_layers,
                layers=layer_names,
                layer_stride=layer_stride,
                block_stride=block_stride,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names, spec)],
    )


def _extensible_allocate(config, layout):
    """Drive the shared allocator through the extensible reserve hook."""
    builder = ExtensibleKVCacheBuilder(config.num_blocks, torch.device("cuda:0"))
    kv_caches = allocate_kv_cache(
        config, torch.device("cuda:0"), layout, reserve=builder.reserve
    )
    return kv_caches, builder.finish()


@dataclass
class _FakeGroup:
    kv_cache_spec: object
    layer_names: list[str]
    kv_cache_group_id: int = 0


@requires_vmm
@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC, KVCacheLayout.BLHNC])
def test_extensible_allocation_and_growth(layout):
    """Only one block committed at first; extend keeps base pointers."""
    spec = _attn_spec()
    config = _single_group_config(spec, num_layers=2, layout=layout)
    kv_caches, buffers = _extensible_allocate(config, layout)
    assert isinstance(buffers, ExtensibleKVCacheBuffers)
    assert buffers.num_blocks_committed == 1
    # Physical commit is granule-rounded per segment: one committed block
    # maps at most one granule in each segment.
    granule = granule_size(0)
    num_segments = 2 if layout is KVCacheLayout.LBHNC else 1
    assert 0 < buffers.physical_bytes <= num_segments * granule

    ptrs_before = {n: t.data_ptr() for n, t in kv_caches.items()}
    for name, view in kv_caches.items():
        # Views span the full declared capacity.
        assert view.shape[0] == NUM_BLOCKS
        # Committed prefix is writable.
        view[0].fill_(1.0)

    buffers.commit(NUM_BLOCKS)
    assert buffers.num_blocks_committed == NUM_BLOCKS
    for name, view in kv_caches.items():
        assert view.data_ptr() == ptrs_before[name]
        view[NUM_BLOCKS - 1].fill_(2.0)
        # Earlier contents survive the grow; new blocks were zeroed.
        assert view[0].eq(1.0).all()
    torch.accelerator.synchronize()
    buffers.free()


@requires_vmm
def test_commit_at_or_below_committed_preserves_data():
    """A non-growing commit must not touch the mapping, even with defragment.

    Elastic EP re-runs warmup (and hence `ensure_blocks`) while the KV cache
    holds live data; releasing there would silently discard it.
    """
    config = _single_group_config(_attn_spec())
    kv_caches, buffers = _extensible_allocate(config, KVCacheLayout.LBHNC)
    try:
        buffers.commit(NUM_BLOCKS)
        view = kv_caches["layer.0"]
        view.fill_(7.0)
        torch.accelerator.synchronize()

        buffers.ensure_blocks(1)
        buffers.commit(NUM_BLOCKS, defragment=True)
        assert buffers.num_blocks_committed == NUM_BLOCKS
        torch.accelerator.synchronize()
        assert view.eq(7.0).all()
    finally:
        buffers.free()


@requires_vmm
def test_extensible_release_and_recommit():
    """Sleep-style release keeps VA and views valid; recommit re-zeroes."""
    spec = _attn_spec()
    config = _single_group_config(spec)
    kv_caches, buffers = _extensible_allocate(config, KVCacheLayout.LBHNC)
    buffers.commit(NUM_BLOCKS)
    view = kv_caches["layer.0"]
    view.fill_(3.0)
    torch.accelerator.synchronize()

    buffers.release_physical()
    assert buffers.num_blocks_committed == 0
    buffers.recommit()
    assert buffers.num_blocks_committed == NUM_BLOCKS
    assert view.data_ptr() == kv_caches["layer.0"].data_ptr()
    assert view.eq(0).all()
    torch.accelerator.synchronize()
    buffers.free()


def test_narrow_kv_caches_to_num_blocks():
    """Connector views are trimmed to the committed logical block prefix."""
    spec = _attn_spec()
    committed = 3
    full = torch.zeros(NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE)
    groups = [_FakeGroup(spec, ["layer.0"])]
    narrowed = narrow_kv_caches_to_num_blocks(
        {"layer.0": full}, groups, [BLOCK_SIZE], committed
    )
    assert narrowed["layer.0"].shape[0] == committed
    assert narrowed["layer.0"].data_ptr() == full.data_ptr()


def test_narrow_applies_virtual_block_split():
    """Kernel-split caches have block_size/kernel ratio more physical blocks."""
    spec = _attn_spec()
    kernel_block_size = BLOCK_SIZE // 2
    physical_blocks = NUM_BLOCKS * 2
    full = torch.zeros(physical_blocks, NUM_HEADS, kernel_block_size, 2 * HEAD_SIZE)
    groups = [_FakeGroup(spec, ["layer.0"])]
    narrowed = narrow_kv_caches_to_num_blocks(
        {"layer.0": full}, groups, [kernel_block_size], 3
    )
    assert narrowed["layer.0"].shape[0] == 3 * 2


def test_narrow_skips_out_of_range_groups():
    spec = _attn_spec()
    full = torch.zeros(NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE)
    groups = [_FakeGroup(spec, ["layer.0"], kv_cache_group_id=1)]
    narrowed = narrow_kv_caches_to_num_blocks(
        {"layer.0": full}, groups, [BLOCK_SIZE], 3
    )
    assert narrowed["layer.0"] is full


@pytest.mark.parametrize(
    "version,expected",
    [
        (70253211, (7, 2)),  # legacy packaging line: defective
        (71160850, (7, 11)),  # TheRock preview: defective
        (71260850, (7, 12)),  # first runtime with ROCm/rocm-systems#2451
        (71460850, (7, 14)),  # first production TheRock release
        (100060850, (10, 0)),
    ],
)
def test_hip_runtime_version_gate(version, expected, monkeypatch):
    """hipRuntimeGetVersion decoding, and which runtimes are ruled out.

    The extensible KV cache is disabled below 7.12: earlier HIP runtimes fail
    hipMemSetAccess non-deterministically (ROCm/rocm-systems#2516).
    """
    driver = object.__new__(HipVmmDriver)
    monkeypatch.setattr(HipVmmDriver, "runtime_version", lambda self: expected)
    assert driver.runtime_version() == expected
    reason = driver.unusable_runtime_reason()
    if expected >= (7, 12):
        assert reason is None
    else:
        assert reason is not None and "2516" in reason
