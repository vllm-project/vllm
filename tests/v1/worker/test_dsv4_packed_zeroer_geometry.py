# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for the packed-DSV4 KV zeroer geometry.

Constructor-crossing regression for the packed-KV zeroing stride fix (upstream
PR #50276): every number below comes from instantiating the REAL production
code (no arithmetic is reimplemented here):

  - ``vllm.v1.core.kv_cache_utils._get_packed_kv_cache_layout``  (packed
    offsets / block stride)
  - ``vllm.v1.worker.gpu.attn_utils._reshape_attention_kv_cache``
    (per-layer column views)
  - ``vllm.models.deepseek_v4.sparse_mla.DeepseekV4FlashMLABackend``
    (``get_kv_cache_shape`` / ``get_kv_cache_block_dim``)
  - ``vllm.v1.worker.utils.AttentionGroup`` / ``KVBlockZeroer``
    (``KVBlockZeroer.__init__`` precomputes the segment tables)

CPU only: no Triton launch, no CUDA tensor, no CUDA call is made. The test
asserts the packed-DSV4 constructor metadata:

  * block stride   = the full packed row (4 aligned pages)
  * per-layer zero = the meaningful page (the spec's unpadded
                     ``real_page_size_bytes``), so adjacent packed layers are
                     never wiped
  * 4 packed layer views; block 99 of the highest-offset layer is the last
    written region and stays within the packed backing.

The pre-fix constructor exposed a five-field ``_meta`` with no block stride, so
its zero width equals the FULL packed row; the extraction below records that
legacy defect and then asserts the fixed semantics, so failure on the old code
is a semantic one (wrong zero width / out-of-bounds block 99), not a
schema-shape accident.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
from vllm.v1.core.kv_cache_utils import _get_packed_kv_cache_layout
from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

pytestmark = pytest.mark.cpu_test

NUM_BLOCKS = 100
BLOCK_SIZE_CFG = 256  # vllm_config.cache_config.block_size for DSV4 (group 0)
COMPRESS = 4  # main-MLA compress_ratio
STORAGE_BS = BLOCK_SIZE_CFG // COMPRESS  # 64
KERNEL_BS = 256  # DeepseekV4FlashMLABackend supports [256]
LAST_BLOCK = NUM_BLOCKS - 1  # block 99


def _extract_segments(meta, base):
    """Read (offset_bytes, block_stride_bytes, zero_span_bytes) per segment.

    Accepts both the pre-fix five-field ``_meta`` (no block stride; the zero
    width is the FULL packed row -- the defect) and the fixed six-field ``_meta``.
    All values come from the real constructor output; nothing is recomputed. Returns
    ``(segs, legacy_why)`` where ``legacy_why`` describes the pre-fix defect
    when detected.
    """
    legacy_why = None
    if len(meta) == 6:
        seg_addrs, seg_block_strides, seg_page_sizes, _max_chunks, _blk, n_segs = meta
    else:
        seg_addrs, seg_page_sizes, _max_chunks, _blk, n_segs = meta
        seg_block_strides = seg_page_sizes
        legacy_why = (
            f"legacy {len(meta)}-field _meta (no block stride): zero width is "
            f"the FULL packed row ({int(seg_page_sizes[0]) * 4} B) -- the defect"
        )
    segs = [
        (int(a) - base, int(bs) * 4, int(ps) * 4)
        for a, bs, ps in zip(seg_addrs, seg_block_strides, seg_page_sizes)
    ]
    return segs, legacy_why, n_segs


def test_dsv4_packed_zeroer_geometry():
    """Real DSV4 constructor metadata separates block stride from zero span."""
    layer_names = [f"model.layers.{i}.self_attn" for i in range(4)]

    specs = [
        MLAAttentionSpec(
            block_size=BLOCK_SIZE_CFG,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            compress_ratio=COMPRESS,
            cache_dtype_str="fp8_ds_mla",
            alignment=576,
            model_version="deepseek_v4",
        )
        for _ in layer_names
    ]
    uniform = UniformTypeKVCacheSpecs.from_specs(
        {ln: sp for ln, sp in zip(layer_names, specs)}
    )
    assert uniform is not None

    # --- Real packed layout planner -------------------------------------------
    block_stride, layers_by_offset = _get_packed_kv_cache_layout(
        [
            AttentionGroup(
                backend=DeepseekV4FlashMLABackend,
                layer_names=layer_names,
                kv_cache_spec=uniform,
                kv_cache_group_id=0,
            )
        ]
    )
    page_bytes = specs[0].page_size_bytes  # DSV4 fp8_ds_mla page after 576B alignment
    real_page_bytes = specs[0].real_page_size_bytes  # unpadded meaningful page
    alignment_gap = page_bytes - real_page_bytes  # trailing planner padding
    total_size = block_stride * NUM_BLOCKS
    offsets = sorted(layers_by_offset)
    assert block_stride == 4 * page_bytes, (block_stride, 4 * page_bytes)
    assert offsets == [i * page_bytes for i in range(4)]

    # --- Real per-layer views (packed branch) --------------------------------
    backing = torch.zeros(total_size, dtype=torch.uint8)
    base = backing.data_ptr()
    views = {}
    for ln, off in zip(layer_names, offsets):
        packing = (off, block_stride)
        kv_shape = DeepseekV4FlashMLABackend.get_kv_cache_shape(
            NUM_BLOCKS, STORAGE_BS, 1, 512, cache_dtype_str="fp8_ds_mla"
        )
        stride_order = tuple(range(len(kv_shape)))
        view = _reshape_attention_kv_cache(
            backing, specs[0], kv_shape, stride_order, NUM_BLOCKS, packing
        )
        views[ln] = view
        assert view.data_ptr() - base == off, "data_ptr must be base+offset"
        assert view.stride(0) * view.element_size() == block_stride, (
            "row stride == block_stride (logical block stride)"
        )

    # --- Real KVBlockZeroer.__init__ (production arithmetic, CPU) ----------
    sctx = {ln: SimpleNamespace(kv_cache=v) for ln, v in views.items()}
    zeroer = KVBlockZeroer(
        torch.device("cpu"),
        attn_groups_iter=[
            AttentionGroup(
                backend=DeepseekV4FlashMLABackend,
                layer_names=layer_names,
                kv_cache_spec=specs[0],
                kv_cache_group_id=0,
            )
        ],
        kernel_block_sizes=[KERNEL_BS],
        cache_dtype="fp8_ds_mla",
        static_forward_context=sctx,
        num_blocks=NUM_BLOCKS,
    )
    segs, legacy_why, n_segs = _extract_segments(zeroer._meta, base)

    # --- Fixed semantics (proven by metadata values, not by schema shape) ----
    assert n_segs == 4, "all 4 packed layers must register as segments"
    assert [off for off, _, _ in segs] == offsets, "one segment per layer offset"
    assert all(bs == block_stride for _, bs, _ in segs), (
        "block stride must be the full packed row (4 aligned pages)"
    )
    assert all(zs == real_page_bytes for _, _, zs in segs), (
        f"zero span must be the meaningful page ({real_page_bytes} B), got "
        f"{[zs for _, _, zs in segs]} B" + (f"; {legacy_why}" if legacy_why else "")
    )

    # --- Block 99 (the real final block) bounds, via constructor metadata ---
    # Production write for block b of a segment is
    #   [off + b*block_stride_bytes, off + b*block_stride_bytes + zero_span_bytes).
    worst = 0
    for off, bs, zs in segs:
        start = off + LAST_BLOCK * bs
        end = start + zs
        oob = end - total_size
        worst = max(worst, oob)
        assert end <= total_size, (
            f"segment@off={off} block {LAST_BLOCK} writes [{start}, {end}) "
            f"-> {oob} bytes past backing ({total_size})"
        )
    assert worst == 0, f"highest-offset final block goes {worst} bytes OOB"

    # Highest-offset segment's block 99 is the last written region and ends exactly at
    # total_size - alignment_gap: the trailing planner padding stays untouched.
    off_last, bs_last, zs_last = segs[-1]
    assert off_last + LAST_BLOCK * bs_last + zs_last == total_size - alignment_gap, (
        "highest-offset final block must end at total_size - alignment_gap "
        f"({alignment_gap}-byte gap intact)"
    )

    # Zero span < block stride proves adjacent packed layers are never wiped.
    assert max(zs for _, _, zs in segs) < block_stride
