# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for kernel re-paging of the kpool indexer and sparse MLA caches
under packed layouts: builder-side page tables and kernel-side views agree."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import kpool_page_geometry


def _flat_page_view():
    # The kpool op module is only importable after the GLM attention module
    # (they import each other); the engine always loads the model first.
    import vllm.models.glm5next.nvidia.attention  # noqa: F401
    from vllm.model_executor.layers.sparse_attn_indexer_kpool import (
        _kpool_flat_page_view,
    )

    return _kpool_flat_page_view


ROW = 132  # 128 fp8 bytes + 4 scale bytes per indexer state


def test_geometry_passthrough_for_kernel_sized_blocks():
    assert kpool_page_geometry(64, 13_000_000, ROW) == (64, 1, 1)
    assert kpool_page_geometry(32, None, ROW) == (32, 1, 1)
    # DeepSeek-V4 C128A: 2 states per block; DeepGEMM takes that block as is.
    assert kpool_page_geometry(2, 5_000_000, ROW) == (2, 1, 1)


def test_geometry_dense_and_packed():
    # 256 states per block (1024 tokens / kpool 4): four 64-state pages.
    assert kpool_page_geometry(256, None, ROW) == (64, 4, 4)
    assert kpool_page_geometry(256, 256 * ROW, ROW) == (64, 4, 4)
    # Packed: 9 indexer pages' worth of other layers between blocks.
    stride = 13 * 64 * ROW
    assert kpool_page_geometry(256, stride, ROW) == (64, 4, 13)
    # 288 states (1152 tokens) only tile 32-state pages.
    assert kpool_page_geometry(288, None, ROW) == (32, 9, 9)


def test_flat_page_view_addresses_each_block_page_once():
    num_blocks, num_states = 5, 256
    stride_pages = 13
    page_bytes = 64 * ROW
    raw = torch.arange(num_blocks * stride_pages * page_bytes, dtype=torch.int32)
    raw = (raw % 251).to(torch.uint8)
    blocked = raw.as_strided(
        (num_blocks, num_states, ROW), (stride_pages * page_bytes, ROW, 1)
    )

    pages = _flat_page_view()(blocked)
    assert pages.shape == ((num_blocks - 1) * stride_pages + 4, 64, ROW)
    for b in range(num_blocks):
        for j in range(4):
            torch.testing.assert_close(
                pages[b * stride_pages + j], blocked[b, j * 64 : (j + 1) * 64]
            )
    # Writes through the page view land in the block view (same storage).
    pages[2 * stride_pages + 3, 5, 7] = 200
    assert blocked[2, 3 * 64 + 5, 7] == 200


def test_flat_page_view_identity_without_repaging():
    view = _flat_page_view()
    kv = torch.zeros(3, 64, ROW, dtype=torch.uint8)
    assert view(kv) is kv
    placeholder = torch.tensor([], dtype=torch.uint8)
    assert view(placeholder) is placeholder


def test_sparse_mla_kernel_paged_view_repages_large_blocks():
    """Row ``r`` of the flat view lands at page ``r // page``, offset ``r % page``."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        _kernel_paged_view,
    )
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

    num_blocks, block_size, head_dim = 3, 256, 8
    page = 32  # FlashInferMLASparseTRTLLMBackend.get_kernel_page_rows()
    stride_rows = 320  # 64 rows of other layers' pages between blocks
    raw = (torch.arange(num_blocks * stride_rows * head_dim) % 253).to(torch.uint8)
    cache = raw.as_strided(
        (num_blocks, block_size, head_dim), (stride_rows * head_dim, head_dim, 1)
    )
    rows, block_stride_rows = flat_kv_row_view(cache, block_size)
    assert block_stride_rows == stride_rows
    paged = _kernel_paged_view(cache, rows, block_size, block_stride_rows, page)
    assert paged.shape[1:] == (1, page, head_dim)
    for b in range(num_blocks):
        for off in (0, 31, 32, 63, 64, 255):
            r = b * block_stride_rows + off
            torch.testing.assert_close(paged[r // page, 0, r % page], cache[b, off])
    # Dense kernel-sized blocks are passed through untouched ...
    small = torch.zeros(4, 64, head_dim, dtype=torch.uint8)
    rows64, stride64 = flat_kv_row_view(small, 64)
    assert stride64 == 64
    assert _kernel_paged_view(small, rows64, 64, stride64, page).shape == (
        4,
        1,
        64,
        head_dim,
    )
    # ... but a 64-row block strided apart by other layers' pages is re-paged.
    strided = raw.as_strided(
        (num_blocks, 64, head_dim), (stride_rows * head_dim, head_dim, 1)
    )
    rows_s, stride_s = flat_kv_row_view(strided, 64)
    paged_s = _kernel_paged_view(strided, rows_s, 64, stride_s, page)
    assert paged_s.shape[1:] == (1, page, head_dim)
    for b in range(num_blocks):
        r = b * stride_s + 40
        torch.testing.assert_close(paged_s[r // page, 0, r % page], strided[b, 40])


def test_layout_alignment_rule_matches_kernel_paging():
    """The layout aligns block strides with the pages the kernels re-page with."""
    from vllm.v1.core.kv_cache_utils import _kernel_page_rows
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    for block_size, kpool in (
        (1024, 4),
        (1152, 4),
        (256, 4),
        (64, 1),
        (4096, 16),
        (256, 128),
    ):
        idx = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            tokens_per_state=kpool,
        )
        assert (
            _kernel_page_rows(idx) == kpool_page_geometry(idx.num_states, None, ROW)[0]
        )
        mla = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            kernel_page_rows=32,
        )
        assert _kernel_page_rows(mla) == (block_size if block_size <= 64 else 32)
        # A backend that takes any block (kernel_page_rows None) needs no alignment.
        assert (
            _kernel_page_rows(
                MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.bfloat16,
                )
            )
            == block_size
        )


@pytest.mark.parametrize("family,expected_page", [(90, 32), (100, 32), (120, 64)])
def test_indexer_block_alignment_matches_deepgemm_page_support(
    monkeypatch, family, expected_page
):
    """DeepGEMM's fp8 paged MQA takes 32-state pages on SM90/SM100 but only
    64 on SM120, so the alignment must make the block re-page accordingly."""
    from vllm.platforms.cuda import CudaPlatformBase

    kpool = 4
    monkeypatch.setattr(
        CudaPlatformBase,
        "is_device_capability_family",
        classmethod(lambda cls, fam: fam == family),
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_kpool=kpool))
    )
    alignment = CudaPlatformBase._get_indexer_block_alignment(cfg)
    assert alignment == kpool * expected_page
    # A 1152-token hybrid block rounded up to the alignment re-pages cleanly.
    block = -(-1152 // alignment) * alignment
    page_states, _, _ = kpool_page_geometry(block // kpool, None, 132)
    assert page_states == expected_page
