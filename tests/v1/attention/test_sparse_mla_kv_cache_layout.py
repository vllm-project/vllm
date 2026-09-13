# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the sparse MLA backends' KV cache layout declarations.

The V3.2-style sparse MLA backends consume the bound cache as per-layer
[B, N, C] rows (MLAAttention.bind_kv_cache asserts the row stride stays
divisible by the row width). Block-outermost layouts break that view, but
no backend declared it, so an explicit VLLM_KV_CACHE_LAYOUT=BLHNC passed
resolution and the engine crash-looped at worker init (issue #55431).
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
    FlashInferMLASparseTRTLLMBackend,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
    FlashInferMLASparseSM90Backend,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    DeepseekV32IndexerBackend,
    KpoolTailBackend,
)
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

V32_SPARSE_BACKENDS = (
    FlashMLASparseBackend,
    FlashInferMLASparseTRTLLMBackend,
    FlashInferMLASparseSM120Backend,
    DeepseekV32IndexerBackend,
)

BLOCK_OUTERMOST = {
    KVCacheLayout.BLHNC,
    KVCacheLayout.BLNHC,
    KVCacheLayout.BHLNC,
}

pytestmark = pytest.mark.skip_global_cleanup


def _mixed_page_specs() -> list[MLAAttentionSpec]:
    # A V3.2-style model's latent and indexer caches differ in page size,
    # which narrows resolution to block-compact layouts.
    return [
        MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
        ),
        MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
        ),
    ]


def _resolve(backends, specs, monkeypatch, requested=None) -> KVCacheLayout:
    supported = [layout.name for layout in get_supported_kv_cache_layouts(backends)]
    monkeypatch.setattr(envs, "VLLM_KV_CACHE_LAYOUT", requested)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_layout=None),
        kv_transfer_config=None,
    )
    return resolve_kv_cache_layout(vllm_config, [supported], specs)


@pytest.mark.parametrize("backend", V32_SPARSE_BACKENDS)
def test_backends_declare_only_layer_compact_layouts(backend):
    declared = backend.supported_kv_cache_layouts()
    assert declared is not None
    assert not BLOCK_OUTERMOST.intersection(declared)
    assert all(layout.is_layer_compact for layout in declared)


def test_combined_supported_set_excludes_block_outermost():
    supported = get_supported_kv_cache_layouts(list(V32_SPARSE_BACKENDS))
    assert not BLOCK_OUTERMOST.intersection(supported)
    assert supported[0] is KVCacheLayout.LBNHC


def test_explicit_blhnc_fails_fast(monkeypatch):
    with pytest.raises(ValueError, match="BLHNC"):
        _resolve(
            [FlashInferMLASparseTRTLLMBackend, DeepseekV32IndexerBackend],
            _mixed_page_specs(),
            monkeypatch,
            requested="BLHNC",
        )


def test_default_resolution_unchanged(monkeypatch):
    layout = _resolve(
        [FlashInferMLASparseTRTLLMBackend, DeepseekV32IndexerBackend],
        _mixed_page_specs(),
        monkeypatch,
    )
    assert layout is KVCacheLayout.LBNHC


@pytest.mark.parametrize("name", ["LBNHC", "LBHNC"])
def test_explicit_layer_compact_layout_accepted(monkeypatch, name):
    layout = _resolve(
        [FlashInferMLASparseTRTLLMBackend, DeepseekV32IndexerBackend],
        _mixed_page_specs(),
        monkeypatch,
        requested=name,
    )
    assert layout.name == name


def test_sm90_pairing_still_resolves_lbhnc():
    supported = get_supported_kv_cache_layouts(
        [FlashInferMLASparseSM90Backend, DeepseekV32IndexerBackend]
    )
    assert supported == [KVCacheLayout.LBHNC]


def test_packing_layout_backends_unaffected():
    assert DeepseekV4IndexerBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.BLHNC,
        KVCacheLayout.BLNHC,
    )
    assert KpoolTailBackend.supported_kv_cache_layouts() == (KVCacheLayout.LBHNC,)
