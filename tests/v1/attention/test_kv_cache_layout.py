# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the KV cache layout resolution contract.

The engine core resolves one layout per model and records it on
``CacheConfig.kv_cache_layout``; worker processes adopt it via
``record_kv_cache_layout``. ``None`` means unresolved; once set the value is
final; readers use ``CacheConfig.get_resolved_kv_cache_layout``, which raises
on ``None``.
"""

from math import prod

import pytest
import torch

from vllm.config import CacheConfig
from vllm.utils.torch_utils import is_non_overlapping_and_dense
from vllm.v1.attention.backends.utils import (
    get_flashinfer_layout_string,
    record_kv_cache_layout,
    resolve_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheLayout,
    KVCacheTensor,
    compute_layer_kv_cache_shape_bytes,
    compute_layout_strides,
    create_kv_cache_views,
)


def test_get_raises_before_resolution():
    with pytest.raises(ValueError, match="has not been resolved"):
        CacheConfig().get_resolved_kv_cache_layout()


def test_record_and_get_round_trip():
    cache_config = CacheConfig()
    record_kv_cache_layout(cache_config, "BLHNC")
    assert cache_config.kv_cache_layout == "BLHNC"
    assert cache_config.get_resolved_kv_cache_layout() is KVCacheLayout.BLHNC


@pytest.mark.parametrize(
    ("alias", "expected"),
    [("NHD", KVCacheLayout.LBNHC), ("HND", KVCacheLayout.LBHNC)],
)
def test_record_normalizes_legacy_aliases(alias: str, expected: KVCacheLayout):
    cache_config = CacheConfig()
    record_kv_cache_layout(cache_config, alias)
    assert cache_config.kv_cache_layout == expected.name
    assert cache_config.get_resolved_kv_cache_layout() is expected


def test_record_is_write_once():
    cache_config = CacheConfig()
    record_kv_cache_layout(cache_config, "LBHNC")
    # Re-recording the same layout (elastic scale-up re-delivery) is fine,
    # including through an alias.
    record_kv_cache_layout(cache_config, "LBHNC")
    record_kv_cache_layout(cache_config, "HND")
    with pytest.raises(ValueError, match="already resolved"):
        record_kv_cache_layout(cache_config, "LBNHC")


def test_record_rejects_unknown_layout():
    with pytest.raises(ValueError, match="Unknown KV cache layout"):
        record_kv_cache_layout(CacheConfig(), "BOGUS")


def test_resolve_records_on_cache_config():
    cache_config = CacheConfig()
    layout = resolve_kv_cache_layout(cache_config, [["LBHNC", "LBNHC"]])
    assert layout is KVCacheLayout.LBHNC
    assert cache_config.kv_cache_layout == "LBHNC"


def test_resolve_honors_preset():
    cache_config = CacheConfig()
    cache_config.kv_cache_layout = "BHLNC"
    # The preset wins even against the supported sets.
    assert resolve_kv_cache_layout(cache_config, [["LBNHC"]]) is KVCacheLayout.BHLNC


def test_resolve_rejects_disagreeing_workers():
    with pytest.raises(AssertionError, match="disagree"):
        resolve_kv_cache_layout(CacheConfig(), [["LBNHC"], ["LBHNC"]])


def test_resolve_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    cache_config = CacheConfig()
    layout = resolve_kv_cache_layout(cache_config, [["LBNHC", "LBHNC"]])
    assert layout is KVCacheLayout.LBHNC

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "BLHNC")
    with pytest.raises(ValueError, match="does not satisfy every"):
        resolve_kv_cache_layout(CacheConfig(), [["LBNHC", "LBHNC"]])


def test_resolve_mixed_hnc_shapes_need_block_compact():
    def spec(num_kv_heads: int) -> FullAttentionSpec:
        return FullAttentionSpec(
            block_size=16,
            num_kv_heads=num_kv_heads,
            head_size=64,
            dtype=torch.float16,
        )

    mixed = [spec(2), spec(4)]
    layout = resolve_kv_cache_layout(CacheConfig(), [["LHBNC", "LBNHC"]], mixed)
    assert layout is KVCacheLayout.LBNHC

    with pytest.raises(ValueError, match="block-compact"):
        resolve_kv_cache_layout(CacheConfig(), [["LHBNC"]], mixed)

    # A uniform HNC shape leaves the preference order untouched.
    layout = resolve_kv_cache_layout(CacheConfig(), [["LHBNC", "LBNHC"]], [spec(2)])
    assert layout is KVCacheLayout.LHBNC


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        (KVCacheLayout.LBNHC, "NHD"),
        (KVCacheLayout.LBHNC, "HND"),
        (KVCacheLayout.BLNHC, "NHD"),
        (KVCacheLayout.BLHNC, "HND"),
        (KVCacheLayout.BHLNC, "HND"),
    ],
)
def test_flashinfer_layout_string(layout: KVCacheLayout, expected: str):
    assert get_flashinfer_layout_string(layout) == expected


def test_flashinfer_layout_string_rejects_lhbnc():
    with pytest.raises(AssertionError):
        get_flashinfer_layout_string(KVCacheLayout.LHBNC)


def test_block_density_matches_layout_compactness():
    """Connectors derive their region split from strides instead of the layout:
    block 0 of a per-layer view is one dense byte run exactly for block-compact
    layouts, and each per-head sub-view is dense otherwise."""
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=4, head_size=32, dtype=torch.float16
    )
    num_blocks, num_layers = 8, 3
    for layout in KVCacheLayout:
        shape = compute_layer_kv_cache_shape_bytes(spec, num_blocks)
        strides = compute_layout_strides(spec, num_blocks, num_layers, layout)
        tensor = KVCacheTensor(
            size=prod(shape) * num_layers,
            layers=[str(i) for i in range(num_layers)],
            layer_stride=strides[0],
            block_stride=strides[1],
        )
        raw = torch.zeros(tensor.size, dtype=torch.int8)
        views = create_kv_cache_views(raw, spec, num_blocks, layout, tensor)
        for view in views:
            assert is_non_overlapping_and_dense(view[0]) == layout.is_block_compact
            if not layout.is_block_compact:
                for head in range(view.shape[1]):
                    assert is_non_overlapping_and_dense(view[:, head][0])
