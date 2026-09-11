# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheLayout

pytestmark = pytest.mark.skip_global_cleanup


class _BlockOuterBackend(AttentionBackend):
    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [64]

    @staticmethod
    def get_name():
        return "BLOCK_OUTER"


class _LayerOuterBackend:
    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBHNC,)


def _hybrid_config(
    *,
    layerwise: bool = True,
    skip_layers: list[str] | None = None,
    connector: bool = False,
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=64,
            user_specified_block_size=True,
            mamba_cache_mode="align",
            kv_cache_layout=None,
            kv_cache_dtype_skip_layers=skip_layers or [],
        ),
        model_config=SimpleNamespace(is_hybrid=True),
        kv_transfer_config=object() if connector else None,
        quant_config=SimpleNamespace(
            has_layerwise_kv_cache=lambda: layerwise,
        ),
    )


def test_layerwise_hybrid_keeps_backend_block_size(monkeypatch):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    config = _hybrid_config()

    with (
        patch.object(
            Platform, "_find_non_ssm_backend", return_value=_BlockOuterBackend
        ),
        patch.object(Platform, "_align_hybrid_block_size") as align_hybrid,
        patch.object(Platform, "_align_heterogeneous_kv_block_size") as align_mixed,
    ):
        Platform.update_block_size_for_backend(config)

    align_hybrid.assert_not_called()
    align_mixed.assert_not_called()
    assert config.cache_config.block_size == 64
    assert config.cache_config.mamba_block_size == 64


def test_layerwise_hybrid_uses_legacy_alignment_without_block_outer_layout(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    config = _hybrid_config()

    with (
        patch.object(
            Platform, "_find_non_ssm_backend", return_value=_LayerOuterBackend
        ),
        patch.object(Platform, "_align_hybrid_block_size") as align_hybrid,
    ):
        Platform.update_block_size_for_backend(config)

    align_hybrid.assert_called_once_with(config, _LayerOuterBackend)


def test_layerwise_hybrid_honors_connector_required_layer_outer_layout(monkeypatch):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    config = _hybrid_config(connector=True)

    with (
        patch.object(
            Platform, "_find_non_ssm_backend", return_value=_BlockOuterBackend
        ),
        patch.object(
            Platform,
            "_get_kv_connector_cache_layout",
            return_value="LBHNC",
        ),
        patch.object(Platform, "_align_hybrid_block_size") as align_hybrid,
    ):
        Platform.update_block_size_for_backend(config)

    align_hybrid.assert_called_once_with(config, _BlockOuterBackend)


def test_layerwise_hybrid_rejects_unsupported_manager_block_size(monkeypatch):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    config = _hybrid_config()
    config.cache_config.block_size = 48

    with (
        patch.object(
            Platform, "_find_non_ssm_backend", return_value=_BlockOuterBackend
        ),
        pytest.raises(
            ValueError,
            match="KV cache block size 48.*BLOCK_OUTER.*Omit --block-size",
        ),
    ):
        Platform.update_block_size_for_backend(config)


def test_layerwise_hybrid_packing_also_covers_skip_layers(monkeypatch):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    config = _hybrid_config(skip_layers=["0"])

    with (
        patch.object(
            Platform, "_find_non_ssm_backend", return_value=_BlockOuterBackend
        ),
        patch.object(Platform, "_align_hybrid_block_size") as align_hybrid,
        patch.object(Platform, "_align_heterogeneous_kv_block_size") as align_mixed,
    ):
        Platform.update_block_size_for_backend(config)

    align_hybrid.assert_not_called()
    align_mixed.assert_not_called()
