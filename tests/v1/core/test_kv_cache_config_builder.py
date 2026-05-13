# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVCacheConfigBuilder resolution."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.core.kv_cache_config_builder import (
    KVCacheConfigBuilder,
    _load_builder,
    resolve_builder,
)


def _make_vllm_config(builder_cls_path: str | None = None) -> MagicMock:
    """Create a minimal mock VllmConfig for builder resolution tests."""
    cfg = MagicMock()
    cfg.model_config.kv_cache_config_builder_cls = builder_cls_path
    return cfg


class CustomBuilder(KVCacheConfigBuilder):
    """A test builder subclass."""

    pass


class TestLoadBuilder:
    def test_load_default_builder(self):
        builder = _load_builder(
            "vllm.v1.core.kv_cache_config_builder.KVCacheConfigBuilder"
        )
        assert isinstance(builder, KVCacheConfigBuilder)

    def test_load_nonexistent_raises(self):
        with pytest.raises((ImportError, AttributeError)):
            _load_builder("vllm.nonexistent.module.Builder")


class TestResolveBuilder:
    @patch("vllm.platforms.current_platform")
    def test_default_when_no_overrides(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = None
        cfg = _make_vllm_config(builder_cls_path=None)
        builder = resolve_builder(cfg)
        assert type(builder) is KVCacheConfigBuilder

    @patch("vllm.platforms.current_platform")
    def test_model_declared_builder(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = None
        cls_path = "tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
        cfg = _make_vllm_config(builder_cls_path=cls_path)
        builder = resolve_builder(cfg)
        assert isinstance(builder, CustomBuilder)

    @patch("vllm.platforms.current_platform")
    def test_platform_overrides_model(self, mock_platform):
        platform_path = "tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
        mock_platform.get_kv_cache_config_builder_cls.return_value = platform_path
        # Model declares default, but platform overrides
        cfg = _make_vllm_config(builder_cls_path=None)
        builder = resolve_builder(cfg)
        assert isinstance(builder, CustomBuilder)
