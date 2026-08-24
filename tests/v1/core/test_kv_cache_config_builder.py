# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVCacheConfigBuilder resolution."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import Platform
from vllm.v1.core.kv_cache_config_builder import (
    KVCacheConfigBuilder,
    get_kv_cache_config_builder,
    get_profiling_kv_cache_config,
)
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)


def _make_vllm_config(builder_cls_path: str | None = None) -> MagicMock:
    """Create a minimal mock VllmConfig for builder resolution tests."""
    cfg = MagicMock()
    cfg.model_config.kv_cache_config_builder_cls = builder_cls_path
    return cfg


CUSTOM_PATH = "tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
DEFAULT_PATH = "vllm.v1.core.kv_cache_planning.DefaultKVCacheConfigBuilder"


class CustomBuilder(DefaultKVCacheConfigBuilder):
    """A test builder subclass."""

    pass


class ExactBlocksBuilder(DefaultKVCacheConfigBuilder):
    """Record the capacity passed through the profiling path."""

    seen_num_blocks: int | None = None

    def get_kv_cache_groups(self, vllm_config, kv_cache_spec):
        return []

    def get_kv_cache_config_from_groups(self, vllm_config, kv_cache_groups, num_blocks):
        type(self).seen_num_blocks = num_blocks
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )


@pytest.fixture(autouse=True)
def _reset_seen_num_blocks():
    ExactBlocksBuilder.seen_num_blocks = None
    yield


class TestPlatformHookResolution:
    """The platform hook owns the resolution priority."""

    def test_default_hook_prefers_model_declaration(self):
        cfg = _make_vllm_config(builder_cls_path=CUSTOM_PATH)
        assert Platform.get_kv_cache_config_builder_cls(cfg) == CUSTOM_PATH

    def test_default_hook_falls_back_to_default_builder(self):
        cfg = _make_vllm_config(builder_cls_path=None)
        assert Platform.get_kv_cache_config_builder_cls(cfg) == DEFAULT_PATH


class TestBuilderResolution:
    @patch("vllm.platforms.current_platform")
    @pytest.mark.parametrize(
        ("builder_cls_path", "expected_cls"),
        [
            (None, DefaultKVCacheConfigBuilder),
            (CUSTOM_PATH, CustomBuilder),
        ],
    )
    def test_resolves_hook_selected_builder(
        self, mock_platform, builder_cls_path, expected_cls
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = (
            builder_cls_path or DEFAULT_PATH
        )
        cfg = _make_vllm_config(builder_cls_path)
        assert type(get_kv_cache_config_builder(cfg)) is expected_cls

    @patch("vllm.platforms.current_platform")
    def test_resolves_fresh_instance_per_call(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        assert get_kv_cache_config_builder(cfg) is not get_kv_cache_config_builder(cfg)

    def test_public_interface_declares_entry_point_and_two_hooks(self):
        assert KVCacheConfigBuilder.__abstractmethods__ == {
            "get_kv_cache_configs",
            "get_kv_cache_groups",
            "get_kv_cache_config_from_groups",
        }
        assert issubclass(DefaultKVCacheConfigBuilder, KVCacheConfigBuilder)

    def test_pool_bytes_per_block_derived_from_unit_placement(self):
        class PlacementBuilder(DefaultKVCacheConfigBuilder):
            def get_kv_cache_groups(self, vllm_config, kv_cache_spec):
                return [KVCacheGroupSpec(["layer"], MagicMock())]

            def get_kv_cache_config_from_groups(
                self, vllm_config, kv_cache_groups, num_blocks
            ):
                return KVCacheConfig(
                    num_blocks=num_blocks,
                    kv_cache_tensors=[
                        KVCacheTensor(
                            size=64 * num_blocks,
                            layers=["layer"],
                            layer_stride=64 * num_blocks,
                            block_stride=64,
                        )
                    ],
                    kv_cache_groups=kv_cache_groups,
                )

        cfg = _make_vllm_config()
        builder = PlacementBuilder()
        groups = builder.get_kv_cache_groups(cfg, {})
        assert builder._get_pool_bytes_per_block(cfg, groups) == 64


class TestPlatformCustomPriority:
    """A vendor platform can override the hook to customize priority."""

    def test_platform_builder_wins_over_model_declaration(self):
        class PlatformFirstPlatform(Platform):
            @classmethod
            def get_kv_cache_config_builder_cls(cls, vllm_config):
                return CUSTOM_PATH

        cfg = _make_vllm_config(builder_cls_path=CUSTOM_PATH)
        # Model declares CustomBuilder too; the platform forces it anyway.
        assert PlatformFirstPlatform.get_kv_cache_config_builder_cls(cfg) == CUSTOM_PATH
        with patch("vllm.platforms.current_platform", PlatformFirstPlatform):
            assert isinstance(get_kv_cache_config_builder(cfg), CustomBuilder)


class TestProfiling:
    @patch("vllm.platforms.current_platform")
    def test_profiling_reuses_exact_block_materializer(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = (
            "tests.v1.core.test_kv_cache_config_builder.ExactBlocksBuilder"
        )
        cfg = _make_vllm_config()
        cfg.cache_config.num_gpu_blocks_override = 11

        result = get_profiling_kv_cache_config(cfg, {}, min_blocks=7)

        assert result.num_blocks == 7
        assert ExactBlocksBuilder.seen_num_blocks == 7
        assert cfg.cache_config.num_gpu_blocks_override == 11
