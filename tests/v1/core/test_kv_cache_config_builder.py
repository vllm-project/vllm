# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVCacheConfigBuilder resolution."""

from unittest.mock import MagicMock, patch

import pytest

from tests.v1.core.utils import full_layer, new_kv_cache_group, new_kv_cache_spec
from vllm.platforms import Platform
from vllm.v1.core.kv_cache_config_builder import (
    KVCacheConfigBuilder,
    get_kv_cache_config_builder,
)
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    get_kv_cache_groups_from_builder,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec


def _make_vllm_config(supports_grouping: bool = False) -> MagicMock:
    """Create a minimal mock VllmConfig for builder resolution tests.

    ``supports_grouping`` selects whether the resolved model class declares
    ``SupportsKVCacheGrouping`` (exposing a ``get_kv_cache_groups`` classmethod).
    """
    cfg = MagicMock()
    cfg.model_config.supports_kv_cache_grouping = supports_grouping
    cfg.attention_config.hisparse_config = None
    if supports_grouping:
        cfg.model_config.registry.resolve_model_cls.return_value = (
            _GroupedModel,
            "arch",
        )
    return cfg


CUSTOM_PATH = "tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
DEFAULT_PATH = "vllm.v1.core.kv_cache_planning.DefaultKVCacheConfigBuilder"


class CustomBuilder(DefaultKVCacheConfigBuilder):
    """A test builder subclass."""

    pass


class _GroupedModel:
    """A test model class with a ``get_kv_cache_groups`` classmethod."""

    @classmethod
    def get_kv_cache_groups(cls, vllm_config, layer_configs):
        return []


class _PartialGroupedModel:
    """A test model class whose grouping omits layers."""

    @classmethod
    def get_kv_cache_groups(cls, vllm_config, layer_configs):
        return [KVCacheGroupSpec.from_layers({"a": layer_configs["a"]})]


class TestPlatformHookResolution:
    """The platform hook selects only the placement builder."""

    def test_default_hook_returns_default_builder(self):
        cfg = _make_vllm_config()
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
        cfg = _make_vllm_config()
        assert type(get_kv_cache_config_builder(cfg)) is expected_cls

    @patch("vllm.platforms.current_platform")
    def test_resolves_fresh_instance_per_call(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        assert get_kv_cache_config_builder(cfg) is not get_kv_cache_config_builder(cfg)

    def test_public_interface_declares_two_placement_hooks(self):
        # get_kv_cache_configs lives in kv_cache_planning and is Core-owned;
        # the builder only declares the placement hooks.
        assert KVCacheConfigBuilder.__abstractmethods__ == {
            "get_kv_cache_groups",
            "get_kv_cache_config_from_groups",
        }
        assert issubclass(DefaultKVCacheConfigBuilder, KVCacheConfigBuilder)
        assert (
            "pool_bytes_per_block"
            not in DefaultKVCacheConfigBuilder.__abstractmethods__
        )
        assert (
            "max_memory_usage_bytes"
            not in DefaultKVCacheConfigBuilder.__abstractmethods__
        )


class TestModelGroupingPriority:
    """A model class with a get_kv_cache_groups classmethod owns grouping."""

    def test_model_classmethod_owns_grouping(self):
        cfg = _make_vllm_config(supports_grouping=True)
        assert DefaultKVCacheConfigBuilder().get_kv_cache_groups(cfg, {}) == []

    def test_model_grouping_rejects_hisparse(self):
        cfg = _make_vllm_config(supports_grouping=True)
        cfg.attention_config.hisparse_config = object()
        with pytest.raises(ValueError, match="HiSparse"):
            DefaultKVCacheConfigBuilder().get_kv_cache_groups(cfg, {})


class TestCoreChecksGrouping:
    """Core checks the grouping whether a model or a platform produced it."""

    _LAYER = full_layer(
        AttentionSpec(block_size=16, num_kv_heads=1, head_size=128, dtype="float16")
    )

    def test_model_grouping_is_checked(self):
        cfg = _make_vllm_config(supports_grouping=True)
        cfg.model_config.registry.resolve_model_cls.return_value = (
            _PartialGroupedModel,
            "arch",
        )
        with pytest.raises(AssertionError, match="miss layers"):
            get_kv_cache_groups_from_builder(
                DefaultKVCacheConfigBuilder(), cfg, {"a": self._LAYER, "b": self._LAYER}
            )

    @pytest.mark.parametrize(
        ("groups", "match"),
        [
            ([new_kv_cache_group(["a"], _LAYER)], "miss layers"),
            (
                [
                    new_kv_cache_group(["a", "b"], _LAYER),
                    new_kv_cache_group(["b"], _LAYER),
                ],
                "overlaps",
            ),
        ],
    )
    def test_platform_grouping_is_checked(self, groups, match):
        class FixedGroupsBuilder(DefaultKVCacheConfigBuilder):
            def get_kv_cache_groups(self, vllm_config, layer_configs):
                return groups

        with pytest.raises(AssertionError, match=match):
            get_kv_cache_groups_from_builder(
                FixedGroupsBuilder(),
                _make_vllm_config(),
                {"a": self._LAYER, "b": self._LAYER},
            )


class TestPlatformCustomPriority:
    """A vendor platform can override the hook to customize priority."""

    def test_platform_builder_wins_over_model_declaration(self):
        class PlatformFirstPlatform(Platform):
            @classmethod
            def get_kv_cache_config_builder_cls(cls, vllm_config):
                return CUSTOM_PATH

        cfg = _make_vllm_config()
        assert PlatformFirstPlatform.get_kv_cache_config_builder_cls(cfg) == CUSTOM_PATH
        with patch("vllm.platforms.current_platform", PlatformFirstPlatform):
            assert isinstance(get_kv_cache_config_builder(cfg), CustomBuilder)


class TestProjectionPreservesGroupMetadata:
    """PP projection must carry host_resident/role/enable_kv_transfer."""

    def test_partial_projection_preserves_metadata(self):
        from vllm.v1.core.kv_cache_planning import _project_kv_cache_groups_to_worker
        from vllm.v1.kv_cache_interface import KVCacheGroupRole

        layer = full_layer(
            AttentionSpec(block_size=16, num_kv_heads=1, head_size=128, dtype="float16")
        )
        group = new_kv_cache_group(
            ["layer1", "layer2", "layer3"],
            layer,
            host_resident=True,
            enable_kv_transfer=False,
            role=KVCacheGroupRole.HISPARSE_INDEXER,
        )

        projected = _project_kv_cache_groups_to_worker(
            [group], {"layer1": layer, "layer2": layer}
        )
        assert len(projected) == 1
        p = projected[0]
        assert p.host_resident is True
        assert p.enable_kv_transfer is False
        assert p.role is KVCacheGroupRole.HISPARSE_INDEXER
        assert p.layer_names == ["layer1", "layer2"]


class TestBytesPerBlockProbe:
    """The single-block probe must not be polluted by num_gpu_blocks_override."""

    def test_probe_suppresses_override(self):
        import torch

        from vllm.config import CacheConfig, ModelConfig, VllmConfig
        from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
        from vllm.v1.core.kv_cache_planning import (
            DefaultKVCacheConfigBuilder,
            _pool_bytes_per_block,
            _pool_bytes_per_block_from_builder,
        )

        spec = AttentionSpec(
            block_size=16, num_kv_heads=1, head_size=128, dtype=torch.float16
        )
        vc = VllmConfig(
            model_config=ModelConfig(max_model_len=100),
            cache_config=CacheConfig(block_size=16),
        )
        resolve_kv_cache_layout(vc, [["BLHNC"]], [full_layer(spec)])
        groups = [new_kv_cache_group([name], spec) for name in "ab"]

        pure_c = _pool_bytes_per_block(vc, groups)
        vc.cache_config.num_gpu_blocks_override = 500
        probed_c = _pool_bytes_per_block_from_builder(
            DefaultKVCacheConfigBuilder(), vc, groups
        )
        assert probed_c == pure_c
        assert vc.cache_config.num_gpu_blocks_override == 500

    @pytest.mark.parametrize("declares_pool_bytes", [False, True])
    def test_several_device_allocations_need_declared_pool_bytes(
        self, declares_pool_bytes
    ):
        """A placement with one allocation per group cannot be sized from one
        backing; the builder must declare its per-block bytes."""
        from vllm.v1.core.kv_cache_planning import _pool_bytes_per_block_from_builder
        from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

        class PerGroupAllocationBuilder(DefaultKVCacheConfigBuilder):
            def get_kv_cache_config_from_groups(self, vllm_config, groups, num_blocks):
                tensors = [
                    KVCacheTensor(
                        size=page * num_blocks,
                        layers=group.layer_names,
                        layer_stride=page * num_blocks,
                        block_stride=page,
                    )
                    for group, page in zip(groups, (64, 32))
                ]
                return KVCacheConfig(num_blocks, tensors, groups)

            def _get_pool_bytes_per_block(self, vllm_config, groups):
                return 64 + 32 if declares_pool_bytes else None

        vc = _make_vllm_config()
        vc.cache_config.num_gpu_blocks_override = None
        groups = [new_kv_cache_group([name], new_kv_cache_spec()) for name in "ab"]
        builder = PerGroupAllocationBuilder()
        if declares_pool_bytes:
            assert _pool_bytes_per_block_from_builder(builder, vc, groups) == 96
        else:
            with pytest.raises(ValueError, match="_get_pool_bytes_per_block"):
                _pool_bytes_per_block_from_builder(builder, vc, groups)
