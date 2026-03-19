# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm.config import (
    CacheConfig,
    VllmConfig,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    CrossAttentionManager,
    FullAttentionManager,
    MambaManager,
    SingleTypeKVCacheManager,
    SinkFullAttentionManager,
    SlidingWindowManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_spec_registry import (
    _REGISTRY_KVCACHESPEC_LIST,
    KVCacheSpecRegistry,
    register_kv_cache_spec,
)


def make_vllm_config() -> VllmConfig:
    return VllmConfig(
        cache_config=CacheConfig(
            block_size=64,
            cache_dtype="bfloat16",
        ),
    )


# register all kvcache specs in enginecore process.
vllm_config = make_vllm_config()
register_all_kvcache_specs(vllm_config)


@dataclass(frozen=True)
class _TrulyUnregisteredSpec(KVCacheSpec):
    """
    A spec that inherits directly from KVCacheSpec with no registered
    ancestor in the MRO.  Used to test that the registry correctly raises
    when no entry can be found.
    """

    @property
    def page_size_bytes(self) -> int:
        return self.block_size * 128

    def max_memory_usage_bytes(self, _) -> int:
        return 0


spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
    SinkFullAttentionSpec: SinkFullAttentionManager,
}

base_args = dict(block_size=64, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
spec_args_map: dict[type[KVCacheSpec], dict[str, Any]] = {
    FullAttentionSpec: dict(
        block_size=64, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    ),
    MLAAttentionSpec: dict(
        block_size=64, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    ),
    SlidingWindowSpec: dict(
        block_size=64,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=128,
    ),
    ChunkedLocalAttentionSpec: dict(
        block_size=64,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        attention_chunk_size=4,
    ),
    MambaSpec: dict(
        block_size=64,
        shapes=((2, 512), (3, 32, 32)),
        dtypes=(torch.float32, torch.float32),
        mamba_cache_mode="align",
        num_speculative_blocks=2,
    ),
    CrossAttentionSpec: dict(
        block_size=64, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    ),
    SinkFullAttentionSpec: dict(
        block_size=64, num_kv_heads=8, head_size=128, dtype=torch.bfloat16, sink_len=16
    ),
}


class TestKVCacheSpecRegistry:
    """Test the core registry functionality."""

    def test_builtin_kvcache_specs_registered(self):
        for spec, manager in spec_manager_map.items():
            assert (
                KVCacheSpecRegistry.get_manager_class(spec(**(spec_args_map[spec])))
                is manager
            )
            assert len(_REGISTRY_KVCACHESPEC_LIST) == len(spec_manager_map)

    def test_custom_spec_register(self):
        """A decorated custom spec resolves to the declared manager."""

        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            uniform_type_base_spec=FullAttentionSpec,
        )
        @dataclass(frozen=True, kw_only=True)
        class _CustomFullSpec(FullAttentionSpec):
            custom_param: int = 16

        spec = _CustomFullSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
            custom_param=100,
        )

        assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
        assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is FullAttentionSpec

        with pytest.raises(
            AssertionError, match="Unexpected target_kv_cache_spec_cls:"
        ):

            @register_kv_cache_spec(
                manager_class=FullAttentionManager,
                uniform_type_base_spec=FullAttentionSpec,
                target_kv_cache_spec_cls=FullAttentionSpec,
            )
            @dataclass(frozen=True, kw_only=True)
            class _CustomFullSpecWithTargetSpec(FullAttentionSpec):
                custom_param: int = 16

        with pytest.raises(
            AssertionError, match="manager_class is required when override=False"
        ):

            @register_kv_cache_spec(
                uniform_type_base_spec=FullAttentionSpec,
            )
            @dataclass(frozen=True, kw_only=True)
            class _CustomFullSpecWithoutManager(FullAttentionSpec):
                custom_param: int = 16

    def test_custom_spec_override(self):
        """A decorated custom spec resolves to the declared manager."""

        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            uniform_type_base_spec=FullAttentionSpec,
            target_kv_cache_spec_cls=FullAttentionSpec,
            override=True,
        )
        @dataclass(frozen=True, kw_only=True)
        class _CustomOverrideFullSpec(FullAttentionSpec):
            custom_param: int = 16

        spec = _CustomOverrideFullSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
            custom_param=100,
        )

        assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
        assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is FullAttentionSpec
        assert (
            _REGISTRY_KVCACHESPEC_LIST[FullAttentionSpec].kvcache_spec_cls
            == _CustomOverrideFullSpec
        )
        with pytest.raises(
            AssertionError,
            match="Please specify a target_kv_cache_spec_cls when override",
        ):

            @register_kv_cache_spec(
                manager_class=FullAttentionManager,
                uniform_type_base_spec=FullAttentionSpec,
                override=True,
            )
            @dataclass(frozen=True, kw_only=True)
            class _CustomOverrideFullSpecWithNoTargetSpec(FullAttentionSpec):
                custom_param: int = 16

    def test_unregistered_spec_no_registered_parent_raises(self):
        """
        A spec whose entire MRO contains no registered class raises ValueError.
        Subclasses of registered specs intentionally *do not* raise — they
        inherit their parent's manager via MRO walking.
        """
        spec = _TrulyUnregisteredSpec(block_size=16)

        with pytest.raises(ValueError, match="No manager registered"):
            KVCacheSpecRegistry.get_manager_class(spec)

        with pytest.raises(ValueError, match="No uniform type base class"):
            KVCacheSpecRegistry.get_uniform_type_base_spec(spec)

    def test_unregistered_subclass_inherits_parent_manager(self):
        """
        An unregistered subclass of a registered spec resolves via MRO
        to its parent's manager — this is intentional registry behaviour.
        """

        @dataclass(frozen=True, kw_only=True)
        class _ImplicitlyInheritedSpec(FullAttentionSpec):
            pass

        spec = _ImplicitlyInheritedSpec(
            block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
        )

        # MRO walk finds FullAttentionSpec → FullAttentionManager
        assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
