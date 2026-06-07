# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
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
    HiddenStateCacheSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    TQFullAttentionSpec,
    UniformTypeKVCacheSpecs,
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
        device_config=DeviceConfig(device="cpu"),
    )


vllm_config = make_vllm_config()
register_all_kvcache_specs(vllm_config)


@pytest.fixture(autouse=True)
def restore_kv_cache_spec_registry():
    registry = _REGISTRY_KVCACHESPEC_LIST.copy()
    yield
    _REGISTRY_KVCACHESPEC_LIST.clear()
    _REGISTRY_KVCACHESPEC_LIST.update(registry)


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
    TQFullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    HiddenStateCacheSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    SlidingWindowMLASpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
    SinkFullAttentionSpec: SinkFullAttentionManager,
}

spec_uniform_base_map: dict[type[KVCacheSpec], type[KVCacheSpec]] = {
    FullAttentionSpec: FullAttentionSpec,
    TQFullAttentionSpec: FullAttentionSpec,
    MLAAttentionSpec: FullAttentionSpec,
    HiddenStateCacheSpec: FullAttentionSpec,
    SlidingWindowSpec: SlidingWindowSpec,
    SlidingWindowMLASpec: SlidingWindowMLASpec,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionSpec,
    MambaSpec: MambaSpec,
    CrossAttentionSpec: CrossAttentionSpec,
    SinkFullAttentionSpec: FullAttentionSpec,
}

spec_args_map: dict[type[KVCacheSpec], dict[str, Any]] = {
    FullAttentionSpec: dict(
        block_size=64, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    ),
    TQFullAttentionSpec: dict(
        block_size=64,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        tq_slot_size=256,
    ),
    MLAAttentionSpec: dict(
        block_size=64, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    ),
    HiddenStateCacheSpec: dict(
        block_size=64, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    ),
    SlidingWindowSpec: dict(
        block_size=64,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=128,
    ),
    SlidingWindowMLASpec: dict(
        block_size=64,
        num_kv_heads=1,
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


def make_spec(spec_cls: type[KVCacheSpec]) -> KVCacheSpec:
    return spec_cls(**spec_args_map[spec_cls])


def are_uniform_specs(*specs: KVCacheSpec) -> bool:
    return UniformTypeKVCacheSpecs.is_uniform_type(
        {f"layer_{i}": spec for i, spec in enumerate(specs)}
    )


class TestKVCacheSpecRegistry:
    """Test the core registry functionality."""

    def test_builtin_kvcache_specs_registered(self):
        assert set(spec_manager_map) <= set(_REGISTRY_KVCACHESPEC_LIST)
        for spec_cls, manager in spec_manager_map.items():
            spec = make_spec(spec_cls)
            assert KVCacheSpecRegistry.get_manager_class(spec) is manager
            assert (
                KVCacheSpecRegistry.get_uniform_type_base_spec(spec)
                is spec_uniform_base_map[spec_cls]
            )

    @pytest.mark.parametrize("spec_cls", list(spec_manager_map))
    def test_custom_spec_register(self, spec_cls):
        """A decorated custom spec resolves to the declared manager."""
        manager = spec_manager_map[spec_cls]
        uniform_base_spec = spec_uniform_base_map[spec_cls]

        @register_kv_cache_spec(
            manager_class=manager,
            uniform_type_base_spec=uniform_base_spec,
        )
        @dataclass(frozen=True, kw_only=True)
        class _CustomSpec(spec_cls):  # type: ignore[valid-type,misc]
            custom_param: int = 16

        spec = _CustomSpec(**spec_args_map[spec_cls], custom_param=100)

        assert KVCacheSpecRegistry.get_manager_class(spec) is manager
        assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is uniform_base_spec

    def test_custom_spec_register_requires_manager(self):
        """Invalid register decorator arguments fail early."""

        with pytest.raises(AssertionError, match="manager_class is required"):

            @register_kv_cache_spec(
                uniform_type_base_spec=FullAttentionSpec,
            )
            @dataclass(frozen=True, kw_only=True)
            class _CustomFullSpecWithoutManager(FullAttentionSpec):
                custom_param: int = 16

    def test_unregistered_spec_no_registered_parent_raises(self):
        """
        A spec whose entire MRO contains no registered class resolves to None.
        Runtime callers should use check_kv_cache_spec_registry to fail early.
        Subclasses of registered specs intentionally do not fail — they inherit
        their parent's manager via MRO walking.
        """
        spec = _TrulyUnregisteredSpec(block_size=16)

        assert KVCacheSpecRegistry.get_manager_class(spec) is None
        assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is None

        with pytest.raises(
            ValueError, match="Unsupported KV cache spec type for layer layer_0"
        ):
            KVCacheSpecRegistry.check_kv_cache_spec_registry({"layer_0": spec})

        with pytest.raises(AssertionError, match="Unsupported KV cache spec type"):
            UniformTypeKVCacheSpecs.is_uniform_type({"layer_0": spec})

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

    @pytest.mark.parametrize("spec_cls", list(spec_manager_map))
    def test_builtin_specs_are_uniform_with_same_spec_type(self, spec_cls):
        spec = make_spec(spec_cls)
        assert are_uniform_specs(spec, replace(spec))

    def test_full_attention_family_specs_are_uniform(self):
        specs = [
            make_spec(FullAttentionSpec),
            make_spec(TQFullAttentionSpec),
            make_spec(MLAAttentionSpec),
            make_spec(HiddenStateCacheSpec),
            make_spec(SinkFullAttentionSpec),
        ]

        assert are_uniform_specs(*specs)

    @pytest.mark.parametrize(
        ("spec_cls", "field", "value"),
        [
            (SlidingWindowSpec, "sliding_window", 256),
            (SlidingWindowMLASpec, "sliding_window", 256),
            (ChunkedLocalAttentionSpec, "attention_chunk_size", 8),
            (MambaSpec, "num_speculative_blocks", 4),
        ],
    )
    def test_specs_with_type_specific_uniform_fields(self, spec_cls, field, value):
        spec = make_spec(spec_cls)
        changed_spec = replace(spec, **{field: value})

        assert not are_uniform_specs(spec, changed_spec)

    @pytest.mark.parametrize(
        ("left_cls", "right_cls"),
        [
            (FullAttentionSpec, CrossAttentionSpec),
            (FullAttentionSpec, SlidingWindowSpec),
            (FullAttentionSpec, ChunkedLocalAttentionSpec),
            (FullAttentionSpec, MambaSpec),
            (SlidingWindowMLASpec, SlidingWindowSpec),
            (ChunkedLocalAttentionSpec, SlidingWindowSpec),
            (MambaSpec, CrossAttentionSpec),
        ],
    )
    def test_different_uniform_groups_are_not_uniform(self, left_cls, right_cls):
        assert not are_uniform_specs(make_spec(left_cls), make_spec(right_cls))

    def test_different_block_sizes_are_not_uniform(self):
        spec = make_spec(FullAttentionSpec)

        assert not are_uniform_specs(spec, replace(spec, block_size=32))

    def test_registered_custom_spec_uses_base_uniform_rule(self):
        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            uniform_type_base_spec=FullAttentionSpec,
        )
        @dataclass(frozen=True, kw_only=True)
        class _CustomFullSpec(FullAttentionSpec):
            custom_param: int = 16

        custom_spec = _CustomFullSpec(
            block_size=64,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
        )

        assert are_uniform_specs(custom_spec, make_spec(FullAttentionSpec))
