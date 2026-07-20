# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import CUDAGraphMode, DeviceConfig, VllmConfig
from vllm.config.attention import AttentionConfig
from vllm.model_executor.layers.attention.attention import _select_attention_impl
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import kv_layouts_compatible
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, init_attn_backend
from vllm.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("value", "expected"),
    [("FLASHINFER", "FLASHINFER"), ("auto", None), (None, None)],
)
def test_prefill_backend_parsing(value, expected):
    backend = AttentionConfig(prefill_backend=value).prefill_backend
    assert (backend.name if backend is not None else None) == expected


def test_prefill_backend_affects_config_hash():
    default_hash = AttentionConfig(backend="FLASH_ATTN").compute_hash()
    routed_hash = AttentionConfig(
        backend="FLASH_ATTN", prefill_backend="FLASHINFER"
    ).compute_hash()
    assert default_hash != routed_hash


class _Backend(AttentionBackend):
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        return "TEST"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4)
        return (0, 1, 2, 3)


class _DifferentCrossLayerPackingBackend(_Backend):
    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            return (1, 2, 0, 3, 4)
        return (0, 1, 2, 3)


class _DifferentLayerLayoutBackend(_Backend):
    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            return (1, 0, 3, 2, 4)
        return (0, 2, 1, 3)


class _DifferentShapeBackend(_Backend):
    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, num_kv_heads, block_size, head_size)


class _WritesInForwardBackend(_Backend):
    forward_includes_kv_cache_update = True


class _NHDBackend(_Backend):
    @classmethod
    def get_required_kv_cache_layout(cls):
        return "NHD"


class _HNDBackend(_Backend):
    @classmethod
    def get_required_kv_cache_layout(cls):
        return "HND"


def _layouts_compatible(prefill_backend, block_size=16):
    return kv_layouts_compatible(
        _Backend,
        prefill_backend,
        head_size=128,
        block_size=block_size,
        kv_cache_dtype=None,
    )


def test_layout_compatibility_ignores_cross_layer_packing():
    assert _layouts_compatible(_DifferentCrossLayerPackingBackend)


@pytest.mark.parametrize(
    "prefill_backend",
    [
        _DifferentLayerLayoutBackend,
        _DifferentShapeBackend,
        _WritesInForwardBackend,
        _HNDBackend,
    ],
)
def test_layout_compatibility_rejects_unsafe_pairings(prefill_backend):
    decode_backend = _NHDBackend if prefill_backend is _HNDBackend else _Backend
    assert not kv_layouts_compatible(
        decode_backend,
        prefill_backend,
        head_size=128,
        block_size=16,
        kv_cache_dtype=None,
    )


class _Builder:
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.reorder_batch_threshold = 1

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return AttentionCGSupport.ALWAYS

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        return type(self).__name__

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return type(self).__name__


class _DecodeBuilder(_Builder):
    pass


class _PrefillBuilder(_Builder):
    pass


class _DecodeBackend(_Backend):
    @staticmethod
    def get_builder_cls():
        return _DecodeBuilder


class _PrefillBackend(_Backend):
    @staticmethod
    def get_builder_cls():
        return _PrefillBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [32]


def _attention_spec(block_size=64):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )


def test_attention_group_selects_builder_for_routed_backend():
    group = AttentionGroup(
        backend=_DecodeBackend,
        layer_names=["layer"],
        kv_cache_spec=_attention_spec(),
        kv_cache_group_id=0,
        prefill_backend=_PrefillBackend,
    )
    group.create_metadata_builders(None, torch.device("cpu"))

    assert isinstance(group.get_metadata_builder(), _DecodeBuilder)
    assert isinstance(
        group.get_metadata_builder(use_prefill_backend=True), _PrefillBuilder
    )


def test_kernel_block_size_is_supported_by_both_routed_backends():
    spec = _attention_spec()
    group = AttentionGroup(
        backend=_DecodeBackend,
        layer_names=["layer"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        prefill_backend=_PrefillBackend,
    )
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )

    assert prepare_kernel_block_sizes(config, [[group]]) == [32]


def test_mrv2_builds_metadata_with_the_routed_backend():
    spec = _attention_spec()
    group = AttentionGroup(
        backend=_DecodeBackend,
        layer_names=["layer"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        prefill_backend=_PrefillBackend,
    )
    group.create_metadata_builders(None, torch.device("cpu"))
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    common_args = dict(
        attn_groups=[[group]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        block_tables=[torch.zeros(1, 1, dtype=torch.int32)],
        slot_mappings=torch.zeros(1, 1, dtype=torch.int64),
        kv_cache_config=config,
    )

    decode_metadata = build_attn_metadata(**common_args)
    prefill_metadata = build_attn_metadata(**common_args, use_prefill_backend=True)

    assert decode_metadata["layer"] == "_DecodeBuilder"
    assert prefill_metadata["layer"] == "_PrefillBuilder"


class _Layer(AttentionLayerBase):
    def get_attn_backend(self):
        return _DecodeBackend

    def get_prefill_attn_backend(self):
        return _PrefillBackend

    def get_kv_cache_spec(self, vllm_config):
        return _attention_spec()


def test_mrv2_groups_decode_and_prefill_backends_together():
    spec = _attention_spec()
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.compilation_config.static_forward_context["layer"] = _Layer()
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )

    groups, _, kernel_block_sizes = init_attn_backend(
        kv_cache_config, config, torch.device("cpu")
    )

    assert groups[0][0].backend is _DecodeBackend
    assert groups[0][0].prefill_backend is _PrefillBackend
    assert kernel_block_sizes == [32]


@pytest.mark.parametrize(
    ("use_prefill_backend", "expected"),
    [(False, "decode"), (True, "prefill")],
)
def test_select_attention_impl(use_prefill_backend, expected):
    layer = SimpleNamespace(impl="decode", prefill_impl="prefill")
    context = SimpleNamespace(
        use_prefill_backend=use_prefill_backend,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    with patch(
        "vllm.model_executor.layers.attention.attention.get_forward_context",
        return_value=context,
    ):
        assert _select_attention_impl(layer) == expected


def test_prefill_backend_is_not_selected_in_full_cudagraph():
    layer = SimpleNamespace(impl="decode", prefill_impl="prefill")
    context = SimpleNamespace(
        use_prefill_backend=True,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
    )
    with (
        patch(
            "vllm.model_executor.layers.attention.attention.get_forward_context",
            return_value=context,
        ),
        pytest.raises(AssertionError, match="full CUDA graph"),
    ):
        _select_attention_impl(layer)


def test_model_runner_v2_allows_backend_routing():
    config = VllmConfig(
        attention_config=AttentionConfig(prefill_backend="FLASHINFER"),
        device_config=DeviceConfig(device="cpu"),
    )
    assert (
        "prefill/decode attention backend routing"
        not in config._get_v2_model_runner_unsupported_features()
    )
