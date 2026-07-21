# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import (
    CUDAGraphMode,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.attention import AttentionConfig
from vllm.model_executor.layers.attention.attention import (
    Attention,
    _select_attention_impl,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    kv_layouts_compatible,
    set_kv_cache_layout,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, init_attn_backend
from vllm.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("value", "expected"),
    [("FLASHINFER", "FLASHINFER"), ("auto", None), (None, None)],
)
def test_decode_backend_parsing(value, expected):
    backend = AttentionConfig(decode_backend=value).decode_backend
    assert (backend.name if isinstance(backend, AttentionBackendEnum) else backend) == (
        expected
    )


def test_decode_backend_defaults_to_auto():
    assert AttentionConfig().decode_backend is None


def test_decode_backend_affects_config_hash():
    default_hash = AttentionConfig(backend="FLASH_ATTN").compute_hash()
    routed_hash = AttentionConfig(
        backend="FLASH_ATTN", decode_backend="FLASHINFER"
    ).compute_hash()
    assert default_hash != routed_hash


def test_decode_auto_selection_ignores_general_backend():
    config = VllmConfig(
        attention_config=AttentionConfig(backend="FLASH_ATTN", decode_backend="auto"),
        device_config=DeviceConfig(device="cpu"),
    )
    selected_backend = object()
    with (
        set_current_vllm_config(config),
        patch(
            "vllm.v1.attention.selector._cached_get_attn_backend",
            return_value=selected_backend,
        ) as selector,
    ):
        result = get_attn_backend(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype=None,
            use_global_backend=False,
        )

    assert result is selected_backend
    assert selector.call_args.kwargs["backend"] is None


class _RoutingImpl(AttentionImpl):
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads=None,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
    ):
        self.scale = scale
        self.kv_cache_dtype = kv_cache_dtype

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _Backend(AttentionBackend):
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        return "TEST"

    @staticmethod
    def get_impl_cls():
        return _RoutingImpl

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


class _FlexibleGeneralBackend(_Backend):
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"


class _HNDDecodeBackend(_HNDBackend):
    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"


def _layouts_compatible(decode_backend, block_size=16):
    return kv_layouts_compatible(
        _Backend,
        decode_backend,
        head_size=128,
        block_size=block_size,
        kv_cache_dtype=None,
    )


def test_layout_compatibility_ignores_cross_layer_packing():
    assert _layouts_compatible(_DifferentCrossLayerPackingBackend)


def test_flexible_general_backend_adopts_decode_required_layout():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))

    set_kv_cache_layout("NHD")
    try:
        with (
            set_current_vllm_config(config),
            patch(
                "vllm.model_executor.layers.attention.attention.get_attn_backend",
                return_value=_HNDDecodeBackend,
            ),
        ):
            layer = Attention(
                num_heads=8,
                head_size=128,
                scale=0.1,
                attn_backend=_FlexibleGeneralBackend,
            )

        assert layer.get_decode_attn_backend() is _HNDDecodeBackend
        assert get_kv_cache_layout() == "HND"
    finally:
        set_kv_cache_layout(None)


@pytest.mark.parametrize(
    "decode_backend",
    [
        _DifferentLayerLayoutBackend,
        _DifferentShapeBackend,
        _WritesInForwardBackend,
        _HNDBackend,
    ],
)
def test_layout_compatibility_rejects_unsafe_pairings(decode_backend):
    general_backend = _NHDBackend if decode_backend is _HNDBackend else _Backend
    assert not kv_layouts_compatible(
        general_backend,
        decode_backend,
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


class _GeneralBuilder(_Builder):
    pass


class _DecodeBuilder(_Builder):
    pass


class _UniformDecodeBuilder(_DecodeBuilder):
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return AttentionCGSupport.UNIFORM_BATCH


class _GeneralBackend(_Backend):
    @staticmethod
    def get_builder_cls():
        return _GeneralBuilder


class _DecodeBackend(_Backend):
    @staticmethod
    def get_builder_cls():
        return _DecodeBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [32]


class _UniformDecodeBackend(_DecodeBackend):
    @staticmethod
    def get_builder_cls():
        return _UniformDecodeBuilder


def _attention_spec(block_size=64):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )


def test_attention_group_selects_builder_for_routed_backend():
    group = AttentionGroup(
        backend=_GeneralBackend,
        decode_backend=_DecodeBackend,
        layer_names=["layer"],
        kv_cache_spec=_attention_spec(),
        kv_cache_group_id=0,
    )
    group.create_metadata_builders(None, torch.device("cpu"))

    assert isinstance(group.get_metadata_builder(), _GeneralBuilder)
    assert isinstance(
        group.get_metadata_builder(use_decode_backend=True), _DecodeBuilder
    )


def test_attention_group_accepts_same_backend_for_both_roles():
    group = AttentionGroup(
        backend=_GeneralBackend,
        decode_backend=_GeneralBackend,
        layer_names=["layer"],
        kv_cache_spec=_attention_spec(),
        kv_cache_group_id=0,
    )
    group.create_metadata_builders(None, torch.device("cpu"))

    assert group.decode_backend is _GeneralBackend
    assert isinstance(
        group.get_metadata_builder(use_decode_backend=True), _GeneralBuilder
    )


def test_kernel_block_size_is_supported_by_both_routed_backends():
    spec = _attention_spec()
    group = AttentionGroup(
        backend=_GeneralBackend,
        decode_backend=_DecodeBackend,
        layer_names=["layer"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
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
        backend=_GeneralBackend,
        decode_backend=_DecodeBackend,
        layer_names=["layer"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
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

    general_metadata = build_attn_metadata(**common_args)
    decode_metadata = build_attn_metadata(**common_args, use_decode_backend=True)

    assert general_metadata["layer"] == "_GeneralBuilder"
    assert decode_metadata["layer"] == "_DecodeBuilder"


class _Layer(AttentionLayerBase):
    def get_attn_backend(self):
        return _GeneralBackend

    def get_decode_attn_backend(self):
        return _DecodeBackend

    def get_kv_cache_spec(self, vllm_config):
        return _attention_spec()


class _UniformDecodeLayer(_Layer):
    def get_decode_attn_backend(self):
        return _UniformDecodeBackend


def test_mrv2_groups_general_and_decode_backends_together():
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

    assert groups[0][0].backend is _GeneralBackend
    assert groups[0][0].decode_backend is _DecodeBackend
    assert kernel_block_sizes == [32]


def test_decode_backend_limits_cudagraph_support():
    spec = _attention_spec()
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.compilation_config.static_forward_context["layer"] = _UniformDecodeLayer()
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )

    _, cg_support, _ = init_attn_backend(kv_cache_config, config, torch.device("cpu"))

    assert cg_support.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert cg_support.min_cg_attn_backend == "_UniformDecodeBackend"


@pytest.mark.parametrize(
    ("use_decode_backend", "expected"),
    [(False, "general"), (True, "decode")],
)
def test_select_attention_impl(use_decode_backend, expected):
    layer = SimpleNamespace(impl="general", decode_impl="decode")
    context = SimpleNamespace(
        use_decode_backend=use_decode_backend,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    with patch(
        "vllm.model_executor.layers.attention.attention.get_forward_context",
        return_value=context,
    ):
        assert _select_attention_impl(layer) == expected


def test_decode_backend_is_selected_in_full_cudagraph():
    layer = SimpleNamespace(impl="general", decode_impl="decode")
    context = SimpleNamespace(
        use_decode_backend=True,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
    )
    with patch(
        "vllm.model_executor.layers.attention.attention.get_forward_context",
        return_value=context,
    ):
        assert _select_attention_impl(layer) == "decode"


def test_dcp_checks_decode_implementation():
    general_impl = SimpleNamespace(need_to_return_lse_for_decode=False)
    decode_impl = SimpleNamespace(need_to_return_lse_for_decode=True)
    layer = SimpleNamespace(impl=general_impl, decode_impl=decode_impl)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=None,
    )

    with patch(
        "vllm.v1.worker.cp_utils.get_layers_from_vllm_config",
        return_value={"layer": layer},
    ):
        check_attention_cp_compatibility(config)


def test_model_runner_v2_allows_backend_routing():
    config = VllmConfig(
        attention_config=AttentionConfig(decode_backend="FLASHINFER"),
        device_config=DeviceConfig(device="cpu"),
    )
    assert (
        "prefill/decode attention backend routing"
        not in config._get_v2_model_runner_unsupported_features()
    )
