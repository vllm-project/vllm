# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    DeviceConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.attention import AttentionConfig
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    MultipleOf,
    subclass_attention_backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import (
    create_composite_attention_backend,
    find_attention_impl_variant,
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
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, init_attn_backend
from vllm.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes

pytestmark = pytest.mark.skip_global_cleanup


def test_decode_backend_affects_config_hash():
    """A routed backend change must invalidate compiled configuration state."""
    default_hash = AttentionConfig(prefill_backend="FLASH_ATTN").compute_hash()
    routed_hash = AttentionConfig(
        prefill_backend="FLASH_ATTN", decode_backend="FLASHINFER"
    ).compute_hash()
    assert default_hash != routed_hash


def test_legacy_backend_sets_both_roles():
    config = AttentionConfig(backend="FLASH_ATTN")

    assert config.prefill_backend == AttentionBackendEnum.FLASH_ATTN
    assert config.decode_backend == AttentionBackendEnum.FLASH_ATTN

    config.backend = AttentionBackendEnum.TRITON_ATTN

    assert config.prefill_backend == AttentionBackendEnum.TRITON_ATTN
    assert config.decode_backend == AttentionBackendEnum.TRITON_ATTN


def test_decode_auto_selection_ignores_general_backend():
    """Decode auto-selection must not inherit a forced general backend."""
    config = VllmConfig(
        attention_config=AttentionConfig(
            prefill_backend="FLASH_ATTN", decode_backend="auto"
        ),
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


def test_draft_attention_backends_are_independent_from_target():
    """Explicit draft backends must override the target model's pair."""

    @dataclass
    class TargetConfig:
        attention_config: AttentionConfig

    target_config = TargetConfig(
        attention_config=AttentionConfig(
            prefill_backend=AttentionBackendEnum.TRITON_ATTN,
            decode_backend=AttentionBackendEnum.TRITON_ATTN,
        )
    )
    proposer = SimpleNamespace(
        vllm_config=target_config,
        speculative_config=SimpleNamespace(
            moe_backend=None,
            resolved_attention_backend=AttentionBackendEnum.FLASH_ATTN,
            resolved_attention_decode_backend=AttentionBackendEnum.FLASHINFER,
            kv_cache_dtype=None,
        ),
    )

    draft_config = SpecDecodeBaseProposer._create_draft_vllm_config(proposer)

    assert (
        draft_config.attention_config.prefill_backend == AttentionBackendEnum.FLASH_ATTN
    )
    assert (
        draft_config.attention_config.decode_backend == AttentionBackendEnum.FLASHINFER
    )


class _RoutingImpl(AttentionImpl):
    def __init__(self, num_heads, head_size, scale, *args, **kwargs):
        self.scale = scale
        self.kv_cache_dtype = "auto"

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _Backend(AttentionBackend):
    forward_includes_kv_cache_update = False
    kernel_block_sizes = [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "TEST"

    @staticmethod
    def get_impl_cls():
        return _RoutingImpl

    @staticmethod
    def get_builder_cls():
        return _Builder

    @classmethod
    def get_supported_kernel_block_sizes(cls):
        return cls.kernel_block_sizes

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (1, 0, 2, 3, 4) if include_num_layers_dimension else (0, 1, 2, 3)


class _DifferentCrossLayerPackingBackend(_Backend):
    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (1, 2, 0, 3, 4) if include_num_layers_dimension else (0, 1, 2, 3)


class _DifferentLayerLayoutBackend(_Backend):
    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (1, 0, 3, 2, 4) if include_num_layers_dimension else (0, 2, 1, 3)


class _DifferentShapeBackend(_Backend):
    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, num_kv_heads, block_size, head_size)


class _Fixed16Backend(_Backend):
    kernel_block_sizes = [16]


class _Fixed64Backend(_Backend):
    kernel_block_sizes = [64]


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


class _NHDDecodeBackend(_NHDBackend):
    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"


def _layouts_compatible(decode_backend, block_size=16):
    return kv_layouts_compatible(
        _Backend,
        decode_backend,
        num_kv_heads=8,
        head_size=128,
        block_size=block_size,
        kv_cache_dtype=None,
    )


def test_layout_compatibility_ignores_cross_layer_packing():
    """Cross-layer packing does not affect either backend's per-layer view."""
    assert _layouts_compatible(_DifferentCrossLayerPackingBackend)


def test_flexible_general_backend_adopts_decode_required_layout():
    """A flexible general backend must adopt a compatible decode requirement."""
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

        backend = layer.get_attn_backend()
        assert backend.general_backend_cls is _FlexibleGeneralBackend
        assert backend.decode_backend_cls is _HNDDecodeBackend
        assert get_kv_cache_layout() == "HND"
    finally:
        set_kv_cache_layout(None)


def test_builder_wrapping_backends_are_not_routed():
    """Wrappers like chunked local attention and static sink inject their
    semantics into the builder. A separately selected decode backend uses its
    own builder, so routing such a layer would silently drop local attention
    (or the sink blocks) on pure-decode batches."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    wrapped = subclass_attention_backend(
        name_prefix="ChunkedLocalAttention_",
        attention_backend_cls=_FlexibleGeneralBackend,
        builder_cls=_GeneralBuilder,
    )
    assert not wrapped.supports_decode_backend_routing()

    with (
        set_current_vllm_config(config),
        patch(
            "vllm.model_executor.layers.attention.attention.get_attn_backend",
            return_value=_HNDDecodeBackend,
        ),
    ):
        layer = Attention(num_heads=8, head_size=128, scale=0.1, attn_backend=wrapped)

    assert layer.get_attn_backend() is wrapped


def test_layers_cannot_require_different_kv_layouts():
    """One model cannot route layers through conflicting physical layouts."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))

    set_kv_cache_layout("NHD")
    try:
        with (
            set_current_vllm_config(config),
            patch(
                "vllm.model_executor.layers.attention.attention.get_attn_backend",
                side_effect=[_HNDDecodeBackend, _NHDDecodeBackend],
            ),
        ):
            Attention(
                num_heads=8,
                head_size=128,
                scale=0.1,
                prefix="layer.0",
                attn_backend=_FlexibleGeneralBackend,
            )
            with pytest.raises(ValueError, match="across layers"):
                Attention(
                    num_heads=8,
                    head_size=128,
                    scale=0.1,
                    prefix="layer.1",
                    attn_backend=_FlexibleGeneralBackend,
                )
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
    """Reject each independently unsafe way two backends can share KV cache."""
    general_backend = _NHDBackend if decode_backend is _HNDBackend else _Backend
    assert not kv_layouts_compatible(
        general_backend,
        decode_backend,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        kv_cache_dtype=None,
    )


def test_identical_backend_is_layout_compatible():
    """An identical backend does not need cross-backend safety checks."""
    assert kv_layouts_compatible(
        _WritesInForwardBackend,
        _WritesInForwardBackend,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        kv_cache_dtype=None,
    )


def test_disjoint_kernel_block_sizes_are_incompatible():
    """Backends must have at least one common kernel block size."""
    assert not kv_layouts_compatible(
        _Fixed16Backend,
        _Fixed64Backend,
        num_kv_heads=8,
        head_size=128,
        block_size=None,
        kv_cache_dtype=None,
    )
    with pytest.raises(ValueError, match="no common kernel block size"):
        create_composite_attention_backend(_Fixed16Backend, _Fixed64Backend)


class _Builder:
    supports_update_block_table = False
    requires_block_table_width = False

    def __init__(self, *args):
        self.reorder_batch_threshold = 1
        self.workspace_buffer = None

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return AttentionCGSupport.ALWAYS

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        return SimpleNamespace(builder=type(self).__name__)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return SimpleNamespace(builder=type(self).__name__)

    def set_workspace_buffer(self, workspace_buffer):
        self.workspace_buffer = workspace_buffer


class _GeneralBuilder(_Builder):
    def __init__(self, *args):
        super().__init__(*args)
        self.provided_workspace = object()

    def _get_workspace_buffer(self):
        return self.provided_workspace


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
    kernel_block_sizes = [32]

    @staticmethod
    def get_builder_cls():
        return _DecodeBuilder


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


def _make_attention_group(decode_backend=_DecodeBackend):
    return AttentionGroup(
        backend=create_composite_attention_backend(_GeneralBackend, decode_backend),
        layer_names=["layer"],
        kv_cache_spec=_attention_spec(),
        kv_cache_group_id=0,
    )


def _metadata_build_args(group):
    spec = group.kv_cache_spec
    return dict(
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
        kv_cache_config=KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
        ),
    )


def test_attention_group_accepts_same_backend_for_both_roles():
    """An identical pair must not allocate a redundant composite or builder."""
    group = _make_attention_group(_GeneralBackend)
    group.create_metadata_builders(None, torch.device("cpu"))

    assert group.backend is _GeneralBackend
    assert isinstance(group.get_metadata_builder(), _GeneralBuilder)


def test_kernel_block_size_is_supported_by_both_routed_backends():
    """The cache block size must satisfy both implementations' constraints."""
    group = _make_attention_group()
    spec = group.kv_cache_spec
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )

    assert prepare_kernel_block_sizes(config, [[group]]) == [32]


def test_mrv2_builds_metadata_with_the_routed_backend():
    """Batch phase must select and tag the matching metadata variant."""
    group = _make_attention_group()
    group.create_metadata_builders(None, torch.device("cpu"))
    common_args = _metadata_build_args(group)

    general_metadata = build_attn_metadata(
        **common_args, is_prefilling=torch.tensor([True])
    )
    decode_metadata = build_attn_metadata(
        **common_args, is_prefilling=torch.tensor([False])
    )

    assert general_metadata["layer"].builder == "_GeneralBuilder"
    assert general_metadata["layer"]._attention_backend_variant == 0
    assert decode_metadata["layer"].builder == "_DecodeBuilder"
    assert decode_metadata["layer"]._attention_backend_variant == 1


def test_cascade_attention_requires_both_backends():
    """The selected backend must support any cascade decision."""
    group = _make_attention_group()
    group.create_metadata_builders(None, torch.device("cpu"))
    builder = group.get_metadata_builder()
    builder.general_builder.use_cascade_attention = lambda *args, **kwargs: True
    builder.decode_builder.use_cascade_attention = lambda *args, **kwargs: False

    assert not builder.use_cascade_attention()


def test_composite_backend_requires_is_prefilling():
    """Missing batch phase metadata must fail instead of guessing from shape."""
    group = _make_attention_group()
    group.create_metadata_builders(None, torch.device("cpu"))

    with pytest.raises(AssertionError, match="require is_prefilling"):
        build_attn_metadata(**_metadata_build_args(group))


class _UniformDecodeLayer(AttentionLayerBase):
    def get_attn_backend(self):
        return create_composite_attention_backend(
            _GeneralBackend, _UniformDecodeBackend
        )

    def get_kv_cache_spec(self, vllm_config):
        return _attention_spec()


class _AlwaysDecodeLayer(AttentionLayerBase):
    def get_attn_backend(self):
        return create_composite_attention_backend(_GeneralBackend, _DecodeBackend)

    def get_kv_cache_spec(self, vllm_config):
        return _attention_spec()


def _init_routed_backend(layer):
    spec = _attention_spec()
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.compilation_config.static_forward_context["layer"] = layer
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    return init_attn_backend(kv_cache_config, config, torch.device("cpu"))


def _init_uniform_decode_backend():
    return _init_routed_backend(_UniformDecodeLayer())


def test_decode_backend_limits_cudagraph_support():
    """The weaker child must bound graph support and identify the culprit."""
    _, cg_support, _ = _init_uniform_decode_backend()

    assert cg_support.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert cg_support.min_cg_attn_backend == "_UniformDecodeBackend"


def test_routing_limits_cudagraph_support_to_uniform_batches():
    """Mixed FULL graphs cannot switch between otherwise capable backends."""
    _, cg_support, _ = _init_routed_backend(_AlwaysDecodeLayer())

    assert cg_support.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert cg_support.min_cg_attn_backend.startswith("Composite[")


def _make_cudagraph_manager(monkeypatch, max_num_seqs=8):
    monkeypatch.setattr(
        cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        cudagraph_utils.current_platform, "get_global_graph_pool", lambda: None
    )
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_AND_PIECEWISE",
        cudagraph_capture_sizes=list(range(1, max_num_seqs + 1)),
    )
    compilation_config.max_cudagraph_capture_size = max_num_seqs
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs
    )
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.speculative_config = None

    manager = cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
    )
    # dispatch() only consults the candidate table once graphs are captured.
    manager._graphs_captured = True
    return manager


def test_prefill_batch_never_dispatches_a_uniform_decode_graph(monkeypatch):
    """A token-uniform batch can still contain prefills (e.g. a one-token final
    prefill chunk). Those must not replay a uniform-decode graph, which was
    captured with the decode-only backend."""
    manager = _make_cudagraph_manager(monkeypatch)
    dispatch_args = dict(num_reqs=4, num_tokens=4, uniform_token_count=1)

    decode_desc = manager.dispatch(**dispatch_args, num_active_loras=0)
    prefill_desc = manager.dispatch(
        **dispatch_args, num_active_loras=0, has_prefill=True
    )

    assert decode_desc.cg_mode == CUDAGraphMode.FULL
    assert decode_desc.decode_only
    assert prefill_desc.cg_mode == CUDAGraphMode.PIECEWISE
    assert not prefill_desc.decode_only


class _SingleTokenDecodeBuilder(_DecodeBuilder):
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


class _SingleTokenDecodeBackend(_DecodeBackend):
    @staticmethod
    def get_builder_cls():
        return _SingleTokenDecodeBuilder


@pytest.mark.parametrize(
    ("decode_backend", "expected_support"),
    [
        (_DecodeBackend, AttentionCGSupport.UNIFORM_BATCH),
        (_SingleTokenDecodeBackend, AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE),
    ],
)
def test_composite_builder_caps_cudagraph_support(decode_backend, expected_support):
    """One FULL graph cannot serve both variants of a mixed batch, so the
    composite must advertise at most UNIFORM_BATCH (or the weaker child)."""
    backend = create_composite_attention_backend(_GeneralBackend, decode_backend)
    support = backend.get_builder_cls().get_cudagraph_support(None, _attention_spec())

    assert support == expected_support


def test_routed_variants_share_one_workspace_buffer():
    """Composition must not hide the workspace protocol: the runner shares one
    buffer across attention groups, and a variant it cannot see would allocate
    a second (hundreds of MiB for FlashInfer)."""
    attn_groups, _, _ = _init_uniform_decode_backend()
    builder = attn_groups[0][0].get_metadata_builder()

    assert (
        builder.decode_builder.workspace_buffer
        is builder.general_builder.provided_workspace
    )


@pytest.mark.parametrize("backend_variant", [0, 1])
def test_composite_impl_selects_from_metadata(backend_variant):
    """The internal metadata tag must select the corresponding child."""
    backend = create_composite_attention_backend(_GeneralBackend, _DecodeBackend)
    impl = backend.get_impl_cls()(8, 128, 0.1)
    metadata = SimpleNamespace(_attention_backend_variant=backend_variant)

    selected = impl.get_impl_for_metadata(metadata)

    assert selected is (impl.decode_impl if backend_variant else impl.general_impl)


def test_composite_impl_defaults_untagged_metadata_to_general():
    """Metadata built outside the composite must safely use the general impl."""
    backend = create_composite_attention_backend(_GeneralBackend, _DecodeBackend)
    impl = backend.get_impl_cls()(8, 128, 0.1)

    assert impl.get_impl_for_metadata(SimpleNamespace()) is impl.general_impl


def test_composite_cache_updates_use_general_impl():
    """Cache updates must be batch-agnostic so one piecewise graph is reusable."""
    calls = []

    class GeneralImpl(_RoutingImpl):
        def fused_rope_kvcache_supported(self):
            return True

        def fused_qk_norm_rope_kvcache_supported(self):
            return True

        def do_kv_cache_update(self, value):
            calls.append(("kv", value))

        def do_rope_and_kv_cache_update(self, value):
            calls.append(("rope", value))

        def do_qk_norm_rope_kvcache_update(self, value):
            calls.append(("qk_rope", value))

    class GeneralBackend(_GeneralBackend):
        @staticmethod
        def get_impl_cls():
            return GeneralImpl

    backend = create_composite_attention_backend(GeneralBackend, _DecodeBackend)
    impl = backend.get_impl_cls()(8, 128, 0.1)

    assert impl.fused_rope_kvcache_supported()
    assert impl.fused_qk_norm_rope_kvcache_supported()
    impl.do_kv_cache_update(1)
    impl.do_rope_and_kv_cache_update(2)
    impl.do_qk_norm_rope_kvcache_update(3)

    assert calls == [("kv", 1), ("rope", 2), ("qk_rope", 3)]


def test_composite_impl_reports_decode_argument_mismatch():
    """A decode impl rejecting general-only impl args must fail clearly."""

    class StrictDecodeImpl(_RoutingImpl):
        def __init__(self, num_heads, head_size, scale):
            super().__init__(num_heads, head_size, scale)

    class StrictDecodeBackend(_DecodeBackend):
        @staticmethod
        def get_impl_cls():
            return StrictDecodeImpl

    backend = create_composite_attention_backend(_GeneralBackend, StrictDecodeBackend)
    with pytest.raises(ValueError, match="does not accept"):
        backend.get_impl_cls()(8, 128, 0.1, chunk_lookback=3)


def test_composite_impl_exposes_backend_specific_variants():
    """Backend-specific setup must see implementations hidden by composition."""

    class GeneralImpl(_RoutingImpl):
        pass

    class DecodeImpl(_RoutingImpl):
        pass

    class GeneralBackend(_GeneralBackend):
        @staticmethod
        def get_impl_cls():
            return GeneralImpl

    class DecodeBackend(_DecodeBackend):
        @staticmethod
        def get_impl_cls():
            return DecodeImpl

    backend = create_composite_attention_backend(GeneralBackend, DecodeBackend)
    impl = backend.get_impl_cls()(8, 128, 0.1)

    assert find_attention_impl_variant(impl, GeneralImpl) is impl.general_impl
    assert find_attention_impl_variant(impl, DecodeImpl) is impl.decode_impl
