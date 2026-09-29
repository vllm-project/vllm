# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import importlib.machinery
import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import WeakKeyDictionary, ref

import pytest
import torch
from torch.nn.parameter import UninitializedParameter

import vllm.model_executor.model_loader.reload.layerwise as reload_layerwise
import vllm.model_executor.model_loader.reload.meta as reload_meta
from vllm.config import ModelConfig
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    initialize_online_processing,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.meta import (
    capture_layer_to_meta,
    get_numel_loaded,
    materialize_layer,
    materialize_meta_tensor,
    restore_layer_on_meta,
    to_meta_tensor,
)
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader,
    default_weight_loader,
)
from vllm.platforms import current_platform


def _fp8_reload_unsupported() -> bool:
    """Whether the FP8 reload/online-quantize tests should be skipped.

    ``supports_fp8()`` returns True on MI250 (gfx90a) because the general
    quantization paths upcast FP8 weights, but gfx90a has no native FP8 and
    cannot run these reload models, so treat it as unsupported here.
    """
    if not current_platform.supports_fp8():
        return True
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx90a

        return on_gfx90a()
    return False


class _AliasedBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.weight = torch.nn.Parameter(weight)
        self.register_buffer(
            "weight_view", self.weight.detach().view(-1), persistent=False
        )


class _ParentAliasedChildBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.conv1d = torch.nn.Linear(3, 2, bias=False)
        self.conv1d.weight.data.copy_(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        self.register_buffer(
            "conv_weights", self.conv1d.weight.detach().view(-1), persistent=False
        )


class _ChildAliasOnlyBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Linear(3, 2, bias=False)
        self.conv1d.weight.data.copy_(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        self.register_buffer(
            "conv_weights", self.conv1d.weight.detach().view(-1), persistent=False
        )


class _AliasedBufferWithUninitializedChildLayer(_AliasedBufferLayer):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Module()
        self.child.register_parameter(
            "lazy_weight", UninitializedParameter(requires_grad=False)
        )


class _NonPersistentBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.register_buffer("scale", torch.tensor(0.25), persistent=False)


class _ReloadableMMEncoderAttention(MMEncoderAttention):
    """Minimal stand-in to test reload lifecycle without encoder initialization."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.weight.weight_loader = default_weight_loader
        self.post_load_called = False

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self.post_load_called = True


class _ReloadableAttentionLayer(
    torch.nn.Module,
    AttentionLayerBase,
):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.weight.weight_loader = default_weight_loader
        self.post_load_called = False

    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config):
        return None

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self.post_load_called = True


def test_move_metatensors():
    tensor = torch.empty((1, 2, 3))
    meta_tensor = to_meta_tensor(tensor)
    materialized_tensor = materialize_meta_tensor(meta_tensor)

    assert meta_tensor.device.type == "meta"
    assert tensor.device == materialized_tensor.device

    assert tensor.dtype == meta_tensor.dtype == materialized_tensor.dtype
    assert tensor.shape == meta_tensor.shape == materialized_tensor.shape
    assert tensor.__class__ == meta_tensor.__class__ == materialized_tensor.__class__
    assert tensor.__dict__ == meta_tensor.__dict__ == materialized_tensor.__dict__


@pytest.mark.parametrize(
    "layer_cls",
    [_ReloadableMMEncoderAttention, _ReloadableAttentionLayer],
)
def test_attention_reload_defers_post_load(default_vllm_config, layer_cls):
    default_vllm_config.model_config = ModelConfig()
    layer = layer_cls()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)

    assert not layer.post_load_called

    finalize_layerwise_reload(model, default_vllm_config.model_config)

    assert layer.post_load_called
    assert torch.equal(layer.weight, loaded_weight)


@pytest.mark.parametrize(
    "layer_cls",
    [_ReloadableMMEncoderAttention, _ReloadableAttentionLayer],
)
def test_attention_first_load_processes_weights(default_vllm_config, layer_cls):
    default_vllm_config.model_config = ModelConfig()
    layer = layer_cls()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)

    initialize_online_processing(layer)
    layer.weight.weight_loader(layer.weight, loaded_weight)

    finalize_layerwise_reload(model, default_vllm_config.model_config)

    assert layer.post_load_called
    assert torch.equal(layer.weight, loaded_weight)


def test_reload_lifecycle():
    layer = torch.nn.Linear(2, 3)
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cpu"),
    )

    restore_layer_on_meta(layer, info)
    for name, tensor in get_layer_tensors(layer).items():
        meta_tensor = getattr(layer, name)
        assert tensor.dtype == meta_tensor.dtype
        assert tensor.shape == meta_tensor.shape
        assert tensor.__class__ == meta_tensor.__class__
        assert tensor.__dict__ == meta_tensor.__dict__

    materialize_layer(layer, info)
    for name, tensor in get_layer_tensors(layer).items():
        materialized_tensor = getattr(layer, name)
        assert tensor.dtype == materialized_tensor.dtype
        assert tensor.shape == materialized_tensor.shape
        assert tensor.__class__ == materialized_tensor.__class__
        assert tensor.__dict__ == materialized_tensor.__dict__


def test_restore_layer_replaces_postprocessed_tensor_attribute():
    layer = torch.nn.Linear(2, 3, bias=False)
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cpu"),
    )
    del layer.weight
    layer.weight = torch.empty(3, 2)

    restore_layer_on_meta(layer, info)

    assert isinstance(layer.weight, torch.nn.Parameter)
    assert layer.weight.is_meta


def test_materialize_layer_preserves_non_meta_tensors():
    """Ensure that materialize_layer does not overwrite non meta tensors."""
    layer = torch.nn.Linear(2, 3, bias=True)

    # Create a non meta bias tensor and meta weight, which can happen with FP8
    bias_values = torch.ones(3)
    layer.bias.data.copy_(bias_values)
    layer.weight = torch.nn.Parameter(layer.weight.data.to("meta"))

    assert layer.weight.is_meta
    assert not layer.bias.is_meta

    # materialize the layer weights after the bias is initialized
    info = LayerReloadingInfo(
        restore_metadata=({}, {}),
        restore_device=torch.device("cpu"),
    )
    materialize_layer(layer, info)

    # Ensure the weight materialized off meta
    assert not layer.weight.is_meta
    assert layer.weight.device.type == "cpu"

    # Ensure that the bias is (still) not meta and values are unchanged
    assert not layer.bias.is_meta
    assert torch.equal(layer.bias.data, bias_values)


_MARLIN_SIZE_K, _MARLIN_SIZE_N, _MARLIN_GROUP_SIZE = 128, 64, 64


def _stub_marlin_ops(monkeypatch):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(
        ops,
        "gptq_marlin_repack",
        lambda w, size_k, size_n, num_bits, is_a_8bit=False: torch.zeros(
            size_k // 16, size_n * 2, dtype=torch.int32
        ),
    )


def _make_marlin_kernel():
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
        MarlinLinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    kernel = object.__new__(MarlinLinearKernel)
    kernel.config = MPLinearLayerConfig(
        full_weight_shape=(_MARLIN_SIZE_K, _MARLIN_SIZE_N),
        partition_weight_shape=(_MARLIN_SIZE_K, _MARLIN_SIZE_N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=_MARLIN_GROUP_SIZE,
        zero_points=False,
    )
    kernel.w_q_name = "qweight"
    kernel.w_s_name = "scales"
    kernel.w_zp_name = None
    return kernel


def _load_marlin_checkpoint_format_weights(layer):
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    layer.qweight = PackedvLLMParameter(
        data=torch.zeros(_MARLIN_SIZE_K // 8, _MARLIN_SIZE_N, dtype=torch.int32),
        input_dim=0,
        output_dim=1,
        packed_dim=0,
        packed_factor=8,
        weight_loader=default_weight_loader,
    )
    layer.scales = GroupQuantScaleParameter(
        data=torch.ones(
            _MARLIN_SIZE_K // _MARLIN_GROUP_SIZE, _MARLIN_SIZE_N, dtype=torch.float16
        ),
        input_dim=0,
        output_dim=1,
        weight_loader=default_weight_loader,
    )


def test_marlin_post_load_does_not_own_workspace(monkeypatch, dist_init):
    """Weight reload must not create layer-owned Marlin lock storage."""
    _stub_marlin_ops(monkeypatch)
    kernel = _make_marlin_kernel()

    layer = torch.nn.Module()
    _load_marlin_checkpoint_format_weights(layer)
    kernel.process_weights_after_loading(layer)

    assert not hasattr(kernel, "workspace")

    _load_marlin_checkpoint_format_weights(layer)
    kernel.process_weights_after_loading(layer)

    assert not hasattr(kernel, "workspace")


@pytest.mark.parametrize("variant", ["fp8", "mxfp8", "nvfp4"])
def test_marlin_prepare_layer_does_not_own_workspace(monkeypatch, variant):
    """Weight preparation must not attach runtime workspace to model layers."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import (
        marlin_utils,
        marlin_utils_fp4,
        marlin_utils_fp8,
    )

    size_k, size_n = 128, 64

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(
        ops,
        "gptq_marlin_repack",
        lambda b_q_weight, size_k, size_n, num_bits, is_a_8bit=False: torch.zeros(
            size_k // 16, size_n * 2, dtype=torch.int32
        ),
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.orig_dtype = torch.float16
    layer.params_dtype = torch.float16

    if variant == "fp8":
        prepare = marlin_utils_fp8.prepare_fp8_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_k, size_n, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )
    elif variant == "mxfp8":
        prepare = marlin_utils_fp8.prepare_mxfp8_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_n, size_k, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.full((size_n, size_k // 32), 127, dtype=torch.uint8),
                requires_grad=False,
            )
    else:
        prepare = marlin_utils_fp4.prepare_fp4_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_n, size_k // 2, dtype=torch.uint8),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.ones(size_n, size_k // 16, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_global_scale = torch.nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )

    load_checkpoint_format_weights()
    prepare(layer)
    assert not hasattr(layer, "workspace")

    # Reload: fresh checkpoint-format tensors, prepare runs again
    load_checkpoint_format_weights()
    prepare(layer)

    assert not hasattr(layer, "workspace")


def test_marlin_workspace_uses_persistent_workspace_manager(monkeypatch):
    """Calls reuse initialized locks; independent streams get separate storage."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.utils import torch_utils
    from vllm.v1.worker import workspace as workspace_module

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    device = torch.device("cpu")
    manager = workspace_module.WorkspaceManager(device)
    monkeypatch.setattr(workspace_module, "_manager", manager)
    stream = "main"
    monkeypatch.setattr(torch_utils, "current_stream", lambda: stream)

    workspace = marlin_utils.get_marlin_workspace(device)
    assert workspace.shape == (4 * marlin_utils.MARLIN_MAX_BLOCKS_PER_SM,)
    assert workspace.dtype == torch.int32
    assert torch.count_nonzero(workspace) == 0

    workspace.fill_(1)
    assert marlin_utils.get_marlin_workspace(device) is workspace
    assert torch.all(workspace == 1)

    stream = "aux"
    aux_workspace = marlin_utils.get_marlin_workspace(device)
    assert aux_workspace.data_ptr() != workspace.data_ptr()
    assert torch.count_nonzero(aux_workspace) == 0

    manager.lock()
    assert marlin_utils.get_marlin_workspace(device) is aux_workspace


def test_marlin_workspace_without_manager(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.v1.worker import workspace as workspace_module

    monkeypatch.setattr(workspace_module, "_manager", None)
    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    first = marlin_utils.get_marlin_workspace(torch.device("cpu"))
    second = marlin_utils.get_marlin_workspace(torch.device("cpu"))
    assert first.data_ptr() != second.data_ptr()
    assert first.shape == (4 * marlin_utils.MARLIN_MAX_BLOCKS_PER_SM,)
    assert torch.count_nonzero(first) == 0


def test_model_cleanup(dist_init, default_vllm_config):
    layer = QKVParallelLinear(2, 3, 4)
    assert layer.weight.weight_loader.__self__ is layer
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cpu"),
    )

    mock_info_dict: WeakKeyDictionary[torch.nn.Module, LayerReloadingInfo] = (
        WeakKeyDictionary()
    )
    mock_info_dict[layer] = info
    layer_ref = ref(layer)

    del layer
    gc.collect()

    assert layer_ref() is None
    assert len(mock_info_dict) == 0


@pytest.mark.parametrize("is_gated", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("padded", [False, True])
def test_padded_moe_reload_releases_each_layer(
    monkeypatch, is_gated, has_bias, tp_rank, padded
):
    """Checkpoint-sized copies finish each layer without global finalization."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    hidden, intermediate, experts = 4, 3, 2
    stored_hidden, stored_intermediate = (8, 8) if padded else (hidden, intermediate)
    config = SimpleNamespace(
        hidden_dim_unpadded=hidden,
        intermediate_size_per_partition_unpadded=intermediate,
        is_act_and_mul=is_gated,
        has_bias=has_bias,
        tp_rank=tp_rank,
        tp_shard_with_padding=False,
        moe_parallel_config=SimpleNamespace(tp_size=2),
    )
    model = torch.nn.ModuleList()
    processed: list[torch.nn.Module] = []
    for _ in range(2):
        method = object.__new__(UnquantizedFusedMoEMethod)
        torch.nn.Module.__init__(method)
        method.moe = config
        # The regression concerns streaming reload, not kernel conversion.
        monkeypatch.setattr(method, "process_weights_after_loading", processed.append)
        layer = object.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.moe_config = config
        layer.quant_config = None
        layer.quant_method = method
        layer.expert_map_manager = SimpleNamespace(map_global_to_local=lambda i: i)
        layer._loaded_expert_biases = set()
        method.create_weights(
            layer,
            experts,
            stored_hidden,
            stored_intermediate,
            torch.float32,
            weight_loader=layer.weight_loader,
        )
        model.append(layer)

    record_metadata_for_reloading(model)
    original_params = [dict(layer.named_parameters()) for layer in model]
    shards = ["w1", "w3", "w2"] if is_gated else ["w1", "w2"]
    for cycle in range(2):
        initialize_layerwise_reload(model)
        for layer_index, layer in enumerate(model):
            info = reload_layerwise.get_layerwise_info(layer)
            inputs = []
            expected = {
                name: torch.full_like(p, float("nan"))
                for name, p in original_params[layer_index].items()
            }
            params = dict(layer.named_parameters())
            calls = [
                (e, s, b)
                for e in range(experts)
                for s in shards
                for b in ([False, True] if has_bias else [False])
            ]
            for call_index, (expert, shard, bias) in enumerate(calls):
                name = ("w2" if shard == "w2" else "w13") + (
                    "_bias" if bias else "_weight"
                )
                shape = (
                    ((hidden,) if bias else (hidden, 2 * intermediate))
                    if shard == "w2"
                    else ((2 * intermediate,) if bias else (2 * intermediate, hidden))
                )
                weight = torch.arange(
                    torch.Size(shape).numel(), dtype=torch.float32
                ).reshape(shape)
                weight = weight + 100 * (1 + call_index + cycle)
                inputs.append(ref(weight))
                # Direct checkpoint loading is the reference for deferred reload.
                layer.weight_loader(expected[name], weight, name, shard, expert)
                param = params[name]
                param.weight_loader(param, weight, name, shard, expert)
                del weight
                if call_index != len(calls) - 1:
                    assert info.can_load(), (
                        "Layer processed before its final checkpoint shard"
                    )

            assert not info.can_load(), (
                "Padding must not defer the layer until finalization"
            )
            assert not info.loaded_weights
            assert len(processed) == cycle * len(model) + layer_index + 1
            assert all(source() is None for source in inputs)
            for name, original in original_params[layer_index].items():
                assert getattr(layer, name) is original
                # Kernel-specific tests cover padding, which is not checkpoint data.
                mask = torch.isfinite(expected[name])
                assert torch.equal(original[mask], expected[name][mask])


def test_get_numel_loaded():
    param = torch.empty(10, device="meta")
    loaded_weight = torch.empty(10)

    def complex_weight_loader(param, loaded_weight):
        param[:3] = loaded_weight[:3]
        param[5:8] = loaded_weight[5:8]
        return "value"

    args = inspect.signature(complex_weight_loader).bind(param, loaded_weight)
    num_loaded, ret = get_numel_loaded(complex_weight_loader, args)
    assert num_loaded == 6
    assert ret == "value"


def test_get_numel_loaded_caps_at_param_size():
    # composed_weight_loader copies into the param twice (the load and the
    # in-place post-load transform), but only param.numel() distinct elements
    # are loaded. get_numel_loaded must not double-count, otherwise a layer's
    # loaded-element total can be reached early and trailing params get dropped.
    param = torch.empty(10)
    loaded_weight = torch.ones(10)
    loader = composed_weight_loader(default_weight_loader, lambda x: x + 1)

    args = inspect.signature(loader).bind(param, loaded_weight)
    num_loaded, _ = get_numel_loaded(loader, args)
    assert num_loaded == 10


def test_layerwise_loading_warning_only_checks_new_layers(monkeypatch):
    layers = [torch.nn.Linear(16, 1, bias=False) for _ in range(2)]

    def partial_weight_loader(param, loaded_weight):
        param.view(-1)[: loaded_weight.numel()].copy_(loaded_weight)

    for layer in layers:
        layer.weight.requires_grad_(False)
        layer.weight.weight_loader = partial_weight_loader
        reload_layerwise.initialize_online_processing(layer)

    monkeypatch.setattr(reload_layerwise, "has_device_tensors", lambda _: True)
    get_info_size = Mock(return_value=0)
    warning_once = Mock()
    monkeypatch.setattr(reload_layerwise, "get_info_size", get_info_size)
    monkeypatch.setattr(reload_layerwise.logger, "warning_once", warning_once)

    reload_layerwise.LOADING_LAYERS.clear()
    try:
        for layer in layers:
            for _ in range(3):
                layer.weight.weight_loader(layer.weight, torch.ones(1))
    finally:
        reload_layerwise.LOADING_LAYERS.clear()

    assert get_info_size.call_count == 2
    warning_once.assert_called_once()


class _ComposedLoaderLayer(torch.nn.Module):
    """Mimics a Mamba2 mixer's equal-numel direct params (A, D, dt_bias).

    ``A`` uses ``composed_weight_loader`` (an extra in-place transform copy),
    matching ``MambaMixer2`` where ``A`` is loaded as ``-exp(A_log)``.
    """

    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(4, dtype=torch.float32))
        self.D = torch.nn.Parameter(torch.ones(4))
        self.dt_bias = torch.nn.Parameter(torch.ones(4))
        self.A.weight_loader = composed_weight_loader(
            default_weight_loader, lambda x: -torch.exp(x.float())
        )
        self.D.weight_loader = default_weight_loader
        self.dt_bias.weight_loader = default_weight_loader


def test_layerwise_reload_composed_loader_does_not_drop_params(monkeypatch):
    # Regression test: a composed_weight_loader param (A) used to double-count
    # its elements, finalizing the layer before the trailing param (D) was
    # loaded and leaving it as uninitialized materialized memory.
    layer = _ComposedLoaderLayer()
    model = torch.nn.Sequential(layer)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(float("nan"))
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    loaded = {
        "A": torch.full((4,), 0.5),
        "dt_bias": torch.full((4,), 3.0),
        "D": torch.full((4,), 7.0),
    }

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    # Mimic real load_weights: resolve params once, then load in checkpoint
    # order with D last (the param that was dropped).
    params = dict(layer.named_parameters())
    for name in ("A", "dt_bias", "D"):
        param = params[name]
        param.weight_loader(param, loaded[name])
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.A, -torch.exp(loaded["A"]))
    assert torch.equal(layer.dt_bias, loaded["dt_bias"])
    assert torch.equal(layer.D, loaded["D"])


class _RecordingQuantMethod(QuantizeMethodBase):
    """Records the layer's bias at the moment processing runs."""

    uses_meta_device = True

    def __init__(self):
        self.bias_at_process = None

    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        pass

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.bias_at_process = layer.bias.detach().clone()


class _LateBiasLayer(torch.nn.Module):
    """Mimics an online-quantized linear: `weight` is created on meta by
    `create_weights()`, which wraps the loaders, and the linear base registers
    `bias` afterwards."""

    def __init__(self, quant_method):
        super().__init__()
        self.quant_method = quant_method
        weight = torch.nn.Parameter(torch.empty(4, 2, device="meta"))
        weight.weight_loader = default_weight_loader
        self.register_parameter("weight", weight)
        initialize_online_processing(self)
        bias = torch.nn.Parameter(torch.zeros(4))
        bias.weight_loader = default_weight_loader
        self.register_parameter("bias", bias)


def test_online_processing_waits_for_late_registered_bias():
    # Regression test: `bias` is skipped by the meta device paths, but it is
    # still loaded by a weight loader. Excluding it from the processing trigger
    # finalized the layer one load early, so the trailing bias was written into
    # an already-processed layer (e.g. over FP8 Marlin's permuted bias).
    quant_method = _RecordingQuantMethod()
    layer = _LateBiasLayer(quant_method)
    loaded_bias = torch.full((4,), 3.0)

    layer.weight.weight_loader(layer.weight, torch.full((4, 2), 2.0))
    assert quant_method.bias_at_process is None

    layer.bias.weight_loader(layer.bias, loaded_bias)
    assert quant_method.bias_at_process is not None
    assert torch.equal(quant_method.bias_at_process, loaded_bias)


def test_layerwise_reload_skips_non_persistent_parameter_alias_buffers(monkeypatch):
    layer = _AliasedBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.weight, loaded_weight)
    assert layer.weight_view.untyped_storage().data_ptr() == (
        layer.weight.untyped_storage().data_ptr()
    )
    assert "weight_view" in layer._non_persistent_buffers_set
    assert "0.weight_view" not in model.state_dict()


def test_capture_layer_to_meta_skips_uninitialized_parameter_storage_ptrs():
    layer = _AliasedBufferWithUninitializedChildLayer()

    _, buffers = capture_layer_to_meta(layer)

    assert "weight_view" not in buffers


def test_layerwise_reload_skips_child_parameter_alias_buffers(monkeypatch):
    layer = _ParentAliasedChildBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_conv = torch.full_like(layer.conv1d.weight, 7.0)
    loaded_scale = torch.full_like(layer.scale, 3.0)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.conv1d.weight.weight_loader(layer.conv1d.weight, loaded_conv)
    layer.scale.weight_loader(layer.scale, loaded_scale)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.conv1d.weight, loaded_conv)
    assert torch.equal(layer.conv_weights, loaded_conv.view(-1))
    assert layer.conv_weights.untyped_storage().data_ptr() == (
        layer.conv1d.weight.untyped_storage().data_ptr()
    )
    assert "conv_weights" in layer._non_persistent_buffers_set
    assert "0.conv_weights" not in model.state_dict()


def test_layerwise_reload_restores_alias_buffer_on_zero_size_layer(monkeypatch):
    layer = _ChildAliasOnlyBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_conv = torch.full_like(layer.conv1d.weight, 7.0)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.conv1d.weight.weight_loader(layer.conv1d.weight, loaded_conv)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.conv_weights, loaded_conv.view(-1))
    assert layer.conv_weights.untyped_storage().data_ptr() == (
        layer.conv1d.weight.untyped_storage().data_ptr()
    )
    assert "conv_weights" in layer._non_persistent_buffers_set
    assert "0.conv_weights" not in model.state_dict()


def test_layerwise_reload_preserves_unloaded_non_persistent_buffers(monkeypatch):
    layer = _NonPersistentBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)
    original_scale = layer.scale.clone()

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.weight, loaded_weight)
    assert torch.equal(layer.scale, original_scale)
    assert "scale" in layer._non_persistent_buffers_set
    assert "0.scale" not in model.state_dict()


def test_layerwise_reload_updates_loaded_non_persistent_buffers(monkeypatch):
    layer = _NonPersistentBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)
    loaded_scale = torch.full_like(layer.scale, 0.5)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)
    layer.scale.weight_loader(layer.scale, loaded_scale)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.weight, loaded_weight)
    assert torch.equal(layer.scale, loaded_scale)
    assert "scale" in layer._non_persistent_buffers_set
    assert "0.scale" not in model.state_dict()


@pytest.fixture
def hpc_rope_norm(monkeypatch, default_vllm_config):
    """Import HpcRopeNorm with the external ``hpc`` package stubbed out."""
    if "hpc" not in sys.modules:
        stub = types.ModuleType("hpc")
        stub.__spec__ = importlib.machinery.ModuleSpec("hpc", loader=None)
        stub.QuantType = types.SimpleNamespace(  # type: ignore[attr-defined]
            QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR=types.SimpleNamespace(value=0)
        )
        monkeypatch.setitem(sys.modules, "hpc", stub)
    from vllm.model_executor.layers.hpc import rope_norm

    monkeypatch.setattr(rope_norm, "_hpc_rope_norm_instances", {})
    return rope_norm


def test_hpc_rope_norm_kernel_sees_refit_norm_weights(monkeypatch, hpc_rope_norm):
    """The fused HPC kernel is handed the live QK-norm weights after a refit.

    Drives the production ``_forward_impl`` with a recording ``hpc`` stub. The
    Q/K norm weights it receives must be the model's own float32 parameters,
    so a layerwise reload that rewrites them in place (same storage) is what
    the kernel sees. Previously the kernel read separate mirrors that no
    reload path refreshed.
    """
    from vllm.model_executor.layers.layernorm import RMSNorm

    head_dim, num_heads, num_kv_heads, block_size = 128, 8, 1, 4
    layer = torch.nn.Module()
    layer.q_norm = RMSNorm(head_dim, 1e-6, dtype=torch.float32)
    layer.k_norm = RMSNorm(head_dim, 1e-6, dtype=torch.float32)
    layer.hpc_rope_norm = hpc_rope_norm.HpcRopeNorm(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cos_sin_cache=torch.ones(16, head_dim),
        use_qk_norm=True,
        fallback_qnorm=layer.q_norm,
        fallback_knorm=layer.k_norm,
        kv_cache_dtype="auto",
        layer_name="hpc_test_layer",
    )
    model = torch.nn.Sequential(layer)
    rnorm = layer.hpc_rope_norm

    calls: list[dict] = []

    def record(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(sys.modules["hpc"], "rope_norm_store_kv", record, raising=False)

    def kernel_norm_weights():
        q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
        qkv = torch.zeros(1, q_size + 2 * kv_size, dtype=torch.bfloat16)
        kv_cache = torch.zeros(
            2, num_kv_heads, block_size, 2 * head_dim, dtype=torch.bfloat16
        )
        attn_layer = types.SimpleNamespace(
            _k_scale=torch.ones(1), _v_scale=torch.ones(1)
        )
        attn_metadata = types.SimpleNamespace(
            num_actual_tokens=1,
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=0,
            num_prefill_tokens=0,
            max_query_len=1,
            decode_query_len=1,
            qo_indptr=None,
            qo_indptr_decode=None,
            slot_mapping=torch.tensor([4]),
            seq_lens=torch.tensor([1]),
            block_table_tensor=torch.tensor([[1]]),
            hpc_kv_written=False,
        )
        output = torch.zeros(1, q_size, dtype=torch.bfloat16)
        rnorm._forward_impl(qkv, kv_cache, attn_metadata, attn_layer, output)
        return calls[-1]["q_norm_weight"], calls[-1]["k_norm_weight"]

    def loaded(value):
        return torch.full((head_dim,), value, dtype=torch.bfloat16)

    default_weight_loader(layer.q_norm.weight, loaded(0.5))
    default_weight_loader(layer.k_norm.weight, loaded(0.25))
    q, k = kernel_norm_weights()
    assert torch.equal(q, loaded(0.5).float())
    assert torch.equal(k, loaded(0.25).float())
    q_ptr, k_ptr = q.data_ptr(), k.data_ptr()

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.q_norm.weight.weight_loader(layer.q_norm.weight, loaded(2.0))
    layer.k_norm.weight.weight_loader(layer.k_norm.weight, loaded(3.0))
    finalize_layerwise_reload(model, model_config=None)

    q, k = kernel_norm_weights()
    assert q.dtype == k.dtype == torch.float32
    assert torch.equal(q, loaded(2.0).float())
    assert torch.equal(k, loaded(3.0).float())
    assert (q.data_ptr(), k.data_ptr()) == (q_ptr, k_ptr)
    assert not hasattr(rnorm, "qnorm_weight")


@pytest.mark.parametrize(
    "tp_size", [pytest.param(1), pytest.param(2, marks=[pytest.mark.slow_test])]
)
@pytest.mark.parametrize(
    "base_model,mul_model,add_model",
    [
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/Qwen3-0.6B-FP8_BLOCK",
            "inference-optimization/Qwen3-0.6B-debug-multiply-FP8_BLOCK",
            "inference-optimization/Qwen3-0.6B-debug-add-FP8_BLOCK",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/Qwen3-0.6B-W4A16-G128",
            "inference-optimization/Qwen3-0.6B-debug-multiply-W4A16-G128",
            "inference-optimization/Qwen3-0.6B-debug-add-W4A16-G128",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC",
            "inference-optimization/DeepSeek-V3-debug-multiply-FP8_DYNAMIC",
            "inference-optimization/DeepSeek-V3-debug-add-FP8_DYNAMIC",
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty-NVFP4A16",
            "inference-optimization/DeepSeek-V3-debug-multiply-NVFP4A16",
            "inference-optimization/DeepSeek-V3-debug-add-NVFP4A16",
            marks=[pytest.mark.slow_test],
        ),
    ],
)
def test_reload_weights(base_model, mul_model, add_model, tp_size, vllm_runner):
    if current_platform.device_count() < tp_size:
        pytest.skip(reason="Not enough CUDA devices")

    if "FP8" in base_model and _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    with vllm_runner(
        model_name=base_model,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=(tp_size > 1 and "DeepSeek" in base_model),
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp


def test_kv_scale_reload(vllm_runner):
    """Test reloading a checkpoint that contains k_scale/v_scale weights."""
    if _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    model = "nm-testing/Llama-3.2-1B-Instruct-FP8-KV"

    # Load dummy weights, then reload real checkpoint
    with vllm_runner(
        model_name=model,
        load_format="dummy",
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc(
            "update_config",
            kwargs={"overrides": {"load_config": {"load_format": "auto"}}},
        )
        llm.collective_rpc("reload_weights", kwargs={"weights_path": model})
        reloaded_perp = llm.generate_prompt_perplexity(
            ["The capital of France is the city of Paris"],
            mask=["The capital of France is"],
        )[0]

    assert reloaded_perp < 10


@pytest.mark.parametrize(
    "tp_size", [pytest.param(1), pytest.param(2, marks=[pytest.mark.slow_test])]
)
@pytest.mark.parametrize(
    "base_model,mul_model,add_model,quantization",
    [
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            "fp8_per_tensor",
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            "fp8_per_tensor",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            "mxfp8",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            "mxfp8",
            marks=[
                pytest.mark.slow_test,
                pytest.mark.xfail(reason="mxfp4 & mla is not supported yet"),
            ],
        ),
    ],
)
def test_online_quantize_reload(
    base_model, mul_model, add_model, quantization, tp_size, vllm_runner
):
    if current_platform.device_count() < tp_size:
        pytest.skip(reason="Not enough GPU devices")

    if quantization == "fp8_per_tensor" and _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    with vllm_runner(
        model_name=base_model,
        quantization=quantization,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=(tp_size > 1 and "DeepSeek" in base_model),
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp
