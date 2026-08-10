# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from weakref import WeakKeyDictionary, ref

import pytest
import torch
from torch.nn.parameter import UninitializedParameter

import vllm.model_executor.model_loader.reload.layerwise as reload_layerwise
import vllm.model_executor.model_loader.reload.meta as reload_meta
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_processing,
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    initialize_online_processing,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.meta import (
    capture_layer_to_meta,
    materialize_layer,
    materialize_meta_tensor,
    restore_layer_on_meta,
    to_meta_tensor,
)
from vllm.model_executor.model_loader.reload.plan import (
    freeze_load_plan,
    get_load_plan,
    load_source,
)
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import (
    get_layer_tensors,
    get_loadable_layer_tensors,
)
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
        self.scale.weight_loader = default_weight_loader


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
        lambda w, perm, size_k, size_n, num_bits, is_a_8bit=False: torch.zeros(
            size_k // 16, size_n * 2, dtype=torch.int32
        ),
    )


def _make_act_order_marlin_kernel():
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
        has_g_idx=True,
    )
    kernel.w_q_name = "qweight"
    kernel.w_s_name = "scales"
    kernel.w_zp_name = None
    kernel.w_gidx_name = "g_idx"
    return kernel


def _load_marlin_checkpoint_format_weights(layer, g_idx):
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
        RowvLLMParameter,
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
    layer.g_idx = RowvLLMParameter(
        data=g_idx.clone(),
        input_dim=0,
        weight_loader=default_weight_loader,
    )


def _random_g_idx(generator):
    return torch.randint(
        0,
        _MARLIN_SIZE_K // _MARLIN_GROUP_SIZE,
        (_MARLIN_SIZE_K,),
        dtype=torch.int32,
        generator=generator,
    )


def test_marlin_post_load_preserves_runtime_tensor_addresses(monkeypatch, dist_init):
    """Marlin workspace and act-order sort indices must be recomputed into
    the same storage when weights are reloaded (RL weight sync), so device
    addresses captured by CUDA graphs remain valid."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    _stub_marlin_ops(monkeypatch)
    kernel = _make_act_order_marlin_kernel()

    generator = torch.Generator().manual_seed(0)
    first_g_idx = _random_g_idx(generator)
    second_g_idx = _random_g_idx(generator)

    layer = torch.nn.Module()
    _load_marlin_checkpoint_format_weights(layer, first_g_idx)
    kernel.process_weights_after_loading(layer)

    workspace_ptr = kernel.workspace.data_ptr()
    sort_indices_ptr = layer.g_idx_sort_indices.data_ptr()

    # Reload: fresh checkpoint-format tensors with a different act-order
    _load_marlin_checkpoint_format_weights(layer, second_g_idx)
    kernel.process_weights_after_loading(layer)

    assert kernel.workspace.data_ptr() == workspace_ptr
    assert torch.all(kernel.workspace == 0)
    assert layer.g_idx_sort_indices.data_ptr() == sort_indices_ptr
    expected_sort_indices = marlin_utils.marlin_sort_g_idx(second_g_idx)[1]
    assert torch.equal(layer.g_idx_sort_indices.data, expected_sort_indices)
    # registered as a Parameter so layerwise reload copy-back preserves it
    assert isinstance(layer.g_idx_sort_indices, torch.nn.Parameter)


@pytest.mark.parametrize("variant", ["fp8", "mxfp8", "nvfp4"])
def test_marlin_prepare_layer_preserves_workspace_address(monkeypatch, variant):
    """The Marlin fallback prepare_* functions rerun on weight reload and must
    reuse the workspace storage whose address captured CUDA graphs hold."""
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
        lambda b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit=False: torch.zeros(
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
    workspace_ptr = layer.workspace.data_ptr()

    # Reload: fresh checkpoint-format tensors, prepare runs again
    load_checkpoint_format_weights()
    prepare(layer)

    assert layer.workspace.data_ptr() == workspace_ptr
    assert torch.all(layer.workspace == 0)


def test_marlin_make_workspace_new_rejects_incompatible_existing(monkeypatch):
    """An incompatible existing workspace means the address captured by CUDA
    graphs is already unusable; allocating a replacement would hide that."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    device = torch.device("cpu")

    workspace = marlin_utils.marlin_make_workspace_new(device)
    reused = marlin_utils.marlin_make_workspace_new(device, existing=workspace)
    assert reused is workspace

    with pytest.raises(ValueError, match="incompatible"):
        marlin_utils.marlin_make_workspace_new(device, 4, existing=workspace)
    with pytest.raises(ValueError, match="incompatible"):
        marlin_utils.marlin_make_workspace_new(
            device, existing=workspace.to(torch.int64)
        )


def test_marlin_act_order_layerwise_reload_accounting(monkeypatch, dist_init):
    """`g_idx_sort_indices` is generated during weight processing and never
    loaded from checkpoints. Registering it as a Parameter must not count it
    toward `load_numel_total`: reload restores the construction-time tensor
    set before sizing, so act-order layers still process during streaming
    instead of deferring (and buffering weights) until finalization."""
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    _stub_marlin_ops(monkeypatch)
    kernel = _make_act_order_marlin_kernel()

    class _KernelQuantMethod(QuantizeMethodBase):
        def create_weights(self, layer, *args, **kwargs):
            raise NotImplementedError

        def apply(self, layer, *args, **kwargs):
            raise NotImplementedError

        def process_weights_after_loading(self, layer):
            kernel.process_weights_after_loading(layer)

    generator = torch.Generator().manual_seed(0)
    layer = torch.nn.Module()
    layer.quant_method = _KernelQuantMethod()
    _load_marlin_checkpoint_format_weights(layer, _random_g_idx(generator))

    # Metadata is recorded at model construction, before any processing
    record_metadata_for_reloading(layer)
    checkpoint_numel = sum(t.numel() for t in get_layer_tensors(layer).values())

    kernel.process_weights_after_loading(layer)
    sort_indices = layer.g_idx_sort_indices

    initialize_layerwise_reload(layer)
    info = get_layerwise_info(layer)
    assert info.load_numel_total == checkpoint_numel

    # Stream a new checkpoint; the layer must process as soon as its last
    # tensor arrives
    new_g_idx = _random_g_idx(generator)
    checkpoint = {
        "qweight": torch.zeros(_MARLIN_SIZE_K // 8, _MARLIN_SIZE_N, dtype=torch.int32),
        "scales": torch.ones(
            _MARLIN_SIZE_K // _MARLIN_GROUP_SIZE, _MARLIN_SIZE_N, dtype=torch.float16
        ),
        "g_idx": new_g_idx,
    }
    for name, weight in checkpoint.items():
        param = getattr(layer, name)
        param.weight_loader(param, weight)

    assert not info.can_load()
    assert not info.loaded_weights
    assert layer.g_idx_sort_indices is sort_indices
    expected_sort_indices = marlin_utils.marlin_sort_g_idx(new_g_idx)[1]
    assert torch.equal(layer.g_idx_sort_indices.data, expected_sort_indices)


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
    for name in ("A", "dt_bias", "D"):
        param = getattr(layer, name)
        with load_source(name):
            param.weight_loader(param, torch.zeros_like(param))
    freeze_load_plan(model)
    initialize_layerwise_reload(model)
    # Mimic real load_weights: resolve params once, then load in checkpoint
    # order with D last (the param that was dropped).
    params = dict(layer.named_parameters())
    for name in ("A", "dt_bias", "D"):
        param = params[name]
        with load_source(name):
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

    _initial_load(
        model,
        lambda: layer.weight.weight_loader(layer.weight, layer.weight.detach().clone()),
    )
    initialize_layerwise_reload(model)
    with load_source("test.weight"):
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
    with load_source("conv1d.weight"):
        layer.conv1d.weight.weight_loader(
            layer.conv1d.weight, layer.conv1d.weight.detach().clone()
        )
    with load_source("scale"):
        layer.scale.weight_loader(layer.scale, layer.scale.detach().clone())
    freeze_load_plan(model)
    initialize_layerwise_reload(model)
    with load_source("conv1d.weight"):
        layer.conv1d.weight.weight_loader(layer.conv1d.weight, loaded_conv)
    with load_source("scale"):
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

    _initial_load(
        model,
        lambda: layer.conv1d.weight.weight_loader(
            layer.conv1d.weight, layer.conv1d.weight.detach().clone()
        ),
    )
    initialize_layerwise_reload(model)
    with load_source("test.weight"):
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

    _initial_load(
        model,
        lambda: layer.weight.weight_loader(layer.weight, layer.weight.detach().clone()),
    )
    initialize_layerwise_reload(model)
    with load_source("test.weight"):
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
    with load_source("weight"):
        layer.weight.weight_loader(layer.weight, layer.weight.detach().clone())
    with load_source("scale"):
        layer.scale.weight_loader(layer.scale, layer.scale.detach().clone())
    freeze_load_plan(model)
    initialize_layerwise_reload(model)
    with load_source("weight"):
        layer.weight.weight_loader(layer.weight, loaded_weight)
    with load_source("scale"):
        layer.scale.weight_loader(layer.scale, loaded_scale)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.weight, loaded_weight)
    assert torch.equal(layer.scale, loaded_scale)
    assert "scale" in layer._non_persistent_buffers_set
    assert "0.scale" not in model.state_dict()


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
            "fp8",
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            "fp8",
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

    if quantization == "fp8" and _fp8_reload_unsupported():
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


class _PaddedDerivedLayer(torch.nn.Module):
    """Mirrors Kimi GDN, whose padding row and loader-derived buffer are storage
    no checkpoint tensor can write, putting an element count out of reach."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 2))
        self.register_buffer("derived", torch.zeros(5), persistent=False)
        derived = self.derived

        def loader(param, loaded_weight, loaded_shard_id=None):
            param.data[:3].copy_(loaded_weight)
            derived.fill_(float(loaded_weight.sum()))

        self.weight.weight_loader = loader


def _initial_load(model, apply_weights, source="test.weight"):
    """Run the recorded initial load and freeze the contract, as load_model does."""
    record_metadata_for_reloading(model)
    with load_source(source):
        apply_weights()
    freeze_load_plan(model)


class _ShardedExpertLayer(torch.nn.Module):
    """Stand-in for vLLM's shared MoE loader, which declines a non-local expert
    via `return_success` without touching storage."""

    def __init__(self, local_experts: list[int]):
        super().__init__()
        self.local_experts = local_experts
        self.w = torch.nn.Parameter(torch.zeros(len(local_experts), 2))

        def loader(param, loaded_weight, expert_id, return_success=False):
            if expert_id not in self.local_experts:
                return False if return_success else None
            param.data[self.local_experts.index(expert_id)].copy_(loaded_weight)
            return True if return_success else None

        self.w.weight_loader = loader


def _load_experts(layer, expert_ids, value, *, return_success=False):
    """Offer one application per expert, as a broadcasting sender does."""
    # Omitted rather than passed as False, so a loader predating the protocol
    # can be driven by the same helper.
    kwargs = {"return_success": True} if return_success else {}
    results = []
    for expert_id in expert_ids:
        with load_source(f"experts.{expert_id}.w"):
            results.append(
                layer.w.weight_loader(
                    layer.w, torch.full((2,), float(value)), expert_id, **kwargs
                )
            )
    return results


def test_non_local_expert_applications_are_absorbed():
    """A broadcast expert that is not local to this rank writes nothing, so it
    must not fail the update whether it lands before or after publish."""
    layer = _ShardedExpertLayer(local_experts=[0, 1])
    model = torch.nn.Sequential(layer)

    record_metadata_for_reloading(model)
    _load_experts(layer, [0, 1, 2, 3], 1.0, return_success=True)
    freeze_load_plan(model)

    initialize_layerwise_reload(model)
    # A contiguous expert map puts this rank's experts first.
    _load_experts(layer, [0, 1], 5.0, return_success=True)
    assert not layer.w.is_meta, "layer must publish on its own expected set"

    # So the trailing non-local applications land on a published layer.
    assert _load_experts(layer, [2, 3], 5.0, return_success=True) == [False, False]
    assert torch.equal(layer.w, torch.full((2, 2), 5.0))

    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(layer.w, torch.full((2, 2), 5.0))


def test_published_layer_absorbs_without_invoking_the_loader():
    """A published layer's storage is the live kernel tensors, so a decline
    must be decided from the contract rather than by running the loader to see
    what it returns."""
    layer = _ShardedExpertLayer(local_experts=[0, 1])
    invocations: list[int] = []
    inner = layer.w.weight_loader

    def counting_loader(param, loaded_weight, expert_id, return_success=False):
        invocations.append(expert_id)
        return inner(param, loaded_weight, expert_id, return_success=return_success)

    layer.w.weight_loader = counting_loader
    model = torch.nn.Sequential(layer)

    record_metadata_for_reloading(model)
    _load_experts(layer, [0, 1, 2, 3], 1.0, return_success=True)
    freeze_load_plan(model)

    initialize_layerwise_reload(model)
    _load_experts(layer, [0, 1], 5.0, return_success=True)
    assert not layer.w.is_meta, "layer must publish on its own expected set"

    invocations.clear()
    assert _load_experts(layer, [2, 3], 5.0, return_success=True) == [False, False]
    assert invocations == [], "the loader must not run against live storage"

    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(layer.w, torch.full((2, 2), 5.0))


class _QuietExpertLayer(torch.nn.Module):
    """An expert loader predating `return_success`, which can only ignore a
    non-local expert rather than report it."""

    def __init__(self, local_experts: list[int]):
        super().__init__()
        self.local_experts = local_experts
        self.w = torch.nn.Parameter(torch.zeros(len(local_experts), 2))

        def loader(param, loaded_weight, expert_id):
            if expert_id in self.local_experts:
                param.data[self.local_experts.index(expert_id)].copy_(loaded_weight)

        self.w.weight_loader = loader


def test_extra_applications_before_publish_do_not_fail_validation():
    """Completion is a lower bound, so an application counted outside the
    contract must be absorbed rather than fail the update."""
    layer = _QuietExpertLayer(local_experts=[0, 1])
    model = torch.nn.Sequential(layer)

    record_metadata_for_reloading(model)
    _load_experts(layer, [0, 1], 1.0)  # rank-filtered, as from disk
    freeze_load_plan(model)

    initialize_layerwise_reload(model)
    # Expert 2 is interleaved so it lands while the layer is still incomplete.
    _load_experts(layer, [0, 2, 1], 5.0)
    assert not layer.w.is_meta, "the expected set alone must publish the layer"

    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(layer.w, torch.full((2, 2), 5.0))


def test_declined_startup_application_stays_off_the_contract():
    """The startup load is offered every expert, not a rank-filtered subset.

    Recording an expert this rank declines would demand it on every reload,
    where the same loader declines it again and it is never observed.
    """
    layer = _ShardedExpertLayer(local_experts=[0, 1])
    model = torch.nn.Sequential(layer)

    record_metadata_for_reloading(model)
    _load_experts(layer, [0, 1, 2, 3], 1.0, return_success=True)
    freeze_load_plan(model)

    assert sum(get_load_plan(layer).values()) == 2, "only local experts wrote"

    initialize_layerwise_reload(model)
    _load_experts(layer, [0, 1, 2, 3], 5.0, return_success=True)
    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(layer.w, torch.full((2, 2), 5.0))


def test_loader_return_value_reaches_the_caller():
    """14 MoE models branch on `return_success`; the wrapper must not swallow it."""
    layer = _ShardedExpertLayer(local_experts=[0])
    model = torch.nn.Sequential(layer)

    record_metadata_for_reloading(model)
    _load_experts(layer, [0], 1.0)
    freeze_load_plan(model)

    initialize_layerwise_reload(model)
    assert _load_experts(layer, [0, 1], 5.0, return_success=True) == [True, False]


def test_write_after_publish_without_decline_protocol_raises():
    """A loader that cannot decline has no business writing to a published layer."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4))
    layer.weight.weight_loader = default_weight_loader
    model = torch.nn.Sequential(layer)

    _initial_load(
        model, lambda: layer.weight.weight_loader(layer.weight, torch.ones(4))
    )

    initialize_layerwise_reload(model)
    with load_source("test.weight"):
        layer.weight.weight_loader(layer.weight, torch.full((4,), 5.0))
    assert not layer.weight.is_meta, "single-source layer publishes immediately"

    with (
        load_source("other.weight"),
        pytest.raises(RuntimeError, match="after the layer completed"),
    ):
        layer.weight.weight_loader(layer.weight, torch.full((4,), 9.0))


def test_load_plan_records_shard_selectors():
    """The contract is a multiset of loader applications keyed by destination
    and selector, so one packed parameter fanning in from N checkpoint tensors
    requires all N on reload."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4))
    calls = []
    layer.weight.weight_loader = lambda param, w, loaded_shard_id=None: calls.append(
        loaded_shard_id
    )
    model = torch.nn.Sequential(layer)

    _initial_load(
        model,
        lambda: [
            layer.weight.weight_loader(layer.weight, torch.ones(2), shard)
            for shard in ("q", "k", "q")
        ],
    )

    plan = get_load_plan(layer)
    assert plan == {
        ("test.weight", "weight", (("loaded_shard_id", "q"),)): 2,
        ("test.weight", "weight", (("loaded_shard_id", "k"),)): 1,
    }
    assert calls == ["q", "k", "q"]


def test_payload_dtype_does_not_change_the_contract():
    """An fp32 payload where the checkpoint held bf16 is the same loader
    application, so it must satisfy the contract rather than read as missing."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    layer.weight.weight_loader = lambda param, w: param.data.copy_(w)
    model = torch.nn.Sequential(layer)

    _initial_load(
        model,
        lambda: layer.weight.weight_loader(
            layer.weight, torch.ones(4, dtype=torch.bfloat16)
        ),
    )

    initialize_layerwise_reload(model)
    with load_source("test.weight"):
        layer.weight.weight_loader(layer.weight, torch.full((4,), 5.0))

    assert reload_layerwise.get_layerwise_info(layer).is_complete()
    assert not layer.weight.is_meta


def test_freeze_load_plan_removes_recorders():
    """Recorders must not survive into serving: a leaked wrapper would keep
    recording and would shadow the real loader identity."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4))
    original = lambda param, w: param.data.copy_(w)  # noqa: E731
    layer.weight.weight_loader = original
    layer.no_loader = torch.nn.Parameter(torch.zeros(2))
    model = torch.nn.Sequential(layer)

    record_metadata_for_reloading(model)
    assert layer.weight.weight_loader is not original

    with load_source("test.weight"):
        layer.weight.weight_loader(layer.weight, torch.ones(4))
    freeze_load_plan(model)

    assert layer.weight.weight_loader is original
    # A tensor that never had a loader must not acquire one.
    assert not hasattr(layer.no_loader, "weight_loader")


def test_padded_derived_layer_completes_online_without_annotation():
    """A layer with padding and a derived buffer must complete during the stream
    with no model-side declaration, or it stays buffered through finalize."""
    layer = _PaddedDerivedLayer()
    model = torch.nn.Sequential(layer)
    initial = torch.full((3, 2), 7.0)
    _initial_load(model, lambda: layer.weight.weight_loader(layer.weight, initial))

    initialize_layerwise_reload(model)
    reloaded = torch.full((3, 2), 9.0)
    with load_source("test.weight"):
        layer.weight.weight_loader(layer.weight, reloaded)

    # Processed during the stream, not deferred: materialized and buffers freed.
    assert not layer.weight.is_meta
    assert not reload_layerwise.get_layerwise_info(layer).loaded_weights
    assert torch.equal(layer.weight[:3], reloaded)
    assert torch.equal(layer.weight[3], torch.zeros(2))
    assert torch.equal(layer.derived, torch.full_like(layer.derived, 54.0))

    finalize_layerwise_processing(model, model_config=None)
    assert torch.equal(layer.weight[:3], reloaded)


def test_runtime_only_buffers_are_not_loadable_destinations():
    """Completion must ignore non-persistent buffers without a loader."""
    layer = _PaddedDerivedLayer()

    assert set(get_loadable_layer_tensors(layer)) == {"weight"}
    assert "derived" in get_layer_tensors(layer)


def test_incomplete_update_defers_then_fails_closed():
    """A partial update must not publish the layer, and must not commit."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4))
    layer.weight.weight_loader = lambda param, w, loaded_shard_id=None: (
        param.data.copy_(w)
    )
    model = torch.nn.Sequential(layer)

    _initial_load(
        model,
        lambda: [
            layer.weight.weight_loader(layer.weight, torch.ones(4), shard)
            for shard in ("q", "k")
        ],
    )

    initialize_layerwise_reload(model)
    with load_source("test.weight"):
        layer.weight.weight_loader(layer.weight, torch.full((4,), 5.0), "q")

    info = reload_layerwise.get_layerwise_info(layer)
    assert not info.is_complete()
    assert layer.weight.is_meta, "layer must stay deferred until finalization"

    with pytest.raises(RuntimeError, match="'k'"):
        finalize_layerwise_processing(model, model_config=None)


def test_reload_without_contract_fails_closed():
    """An update without a startup contract must not guess or mutate meta data."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4))
    layer.weight.weight_loader = default_weight_loader
    model = torch.nn.Sequential(layer)

    record_metadata_for_reloading(model)
    freeze_load_plan(model)  # nothing was loaded, so no contract is recorded
    assert get_load_plan(layer) is None

    initialize_layerwise_reload(model)
    with load_source("test.weight"):
        layer.weight.weight_loader(layer.weight, torch.full((4,), 5.0))

    # No contract means no completion, so the transaction must not commit.
    with pytest.raises(RuntimeError, match="no startup contract"):
        finalize_layerwise_reload(model, model_config=None)


def test_load_plan_distinguishes_canonical_sources():
    """A duplicate source cannot substitute for another source with the same I/O."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4))
    layer.weight.weight_loader = default_weight_loader
    model = torch.nn.Sequential(layer)

    record_metadata_for_reloading(model)
    for source in ("source_a.weight", "source_b.weight"):
        with load_source(source):
            layer.weight.weight_loader(layer.weight, torch.ones(4))
    freeze_load_plan(model)

    initialize_layerwise_reload(model)
    for value in (2.0, 3.0):
        with load_source("source_b.weight"):
            layer.weight.weight_loader(layer.weight, torch.full((4,), value))

    # Sending one source twice is absorbed rather than rejected, but it cannot
    # stand in for the source that never arrived, so the layer stays deferred.
    info = reload_layerwise.get_layerwise_info(layer)
    assert not info.is_complete()
    assert layer.weight.is_meta


def test_load_source_reaches_a_nested_custom_loader():
    """`AutoWeightsLoader` hands a derived iterator to a child's `load_weights`,
    so one tag at the model boundary is still active when the child loads."""
    from vllm.model_executor.models.utils import AutoWeightsLoader

    class Child(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4))
            self.weight.weight_loader = default_weight_loader

        def load_weights(self, weights):
            loaded = set()
            for name, weight in weights:
                self.weight.weight_loader(self.weight, weight)
                loaded.add(name)
            return loaded

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Child()

        def load_weights(self, weights):
            return AutoWeightsLoader(self).load_weights(weights)

    model = Parent()
    record_metadata_for_reloading(model)
    model.load_weights([("child.weight", torch.ones(4))])
    freeze_load_plan(model)

    assert get_load_plan(model.child) == {("child.weight", "weight", ()): 1}


def test_model_load_weights_propagates_source_to_direct_custom_loader():
    """The root input stream covers models that do not use AutoWeightsLoader."""

    class DirectModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4))
            self.weight.weight_loader = default_weight_loader

        def load_weights(self, weights):
            for _, loaded_weight in weights:
                self.weight.weight_loader(self.weight, loaded_weight)

    model = DirectModel()
    record_metadata_for_reloading(model)
    model.load_weights([("canonical.weight", torch.ones(4))])
    freeze_load_plan(model)

    plan = get_load_plan(model)
    assert plan is not None
    assert {key[0] for key in plan} == {"canonical.weight"}

    initialize_layerwise_reload(model)
    model.load_weights([("canonical.weight", torch.full((4,), 7.0))])
    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(model.weight, torch.full((4,), 7.0))


def test_applied_layer_rejects_trailing_application_until_finish():
    """Online copy-back must not remove the transaction guard."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4))
    layer.weight.weight_loader = default_weight_loader
    model = torch.nn.Sequential(layer)
    _initial_load(
        model,
        lambda: layer.weight.weight_loader(layer.weight, torch.ones(4)),
    )

    initialize_layerwise_reload(model)
    staged_param = layer.weight
    staged_loader = staged_param.weight_loader
    with load_source("test.weight"):
        staged_loader(staged_param, torch.full((4,), 5.0))

    info = reload_layerwise.get_layerwise_info(layer)
    assert info.applied
    assert torch.equal(layer.weight, torch.full((4,), 5.0))

    with (
        load_source("test.weight"),
        pytest.raises(RuntimeError, match="after the layer completed"),
    ):
        layer.weight.weight_loader(layer.weight, torch.full((4,), 9.0))
    with (
        load_source("test.weight"),
        pytest.raises(RuntimeError, match="after the layer completed"),
    ):
        staged_loader(staged_param, torch.full((4,), 9.0))
    assert torch.equal(layer.weight, torch.full((4,), 5.0))

    finalize_layerwise_reload(model, model_config=None)
    assert not reload_layerwise.get_layerwise_info(layer).can_load()


def test_frozen_plan_is_reused_across_transactions():
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4))
    layer.weight.weight_loader = default_weight_loader
    model = torch.nn.Sequential(layer)
    _initial_load(
        model,
        lambda: layer.weight.weight_loader(layer.weight, torch.ones(4)),
    )

    for value in (3.0, 7.0):
        initialize_layerwise_reload(model)
        with load_source("test.weight"):
            layer.weight.weight_loader(layer.weight, torch.full((4,), value))
        finalize_layerwise_reload(model, model_config=None)
        assert torch.equal(layer.weight, torch.full((4,), value))


def test_online_processing_armed_before_the_recorder_still_gets_a_plan():
    """Online quantization arms a layer before the recorder is installed, and
    the recording must still reach it or every reload falls back to finalize."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4))
    layer.weight.weight_loader = default_weight_loader
    model = torch.nn.Sequential(layer)

    initialize_online_processing(layer)
    record_metadata_for_reloading(model)
    with load_source("test.weight"):
        layer.weight.weight_loader(layer.weight, torch.ones(4))
    freeze_load_plan(model)

    assert get_load_plan(layer)

    initialize_layerwise_reload(model)
    with load_source("test.weight"):
        layer.weight.weight_loader(layer.weight, torch.full((4,), 5.0))
    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(layer.weight, torch.full((4,), 5.0))
