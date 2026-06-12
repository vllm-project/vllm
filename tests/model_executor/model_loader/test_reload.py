# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import inspect
from weakref import WeakKeyDictionary, ref

import pytest
import torch
from torch.nn.parameter import UninitializedParameter

import vllm.model_executor.model_loader.reload.meta as reload_meta
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    make_load_weights_safe_for_reload,
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


class _AliasedBufferWithUninitializedChildLayer(_AliasedBufferLayer):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Module()
        self.child.register_parameter(
            "lazy_weight", UninitializedParameter(requires_grad=False)
        )


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


class _LayoutSwapMethod(QuantizeMethodBase):
    """Mimics `UnquantizedFusedMoEMethod` for the FlashInfer CUTLASS backend.

    `process_weights_after_loading` swaps the two halves of `layer.weight`
    in place, modelling `swap_w13_to_w31`. The per-shard weight loader writes
    the first half on `shard_id="w1"` and the second half on `shard_id="w3"`
    — same convention as `FusedMoE._load_w13`. The combination triggers
    https://github.com/vllm-project/vllm/issues/42821 when `load_weights`
    is called a second time without re-routing through the layerwise reload
    pipeline.
    """

    def create_weights(self, layer, *args, **kwargs):  # pragma: no cover
        return

    def apply(self, layer, *args, **kwargs):  # pragma: no cover
        return layer.weight

    def process_weights_after_loading(self, layer):
        # Swap halves of layer.weight in place (analog of `swap_w13_to_w31`).
        # After loading checkpoint-format `[w1; w3]`, the swap produces
        # `[w3; w1]` which is the kernel-expected layout.
        n = layer.weight.shape[0] // 2
        swapped = torch.cat(
            [layer.weight.data[n:].clone(), layer.weight.data[:n].clone()], dim=0
        )
        layer.weight.data.copy_(swapped)


class _LayoutSwapLayer(torch.nn.Module):
    """Layer with a sharded checkpoint weight loader + destructive process step."""

    def __init__(self, half_size: int = 2):
        super().__init__()
        self._half_size = half_size
        self.weight = torch.nn.Parameter(
            torch.zeros(2 * half_size, dtype=torch.float32)
        )

        def shard_loader(param, loaded_weight, shard_id):
            if shard_id == "w1":
                param.data[:half_size].copy_(loaded_weight)
            else:
                assert shard_id == "w3"
                param.data[half_size:].copy_(loaded_weight)

        self.weight.weight_loader = shard_loader
        self.quant_method = _LayoutSwapMethod()


class _LayoutSwapModel(torch.nn.Module):
    """Tiny model with a single `_LayoutSwapLayer` for regression testing."""

    def __init__(self):
        super().__init__()
        self.layer = _LayoutSwapLayer()

    def load_weights(self, weights):
        loaded = set()
        for name, value, shard_id in weights:
            assert name == "layer.weight"
            self.layer.weight.weight_loader(
                self.layer.weight, value, shard_id=shard_id
            )
            loaded.add(name)
        return loaded


def test_make_load_weights_safe_for_reload_is_idempotent():
    """Re-wrapping `model.load_weights` is a no-op.

    Guards against accumulating layers of `initialize_layerwise_reload`
    indirection if a loader's `load_model` is invoked more than once on
    the same model instance.
    """
    model = _LayoutSwapModel()
    record_metadata_for_reloading(model)

    make_load_weights_safe_for_reload(model, model_config=None)
    wrapped_once = model.load_weights
    assert getattr(wrapped_once, "_vllm_safe_reload_wrapped", False)

    make_load_weights_safe_for_reload(model, model_config=None)
    assert model.load_weights is wrapped_once


def test_load_weights_idempotent_under_destructive_process_step():
    """Regression test for https://github.com/vllm-project/vllm/issues/42821.

    Calling `model.load_weights` a second time with the same checkpoint must
    not silently corrupt parameters whose `process_weights_after_loading`
    rewrites their layout. Without the wrapper, the second invocation writes
    checkpoint-format bytes into the swapped-layout buffer and the layer
    drifts away from its post-init state on every reload.
    """
    model = _LayoutSwapModel()
    record_metadata_for_reloading(model)

    # Initial load: checkpoint provides [w1=(1,2), w3=(3,4)] which yields
    # the checkpoint-format buffer [1, 2, 3, 4]. The layout swap then
    # produces the kernel-format buffer [3, 4, 1, 2].
    initial_weights = [
        ("layer.weight", torch.tensor([1.0, 2.0]), "w1"),
        ("layer.weight", torch.tensor([3.0, 4.0]), "w3"),
    ]
    model.load_weights(iter(initial_weights))
    model.layer.quant_method.process_weights_after_loading(model.layer)

    post_init_state = model.layer.weight.data.clone()
    assert torch.equal(post_init_state, torch.tensor([3.0, 4.0, 1.0, 2.0]))

    # Without the wrapper, a second `load_weights` writes raw [1, 2, 3, 4]
    # into the swapped-layout buffer, leaving it inconsistent with the
    # kernel's expected [3, 4, 1, 2] layout.
    make_load_weights_safe_for_reload(model, model_config=None)
    model.load_weights(iter(initial_weights))

    assert torch.equal(model.layer.weight.data, post_init_state), (
        f"Reload corrupted parameter layout: got {model.layer.weight.data}, "
        f"expected {post_init_state}"
    )

    # Idempotency across many reloads.
    for _ in range(3):
        model.load_weights(iter(initial_weights))
        assert torch.equal(model.layer.weight.data, post_init_state)


def test_safe_reload_wrapper_preserves_kernel_storage_address():
    """The wrapper preserves the parameter's storage `data_ptr` across reload.

    This is critical for captured CUDA graphs in RL weight-update loops.
    """
    model = _LayoutSwapModel()
    record_metadata_for_reloading(model)

    initial_weights = [
        ("layer.weight", torch.tensor([1.0, 2.0]), "w1"),
        ("layer.weight", torch.tensor([3.0, 4.0]), "w3"),
    ]
    model.load_weights(iter(initial_weights))
    model.layer.quant_method.process_weights_after_loading(model.layer)
    storage_before = model.layer.weight.untyped_storage().data_ptr()

    make_load_weights_safe_for_reload(model, model_config=None)
    model.load_weights(iter(initial_weights))
    storage_after = model.layer.weight.untyped_storage().data_ptr()

    assert storage_before == storage_after, (
        "Wrapper must preserve parameter storage address across reload."
    )


def test_safe_reload_wrapper_finalizes_on_loader_exception():
    """If the inner `load_weights` raises, the wrapper still runs finalize.

    The `finally` branch must call `finalize_layerwise_reload` so that
    per-layer `info` is reset; otherwise the next `load_weights` call would
    short-circuit `initialize_layerwise_reload` (which skips layers whose
    `info.can_load()` is already True), causing wedged reload state.
    """
    model = _LayoutSwapModel()
    record_metadata_for_reloading(model)

    initial_weights = [
        ("layer.weight", torch.tensor([1.0, 2.0]), "w1"),
        ("layer.weight", torch.tensor([3.0, 4.0]), "w3"),
    ]
    model.load_weights(iter(initial_weights))
    model.layer.quant_method.process_weights_after_loading(model.layer)
    post_init_state = model.layer.weight.data.clone()

    make_load_weights_safe_for_reload(model, model_config=None)

    class _ExplodingIter:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("simulated checkpoint read failure")

    with pytest.raises(RuntimeError, match="simulated checkpoint read failure"):
        model.load_weights(_ExplodingIter())

    # The next successful reload must produce the same post-init state,
    # i.e. the wrapper recovers cleanly from the exception (`info` was
    # reset by the `finally`-clause finalize so the second reload runs
    # the full pipeline rather than short-circuiting).
    model.load_weights(iter(initial_weights))
    assert torch.equal(model.layer.weight.data, post_init_state), (
        f"Wrapper failed to recover after exception: "
        f"got {model.layer.weight.data}, expected {post_init_state}"
    )


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
