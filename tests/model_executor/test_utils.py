# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.model_executor.parameter import ModelWeightParameter, PackedvLLMParameter
from vllm.model_executor.utils import replace_parameter, set_derived_buffer


@pytest.fixture
def single_rank_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    """`BasevLLMParameter.__init__` queries the TP group, which is not
    initialized in a unit test. Pin it to a single rank.
    """
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )


@pytest.mark.parametrize("prefer_copy", [False, True])
@pytest.mark.parametrize("wrap_in_parameter", [False, True])
def test_replace_parameter_preserves_custom_attribute(
    prefer_copy: bool, wrap_in_parameter: bool
) -> None:
    """`replace_parameter` must carry over attributes attached to the
    replacement tensor (e.g. the `is_shuffled` flag set by AITER weight
    preprocessing in `Fp8MoEMethod.process_weights_after_loading`).

    The replacement is passed both as a plain `torch.Tensor` and as a
    `torch.nn.Parameter`, since the latter is unwrapped through `.data`
    internally, which does not carry attributes over either.
    """
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    )
    original_data_ptr = layer.weight.data_ptr()

    new_data: torch.Tensor = torch.ones(4, 4).t()
    if wrap_in_parameter:
        new_data = torch.nn.Parameter(new_data, requires_grad=False)
    new_data.is_shuffled = True

    # Sanity check on the assumption above: tensor internals are not
    # reachable through `__dict__`, only user-set custom attributes are.
    assert new_data.__dict__.keys() == {"is_shuffled"}

    replace_parameter(layer, "weight", new_data, prefer_copy=prefer_copy)

    assert isinstance(layer.weight, torch.nn.Parameter)
    assert layer.weight.is_shuffled is True
    assert torch.equal(layer.weight.data, new_data)
    assert layer.weight.device == new_data.device

    if prefer_copy:
        # The existing storage is reused, so addresses captured in CUDA graphs
        # stay valid across the update.
        assert layer.weight.data_ptr() == original_data_ptr
    else:
        assert layer.weight.stride() == new_data.stride()
        assert layer.weight.data_ptr() == new_data.data_ptr()


@pytest.mark.parametrize("prefer_copy", [False, True])
def test_replace_parameter_preserves_weight_loader(prefer_copy: bool) -> None:
    """The reload path must survive replacement: the old parameter's
    `weight_loader` is carried over as-is, so it is still invoked as
    `weight_loader(param, loaded_weight)`.

    Real loaders are frequently bound methods of the layer (`RoutedExperts`
    hands `self.weight_loader` to `create_weights`); what must not happen is
    the carry-over re-binding the loader to the new parameter, which would
    shift `loaded_weight` into the `param` slot.
    """
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        calls.append((param, loaded_weight))

    layer = torch.nn.Module()
    old_param = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    old_param.weight_loader = weight_loader
    layer.register_parameter("weight", old_param)

    replace_parameter(layer, "weight", torch.ones(4, 4), prefer_copy=prefer_copy)

    # Re-binding to the new parameter would shift `loaded_weight` into the
    # `param` slot, so the plain function must not have grown a `__self__`.
    assert not hasattr(layer.weight.weight_loader, "__self__")

    loaded_weight = torch.full((4, 4), 2.0)
    layer.weight.weight_loader(layer.weight, loaded_weight)

    assert len(calls) == 1
    assert calls[0][0] is layer.weight
    assert calls[0][1] is loaded_weight


def test_replace_parameter_weight_loader_comes_from_old_parameter() -> None:
    """`weight_loader` is deliberately excluded from the attribute carry-over:
    the old parameter's loader is authoritative. A loader riding along on the
    replacement tensor must neither shadow it nor trip the overwrite assertion
    in `set_weight_attrs`, and the other attributes must still be carried over.

    `_weight_loader` is excluded for the same reason: it is the backing field
    of `BasevLLMParameter.weight_loader`, so leaving it in would smuggle a
    stale loader past the `weight_loader` exclusion.
    """
    calls: list[str] = []

    def old_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        calls.append("old")

    def stale_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        calls.append("stale")

    layer = torch.nn.Module()
    old_param = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    old_param.weight_loader = old_weight_loader
    layer.register_parameter("weight", old_param)

    new_data = torch.ones(4, 4)
    new_data.weight_loader = stale_weight_loader
    new_data._weight_loader = stale_weight_loader
    new_data.is_shuffled = True

    replace_parameter(layer, "weight", new_data)

    layer.weight.weight_loader(layer.weight, torch.full((4, 4), 2.0))

    assert calls == ["old"]
    assert not hasattr(layer.weight, "_weight_loader")
    assert layer.weight.is_shuffled is True


def test_replace_parameter_does_not_rebind_plain_function_attribute() -> None:
    """A plain function stored on the replacement tensor must be carried over
    verbatim; re-binding it as a method of the new parameter would silently
    shift the arguments of every subsequent call.
    """

    def scale_for(group_size: int) -> int:
        return group_size * 2

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    )

    new_data = torch.ones(4, 4)
    new_data.scale_for = scale_for

    replace_parameter(layer, "weight", new_data)

    assert layer.weight.scale_for is scale_for
    assert layer.weight.scale_for(3) == 6


def test_replace_parameter_preserves_bound_method_attribute() -> None:
    """A callable already bound to another object must keep pointing at that
    object after replacement.
    """

    class KernelDispatcher:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def record(self, value: int) -> None:
            self.calls.append(value)

    dispatcher = KernelDispatcher()

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    )

    new_data = torch.ones(4, 4)
    new_data.record = dispatcher.record

    replace_parameter(layer, "weight", new_data)

    assert layer.weight.record.__self__ is dispatcher
    layer.weight.record(7)
    assert dispatcher.calls == [7]


@pytest.mark.parametrize("param_kind", ["plain", "model_weight", "packed"])
def test_replace_parameter_attributes_from_the_layers_own_parameter(
    single_rank_tp: None, param_kind: str
) -> None:
    """Several callers hand back the parameter they were given, after mutating
    its `.data` in place: the HUMMING branch of
    `convert_to_fp8_moe_kernel_format` (`w13 = layer.w13_weight`), the
    `AITER_MXFP4_BF16` branch of
    `convert_gpt_oss_weight_to_mxfp4_moe_kernel_format`, the no-transpose
    branch of `XPUFP8ScaledMM` (`layer_weight = w`), and `auto_awq`/`auto_gptq`,
    which pass whatever is currently registered -- possibly still a
    `BasevLLMParameter` subclass rather than a plain `Parameter`.

    `new_data is old_param` there, so every attribute of the replacement is by
    definition an attribute of the old parameter; pin exactly which survive.

    For the vLLM parameter classes only the private backing fields come along.
    The public names weight loading branches on -- `output_dim`, `input_dim`,
    `packed_dim` (`getattr(param, "output_dim", None)` in `linear.py`) -- are
    class-level properties, so they do not survive onto the plain replacement
    and cannot be read back stale after the layout has changed.
    """

    class Layer(torch.nn.Module):
        def weight_loader(
            self, param: torch.Tensor, loaded_weight: torch.Tensor
        ) -> None:
            pass

    layer = Layer()
    data = torch.zeros(4, 4)
    loader = layer.weight_loader
    vllm_param_attrs = {
        "_weight_loader": loader,
        "_input_dim": 1,
        "_output_dim": 0,
        "tp_rank": 0,
        "tp_size": 1,
    }

    old_param: torch.nn.Parameter
    if param_kind == "plain":
        old_param = torch.nn.Parameter(data, requires_grad=False)
        old_param.weight_loader = loader
        # MoE scale marker, read back by `RoutedExperts.weight_loader`.
        old_param.quant_method = "block"
        source_attrs = {"weight_loader": loader, "quant_method": "block"}
    elif param_kind == "model_weight":
        old_param = ModelWeightParameter(
            data=data, input_dim=1, output_dim=0, weight_loader=loader
        )
        source_attrs = dict(vllm_param_attrs)
    else:
        old_param = PackedvLLMParameter(
            data=data,
            input_dim=1,
            output_dim=0,
            packed_dim=0,
            packed_factor=8,
            weight_loader=loader,
        )
        source_attrs = dict(vllm_param_attrs)
        source_attrs |= {
            "_packed_dim": 0,
            "_packed_factor": 8,
            "_marlin_tile_size": None,
        }

    layer.register_parameter("weight", old_param)
    old_param.data = torch.ones(4, 4)

    # Pin what the caller hands back: on the vLLM classes `weight_loader` is a
    # property, so the loader sits under `_weight_loader` instead.
    assert dict(old_param.__dict__) == source_attrs

    replace_parameter(layer, "weight", old_param)

    # Everything carries over by value, except that the loader is re-read from
    # the old parameter and lands under its public name. Comparing values (not
    # just keys) is what rules out the loader being re-bound to the new
    # parameter, which would shift `loaded_weight` into the `param` slot.
    expected = dict(source_attrs)
    expected.pop("_weight_loader", None)
    expected["weight_loader"] = loader

    assert type(layer.weight) is torch.nn.Parameter
    assert dict(layer.weight.__dict__) == expected

    if param_kind != "plain":
        for public_name in ("output_dim", "input_dim", "packed_dim", "packed_factor"):
            assert getattr(layer.weight, public_name, None) is None


def test_set_derived_buffer_requires_registration() -> None:
    module = torch.nn.Module()
    with pytest.raises(KeyError):
        set_derived_buffer(module, "derived", torch.zeros(2))


def test_set_derived_buffer_fills_placeholder_then_copies_in_place() -> None:
    """A derived tensor lands in named_buffers() and keeps its address."""
    module = torch.nn.Module()
    module.register_buffer("derived", None, persistent=False)

    set_derived_buffer(module, "derived", torch.ones(2))
    first = module.derived
    assert dict(module.named_buffers())["derived"] is first
    assert "derived" not in module.state_dict()

    set_derived_buffer(module, "derived", torch.full((2,), 3.0))
    assert module.derived is first
    assert torch.equal(first, torch.full((2,), 3.0))

    with pytest.raises(ValueError):
        set_derived_buffer(module, "derived", torch.ones(3))
    set_derived_buffer(module, "derived", None)
    assert module.derived is None


def test_set_derived_buffer_survives_layerwise_reload() -> None:
    """None-placeholder buffers must survive restore_layer_on_meta so
    that set_derived_buffer can fill them after a reload."""
    from vllm.model_executor.model_loader.reload.meta import (
        capture_layer_to_meta,
        restore_layer_on_meta,
    )
    from vllm.model_executor.model_loader.reload.types import (
        LayerReloadingInfo,
    )

    module = torch.nn.Module()
    module.register_buffer("derived", None, persistent=False)
    module.weight = torch.nn.Parameter(torch.ones(4))

    # Capture metadata while the derived buffer is still None
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(module),
        restore_device=torch.device("cpu"),
    )

    # Simulate cold start filling the buffer
    set_derived_buffer(module, "derived", torch.full((3,), 7.0))
    assert module.derived is not None

    # Simulate reload: restore_layer_on_meta wipes and re-registers
    restore_layer_on_meta(module, info)

    # The None placeholder must be back as a registered buffer
    assert "derived" in module._buffers
    assert module._buffers["derived"] is None

    # set_derived_buffer must succeed on the restored placeholder
    set_derived_buffer(module, "derived", torch.full((3,), 9.0))
    assert torch.equal(module.derived, torch.full((3,), 9.0))
    assert "derived" in dict(module.named_buffers())


def test_derived_state_stays_registered_across_layerwise_reload() -> None:
    """After a layerwise reload, every registered derived tensor is the one the
    processed layer uses: set_derived_buffer refills its storage in place, and
    buffers rebound or held by non-module objects follow the new tensors."""
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )
    from vllm.model_executor.utils import (
        register_constant_buffer,
        register_held_tensors,
    )

    class Holder:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

        def persistent_tensors(self) -> dict[str, torch.Tensor]:
            return {"tensor": self.tensor}

    class Method(QuantizeMethodBase):
        def create_weights(self, *args, **kwargs) -> None:
            pass

        def apply(self, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            set_derived_buffer(layer, "derived", layer.weight * 2)
            layer.register_buffer("rebound", layer.weight * 3, persistent=False)
            self.holder = Holder(layer.weight * 4)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    layer.register_buffer("derived", None, persistent=False)
    register_constant_buffer(layer, "constant", torch.ones(3))
    layer.quant_method = Method()
    model = torch.nn.Sequential(layer)
    record_metadata_for_reloading(model)

    layer.weight.data.fill_(1.0)
    layer.quant_method.process_weights_after_loading(layer)
    register_held_tensors(model)
    derived = layer.derived

    initialize_layerwise_reload(model)
    weight = layer.weight
    weight.weight_loader(weight, torch.full((4,), 5.0))
    # The constant does not hold back processing until finalize.
    assert torch.equal(layer.derived, torch.full((4,), 10.0))
    finalize_layerwise_reload(model, None)

    assert layer.derived is derived
    assert torch.equal(layer.derived, torch.full((4,), 10.0))
    assert torch.equal(layer.rebound, torch.full((4,), 15.0))
    held = layer.quant_method.holder.tensor
    assert torch.equal(held, torch.full((4,), 20.0))
    buffers = dict(model.named_buffers())
    assert buffers["0.rebound"] is layer.rebound
    assert any(b is held for b in buffers.values())


def test_register_held_tensors_views_wrappers_and_rerun() -> None:
    """Held tensors are deduplicated per view (not per storage), wrapper
    subclasses are resolved to their inner tensors, and a re-run follows the
    tensors the holders keep now."""
    from torch.testing._internal.two_tensor import TwoTensor

    from vllm.model_executor.utils import register_held_tensors

    class Holder:
        def __init__(self, **tensors: torch.Tensor) -> None:
            self.tensors = tensors

        def persistent_tensors(self) -> dict[str, torch.Tensor]:
            return self.tensors

    storage = torch.arange(8.0)
    module = torch.nn.Module()
    module.register_parameter(
        "wrapped",
        torch.nn.Parameter(
            TwoTensor(torch.ones(2), torch.zeros(2)), requires_grad=False
        ),
    )
    module.holder = Holder(lo=storage[:4], hi=storage[4:])
    register_held_tensors(module)
    assert set(module._buffers) == {"_held_holder_lo", "_held_holder_hi"}

    module.holder.tensors["lo"] = torch.full((4,), 2.0)
    register_held_tensors(module)
    assert module._held_holder_lo is module.holder.tensors["lo"]
    assert set(module._buffers) == {"_held_holder_lo", "_held_holder_hi"}

    # Same offset/shape/stride but another dtype covers other bytes.
    raw = torch.arange(16, dtype=torch.uint8)
    module.holder.tensors = {"bytes": raw[:4], "words": raw.view(torch.int32)}
    register_held_tensors(module)
    assert set(module._buffers) == {"_held_holder_bytes", "_held_holder_words"}

    # A holder still on meta (model built on meta) keeps the imported buffer.
    imported = module._held_holder_words
    module.holder.tensors = {"words": torch.empty(4, dtype=torch.int32, device="meta")}
    register_held_tensors(module)
    assert module._held_holder_words is imported
