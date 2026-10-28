# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PackedvLLMParameter,
    permute_param_layout_,
)
from vllm.model_executor.utils import (
    get_live_param,
    kernel_parameters,
    move_parameter,
    replace_parameter,
    retire_parameter,
    set_kernel_parameter,
)


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


@pytest.mark.parametrize("wrap_in_parameter", [False, True])
def test_replace_parameter_preserves_custom_attribute(wrap_in_parameter: bool) -> None:
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
    original = layer.weight
    original_ptr = original.data_ptr()

    new_data: torch.Tensor = torch.ones(4, 4).t()
    if wrap_in_parameter:
        new_data = torch.nn.Parameter(new_data, requires_grad=False)
    new_data.is_shuffled = True

    # Sanity check on the assumption above: tensor internals are not
    # reachable through `__dict__`, only user-set custom attributes are.
    assert new_data.__dict__.keys() == {"is_shuffled"}

    replace_parameter(layer, "weight", new_data)

    assert isinstance(layer.weight, torch.nn.Parameter)
    assert layer.weight.is_shuffled is True
    assert torch.equal(layer.weight.data, new_data)
    assert layer.weight.device == new_data.device

    assert layer.weight is original
    assert layer.weight.stride() == new_data.stride()
    # same byte size: copied into the loaded storage, address kept
    assert layer.weight.data_ptr() == original_ptr


@pytest.mark.parametrize(
    "new_data,copies",
    [
        (torch.ones(4, 4, dtype=torch.float32, device="cpu"), True),
        (torch.ones(4, 4, dtype=torch.float32, device="cpu").t(), True),
        (torch.ones(8, 4, dtype=torch.float32, device="cpu"), False),
        # A result that fits (equal or smaller) goes into the loaded storage,
        # which never shrinks.
        (torch.ones(4, 4, dtype=torch.bfloat16, device="cpu"), True),
        (torch.ones(4, 2, dtype=torch.float32, device="cpu"), True),
        (torch.tensor(1.0, device="cpu"), True),
    ],
    ids=[
        "same-layout-copies",
        "new-strides-copies",
        "larger-shape",
        "smaller-dtype-copies",
        "smaller-shape-copies",
        "scalar-copies",
    ],
)
def test_replace_parameter_keeps_object_by_default(
    single_rank_tp: None, new_data: torch.Tensor, copies: bool
) -> None:
    def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        pass

    layer = torch.nn.Module()
    old_param = ModelWeightParameter(
        data=torch.zeros(4, 4, dtype=torch.float32, device="cpu"),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight", old_param)
    old_data_ptr = old_param.data_ptr()

    replace_parameter(layer, "weight", new_data)

    assert layer.weight is old_param
    assert layer.weight.weight_loader is weight_loader
    assert (layer.weight.input_dim, layer.weight.output_dim) == (1, 0)
    assert layer.weight.dtype == new_data.dtype
    assert layer.weight.stride() == new_data.stride()
    assert torch.equal(layer.weight.data, new_data)
    # Copying in place keeps addresses captured in CUDA graphs valid.
    assert (layer.weight.data_ptr() == old_data_ptr) == copies
    assert layer.weight.untyped_storage().nbytes() >= 4 * 4 * 4


def test_replace_parameter_online_quantization_frees_source() -> None:
    layer = torch.nn.Module()
    layer.quant_method = type("Online", (), {"uses_meta_device": True})()
    layer.register_parameter("weight", torch.nn.Parameter(torch.zeros(4, 4)))

    replace_parameter(layer, "weight", torch.ones(4, 4, dtype=torch.bfloat16))

    assert layer.weight.untyped_storage().nbytes() == 4 * 4 * 2


def test_humming_replace_keeps_loaded_params(single_rank_tp: None) -> None:
    """A new tensor in a leftover's storage, or taking a large leftover's place,
    is written to that leftover param, registered under both names."""
    from vllm.model_executor.layers.quantization.utils.humming.linear import (
        replace_layer_parameters,
    )

    layer = torch.nn.Module()
    packed = torch.nn.Parameter(torch.ones(64, 8), requires_grad=False)
    scale_inv = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    layer.register_parameter("weight_packed", packed)
    layer.register_parameter("weight_scale_inv", scale_inv)
    tensors = {
        "weight": torch.full((32, 8), 2.0),
        "weight_scale": scale_inv.data.view(2, 2),
    }

    replace_layer_parameters(layer, tensors, list(layer._parameters))

    assert layer.weight is packed and layer.weight_scale is scale_inv
    assert torch.equal(layer.weight, torch.full((32, 8), 2.0))
    assert packed.untyped_storage().nbytes() == 64 * 8 * packed.element_size()
    assert set(kernel_parameters(layer)) == {"weight", "weight_scale"}


def test_replace_parameter_preserves_weight_loader() -> None:
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

    replace_parameter(layer, "weight", torch.ones(4, 4))

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


def test_replace_parameter_registers_new_param_for_requantization_source() -> None:
    layer = torch.nn.Module()
    old_param = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    old_param.requantization_source = True
    layer.register_parameter("weight", old_param)

    replace_parameter(layer, "weight", torch.ones(4, 4))

    assert layer.weight is not old_param
    assert torch.equal(old_param, torch.zeros(4, 4))
    assert torch.equal(layer.weight, torch.ones(4, 4))


def test_move_parameter_aliases_one_param(single_rank_tp: None) -> None:
    """Both names refer to the loaded param; a later replace updates both."""

    def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        pass

    layer = torch.nn.Module()
    src = ModelWeightParameter(
        data=torch.ones(4, 4), input_dim=1, output_dim=0, weight_loader=weight_loader
    )
    layer.register_parameter("weight_packed", src)

    move_parameter(layer, "weight_packed", "weight")
    replace_parameter(layer, "weight", torch.full((2, 8), 2.0))

    assert layer.weight_packed is src and layer.weight is src
    assert torch.equal(src.data, torch.full((2, 8), 2.0))
    assert src.weight_loader is weight_loader
    assert src._input_dim == 1 and src._output_dim == 0
    assert list(kernel_parameters(layer)) == ["weight"]

    # Kernels writing through the checkpoint name make it the runtime name again.
    set_kernel_parameter(layer, "weight_packed", torch.ones(2, 8))
    assert list(kernel_parameters(layer)) == ["weight_packed"]


def test_move_parameter_rejects_another_param(single_rank_tp: None) -> None:
    layer = torch.nn.Module()
    layer.register_parameter("a", torch.nn.Parameter(torch.ones(2)))
    layer.register_parameter("b", torch.nn.Parameter(torch.ones(2)))
    with pytest.raises(ValueError):
        move_parameter(layer, "a", "b")


def test_retired_param_keeps_data_but_is_not_live(single_rank_tp: None) -> None:
    layer = torch.nn.Module()
    layer.register_parameter("scale", torch.nn.Parameter(torch.ones(2)))
    layer.register_parameter("weight", torch.nn.Parameter(torch.ones(4)))

    retire_parameter(layer, "scale")

    assert torch.equal(layer.scale, torch.ones(2))
    assert get_live_param(layer, "scale") is None
    assert list(kernel_parameters(layer)) == ["weight"]
    set_kernel_parameter(layer, "scale", torch.zeros(2))
    assert get_live_param(layer, "scale") is layer.scale


def test_replace_parameter_with_own_parameter_keeps_layout(single_rank_tp) -> None:
    """Handing back the registered parameter keeps it unchanged."""

    def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        pass

    layer = torch.nn.Module()
    old_param = PackedvLLMParameter(
        data=torch.zeros(4, 4),
        input_dim=1,
        output_dim=0,
        packed_dim=0,
        packed_factor=8,
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight", old_param)
    old_param.data = torch.ones(4, 4)

    replace_parameter(layer, "weight", old_param)

    assert layer.weight is old_param
    assert type(layer.weight) is PackedvLLMParameter
    assert layer.weight.weight_loader is weight_loader
    assert (layer.weight.input_dim, layer.weight.output_dim) == (1, 0)
    assert (layer.weight.packed_dim, layer.weight.packed_factor) == (0, 8)
    assert torch.equal(layer.weight, torch.ones(4, 4))


def test_kept_param_ops_skip_python_torch_function(single_rank_tp: None) -> None:
    """Kept vLLM params must dispatch eager ops like a plain Parameter: a Python
    `__torch_function__` would put every op on them through the interpreter."""
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        ModelWeightParameter(
            data=torch.zeros(4, 4),
            input_dim=1,
            output_dim=0,
            weight_loader=lambda param, loaded_weight: None,
        ),
    )
    replace_parameter(layer, "weight", torch.ones(4, 4))

    assert type(layer.weight) is ModelWeightParameter
    # A kept param must dispatch like a plain Parameter. Compare against a
    # plain Parameter rather than asserting an absolute False: on ROCm,
    # has_torch_function can return True for plain Parameters too.
    plain = torch.nn.Parameter(torch.ones(4, 4))
    assert torch.overrides.has_torch_function(
        (layer.weight,)
    ) == torch.overrides.has_torch_function((plain,))
    out = layer.weight + 1
    assert type(out) is torch.Tensor
    assert torch.equal(out, torch.full((4, 4), 2.0))


def test_transposed_kept_param_keeps_checkpoint_dims(single_rank_tp: None) -> None:
    """The loader shards a kept param by its dims on reload, so a kernel transform
    that transposes it must not leave the post-transform dims on it."""
    layer = torch.nn.Module()
    weight = ModelWeightParameter(
        data=torch.arange(6.0).view(2, 3),
        input_dim=1,
        output_dim=0,
        weight_loader=lambda param, loaded_weight: None,
    )
    layer.register_parameter("weight", weight)

    MPLinearKernel._transform_param(
        None,  # type: ignore[arg-type]
        layer,
        "weight",
        lambda x: permute_param_layout_(x, input_dim=0, output_dim=1),
    )

    assert layer.weight is weight
    assert torch.equal(weight, torch.arange(6.0).view(2, 3).t())
    assert (weight.input_dim, weight.output_dim) == (1, 0)


def test_in_own_storage_skips_strided_param(single_rank_tp: None) -> None:
    """A param that is a strided view of a shared buffer is rebound, not written
    into: a contiguous copy would land on the neighbour's elements."""
    buffer = torch.zeros(4, 8)
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(buffer[:, ::2], requires_grad=False)
    )

    replace_parameter(layer, "weight", torch.ones(8))

    assert torch.equal(buffer, torch.zeros(4, 8))
    assert torch.equal(layer.weight, torch.ones(8))


def test_humming_nvfp4_after_quark_revives_input_scale_2(
    single_rank_tp: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quark retires its loaded `input_scale_2`; Humming then reads that name, so
    it writes the global input scale to the retired param instead of aliasing."""
    from vllm.model_executor.kernels.linear.nvfp4 import humming
    from vllm.model_executor.layers.quantization.quark.schemes.quark_nvfp4 import (
        QuarkNVFP4,
    )

    monkeypatch.setattr(
        humming, "prepare_humming_linear_layer_config", lambda *a, **k: None
    )
    monkeypatch.setattr(humming, "get_humming_linear_compute_config", lambda: None)
    monkeypatch.setattr(humming, "create_humming_locks", lambda device: None)
    scheme = object.__new__(QuarkNVFP4)
    scheme.group_size = 16
    scheme.kernel = object.__new__(humming.HummingNvFp4LinearKernel)

    layer = torch.nn.Module()
    scheme.create_weights(layer, [16, 16], 32, torch.bfloat16, lambda p, w: None)
    for param in layer.parameters():
        param.data.copy_(torch.full(param.shape, 2.0).to(param.dtype))
    input_scale_2 = layer.input_scale_2

    scheme.process_weights_after_loading(layer)

    assert layer.input_scale_2 is input_scale_2
    assert get_live_param(layer, "input_scale_2") is input_scale_2
    assert torch.equal(input_scale_2, layer.input_global_scale.reshape(-1))
    assert {"weight_packed", "input_scale_2"} <= set(kernel_parameters(layer))


def test_eplb_expert_weights_use_runtime_names(single_rank_tp: None) -> None:
    """EPLB filters by runtime name: an aliased checkpoint name must not hide a
    freed full weight, and retired params are not expert weights."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    layer = torch.nn.Module()
    layer.local_num_experts = 2
    for name in ("w13_weight_packed", "w2_weight_packed", "w13_weight_scale_2"):
        layer.register_parameter(
            name, torch.nn.Parameter(torch.ones(2, 4), requires_grad=False)
        )
    layer.register_parameter(
        "w13_shard", torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    )
    move_parameter(layer, "w13_weight_packed", "w13_weight")
    move_parameter(layer, "w2_weight_packed", "w2_weight")
    retire_parameter(layer, "w13_weight_scale_2")
    for name in ("w13_weight", "w2_weight"):  # freed full-width weights
        getattr(layer, name).untyped_storage().resize_(0)
    layer.localized_freed_full_weight_names = ("w13_weight", "w2_weight")

    weights = RoutedExperts.get_expert_weights(layer)  # type: ignore[arg-type]

    assert [w.untyped_storage().data_ptr() for w in weights] == [
        layer.w13_shard.untyped_storage().data_ptr()
    ]
