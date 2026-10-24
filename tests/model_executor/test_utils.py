# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch
from safetensors.torch import safe_open, save_file

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PackedvLLMParameter,
    permute_param_layout_,
)
from vllm.model_executor.utils import copy_weight_, replace_parameter


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
    assert layer.weight.data_ptr() == new_data.data_ptr()


@pytest.mark.parametrize(
    "new_data,copies",
    [
        (torch.ones(4, 4, dtype=torch.float32, device="cpu"), True),
        (torch.ones(4, 4, dtype=torch.float32, device="cpu").t(), False),
        (torch.ones(8, 4, dtype=torch.float32, device="cpu"), False),
        (torch.ones(4, 4, dtype=torch.bfloat16, device="cpu"), False),
    ],
    ids=["same-layout-copies", "new-strides", "new-shape", "new-dtype"],
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


@pytest.fixture
def prefault_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the integrated-GPU path of `copy_weight_` without a GPU."""
    monkeypatch.setattr(
        "vllm.model_executor.utils._is_integrated_gpu", lambda device: True
    )


@pytest.mark.parametrize(
    "src",
    [
        torch.tensor(0.5),
        torch.empty(0, 8),
        torch.randn(64, 1024).to(torch.bfloat16),
        torch.randint(0, 256, (64, 1024), dtype=torch.uint8).view(torch.float8_e4m3fn),
        torch.randn(64, 1024).narrow(1, 512, 512),
    ],
    ids=["0d", "empty", "bf16", "fp8", "dim1_narrow"],
)
def test_copy_weight_prefault_copies_exact_bytes(
    prefault_on: None, src: torch.Tensor
) -> None:
    """The touch only runs on integrated GPUs, which CI doesn't have. It must
    accept the shapes and dtypes weight loaders pass and leave the copy
    unchanged.
    """
    dst = torch.empty_like(src, memory_format=torch.contiguous_format)
    copy_weight_(dst, src)
    assert torch.equal(
        dst.reshape(-1).view(torch.uint8), src.reshape(-1).view(torch.uint8)
    )


def test_copy_weight_prefault_maps_the_mmap_view(
    prefault_on: None, tmp_path: Path
) -> None:
    """Every page of the safetensors view must be mapped before the copy: a
    host-to-device copy that takes those faults itself is the slow path on
    integrated GPUs (#58726). The meta destination keeps the copy itself from
    reading the source.
    """
    path = str((tmp_path / "w.safetensors").resolve())
    save_file({"w": torch.randn(1 << 20)}, path)

    def mapped_bytes() -> int:
        total, in_file = 0, False
        with open("/proc/self/smaps") as smaps:
            for line in smaps:
                fields = line.split()
                if ":" not in fields[0]:
                    in_file = fields[-1] == path
                elif in_file and fields[0] == "Rss:":
                    total += int(fields[1]) * 1024
        return total

    with safe_open(path, framework="pt") as f:
        src = f.get_tensor("w")
        assert mapped_bytes() < src.nbytes
        copy_weight_(torch.empty_like(src, device="meta"), src)
        assert mapped_bytes() >= src.nbytes
