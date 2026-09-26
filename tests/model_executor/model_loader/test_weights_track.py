# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the strict weight-loading check in DefaultModelLoader.

`track_weights_loading` refuses a load that left a parameter uninitialised. It
is off by default for a quantized model and is enabled with
`--model-loader-extra-config '{"enable_weights_track": true}'`.

Its exemption used to be per-module: any module whose `quant_method` merely
defined `process_weights_after_loading` had every one of its parameters marked
as loaded. Nearly every quantization method defines it, so the check covered
nothing on the models the flag is enabled for. These tests pin both halves of
the narrowed behaviour: what must now be reported, and what must keep passing.
"""

import pytest
import torch
from torch import nn

from vllm.config.load import LoadConfig
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.utils import AutoWeightsLoader


class _SerializedQuantMethod:
    """A quant method for a checkpoint that carries its own scales.

    Fp8LinearMethod, Fp8MoEMethod, awq, gptq and compressed-tensors all define
    `process_weights_after_loading`; it is the attribute the exemption keys on.
    """

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        pass


class _OnlineQuantMethod(_SerializedQuantMethod):
    """Online quantization: vLLM computes the scales, the checkpoint has none."""

    uses_meta_device = True


def _param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.zeros(*shape), requires_grad=False)


def _module(params: dict[str, nn.Parameter], quant_method=None) -> nn.Module:
    module = nn.Module()
    for name, value in params.items():
        module.register_parameter(name, value)
    if quant_method is not None:
        module.quant_method = quant_method
    return module


def _model(children: dict[str, nn.Module]) -> nn.Module:
    model = nn.Module()
    for name, child in children.items():
        model.add_module(name, child)
    return model


def _track(model: nn.Module, loaded: set[str]) -> str | None:
    """Return the raised message, or None when the load was accepted."""
    loader = DefaultModelLoader(LoadConfig())
    try:
        loader.track_weights_loading(model, set(loaded))
    except ValueError as exc:
        return str(exc)
    return None


def test_a_missing_block_scale_is_reported():
    """The case this check exists for, and used to miss.

    A serialized-fp8 checkpoint whose block scale is absent loads silently and
    the layer then computes with `weight_scale_inv` at its uninitialised
    value: the server boots, serves and reports healthy while the outputs are
    wrong.
    """
    model = _model(
        {
            "wo_a": _module(
                {"weight": _param(4, 4), "weight_scale_inv": _param(1, 1)},
                _SerializedQuantMethod(),
            )
        }
    )
    message = _track(model, {"wo_a.weight"})
    assert message is not None
    assert "wo_a.weight_scale_inv" in message


def test_a_missing_quantized_weight_is_reported():
    """Not only scales: the weight itself was exempt too."""
    model = _model(
        {"proj": _module({"weight": _param(4, 4)}, _SerializedQuantMethod())}
    )
    message = _track(model, set())
    assert message is not None
    assert "proj.weight" in message


@pytest.mark.parametrize("scale_name", ["k_scale", "v_scale", "q_scale", "prob_scale"])
def test_kv_cache_scales_stay_optional(scale_name: str):
    """The exemption's stated purpose: most checkpoints omit these."""
    model = _model({"attn": _module({scale_name: _param(1)}, _SerializedQuantMethod())})
    assert _track(model, set()) is None


def test_online_quantization_still_exempts_the_whole_module():
    """`uses_meta_device` means nothing here is expected from the checkpoint."""
    model = _model(
        {
            "proj": _module(
                {"weight": _param(4, 4), "weight_scale": _param(1)},
                _OnlineQuantMethod(),
            )
        }
    )
    assert _track(model, set()) is None


def test_a_fully_loaded_quantized_layer_passes():
    model = _model(
        {
            "proj": _module(
                {"weight": _param(4, 4), "weight_scale_inv": _param(1, 1)},
                _SerializedQuantMethod(),
            )
        }
    )
    assert _track(model, {"proj.weight", "proj.weight_scale_inv"}) is None


def test_unquantized_layers_are_unaffected():
    """The default path (`quantization is None`) must behave exactly as before."""
    model = _model({"lin": _module({"weight": _param(4, 4)})})
    assert _track(model, {"lin.weight"}) is None
    assert "lin.weight" in (_track(model, set()) or "")


def test_loaded_weights_none_is_a_no_op():
    """A model whose `load_weights` returns nothing cannot be checked."""
    model = _model({"lin": _module({"weight": _param(4, 4)})})
    loader = DefaultModelLoader(LoadConfig())
    assert loader.track_weights_loading(model, None) is None


def test_a_mixed_model_reports_only_the_genuine_gap():
    """One real omission beside every legitimate exemption."""
    model = _model(
        {
            "attn": _module(
                {"k_scale": _param(1), "v_scale": _param(1)}, _SerializedQuantMethod()
            ),
            "online": _module({"weight": _param(4, 4)}, _OnlineQuantMethod()),
            "plain": _module({"weight": _param(4, 4)}),
            "broken": _module(
                {"weight": _param(4, 4), "weight_scale_inv": _param(1, 1)},
                _SerializedQuantMethod(),
            ),
        }
    )
    message = _track(model, {"online.weight", "plain.weight", "broken.weight"})
    assert message is not None
    assert "broken.weight_scale_inv" in message
    for unexpected in ("k_scale", "v_scale", "online.weight", "plain.weight"):
        assert unexpected not in message


def test_nested_parameter_names_resolve_against_the_module_prefix():
    """`named_parameters` is recursive, so the suffix test must read the leaf."""
    inner = _module({"k_scale": _param(1)})
    outer = nn.Module()
    outer.add_module("impl", inner)
    outer.quant_method = _SerializedQuantMethod()
    model = _model({"attn": outer})
    assert _track(model, set()) is None


def test_moe_runner_reports_expert_names_as_registered():
    """MoE layers load through `MoERunner.load_weights`, which forwards to its
    `routed_experts` child. `AutoWeightsLoader` qualifies what the runner
    returns with the runner's own prefix, so the names must include
    `routed_experts.`, or every expert weight is reported as not loaded."""

    class RoutedExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w13_weight = _param(2, 8, 4)
            self.w2_weight = _param(2, 4, 4)
            self.quant_method = _SerializedQuantMethod()

        def load_weights(self, weights):
            for name, _ in weights:
                yield "w13_weight" if "gate_up_proj" in name else "w2_weight"

    class Runner(nn.Module):
        load_weights = MoERunner.load_weights

        def __init__(self) -> None:
            super().__init__()
            self.routed_experts = RoutedExperts()

    model = _model({"experts": Runner()})
    loaded = AutoWeightsLoader(model).load_weights(
        [
            ("experts.gate_up_proj", torch.zeros(2, 8, 4)),
            ("experts.down_proj", torch.zeros(2, 4, 4)),
        ]
    )
    assert loaded == {name for name, _ in model.named_parameters()}
    assert _track(model, loaded) is None
