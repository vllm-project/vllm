# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side references (quant configs, kernels) keep startup storage across reload."""

from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.model_loader.reload.side_state import (
    alias_onto,
    collect_side_tensors,
    pin_side_tensors,
)


@dataclass(frozen=True)
class _QuantConfig:
    scale: torch.Tensor
    alpha: torch.Tensor


class _Kernel:
    def __init__(self, w: torch.Tensor, extra: tuple):
        self.w = w
        self.extra = extra


class _Method:
    def setup(self, layer: torch.nn.Module, workspace=None) -> None:
        """What process_weights_after_loading does: derive and stash."""
        self.quant_config = _QuantConfig(layer.w_scale, 1.0 / layer.w_scale)
        self.kernel = _Kernel(layer.weight.view(-1), (layer.w_scale[:2], 3))
        self.workspace = torch.zeros(4) if workspace is None else workspace


def _layer(seed: float, workspace: torch.Tensor | None = None) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.full((2, 4), seed), requires_grad=False)
    layer.w_scale = torch.nn.Parameter(torch.full((4,), seed), requires_grad=False)
    layer.quant_method = _Method()
    layer.quant_method.setup(layer, workspace)
    return layer


def _reload(layer: torch.nn.Module, value: float) -> list[str]:
    """Mimic layerwise: rerun setup on fresh temporaries, copy back, pin."""
    kernels = {"weight": layer.weight, "w_scale": layer.w_scale}
    before = collect_side_tensors(layer)
    for name, t in kernels.items():
        delattr(layer, name)
        layer.register_parameter(
            name, torch.nn.Parameter(torch.full_like(t, value), requires_grad=False)
        )
    layer.quant_method.setup(layer)
    for name, kernel in kernels.items():
        temp = getattr(layer, name)
        kernel.data.copy_(temp)
        alias_onto(temp, kernel.data)
        delattr(layer, name)
        layer.register_parameter(name, kernel)
    return pin_side_tensors(layer, before)


def _ptrs(layer: torch.nn.Module) -> dict:
    return {p: t.data_ptr() for p, t in collect_side_tensors(layer).items()}


class TestPinSideTensors:
    @pytest.mark.parametrize("value", [2.0, 4.0], ids=["once", "other-value"])
    def test_side_refs_keep_startup_storage(self, value):
        layer = _layer(1.0)
        start = _ptrs(layer)

        assert _reload(layer, value) == []

        qm = layer.quant_method
        assert _ptrs(layer) == start
        # temporaries, views of them and derived tensors carry the new values
        torch.testing.assert_close(qm.quant_config.scale, torch.full((4,), value))
        torch.testing.assert_close(qm.quant_config.alpha, torch.full((4,), 1 / value))
        torch.testing.assert_close(qm.kernel.w, torch.full((8,), value))
        torch.testing.assert_close(qm.kernel.extra[0], torch.full((2,), value))

    def test_reload_twice_is_stable(self):
        layer = _layer(1.0)
        start = _ptrs(layer)
        _reload(layer, 2.0)
        _reload(layer, 2.0)
        assert _ptrs(layer) == start

    @pytest.mark.parametrize(
        "startup",
        [torch.zeros(5), torch.zeros(1).expand(4)],
        ids=["reshaped", "expanded"],
    )
    def test_unpinnable_side_tensor_is_reported(self, startup):
        layer = _layer(1.0, startup)
        assert _reload(layer, 2.0) == [".quant_method.workspace"]
