# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.modelopt import (
    KNvfp4Static,
    Role,
)


def _layer_with_scale(nan: bool, prefix: str | None = None):
    value = float("nan") if nan else 1.0
    return SimpleNamespace(
        weight_scale=torch.full((4, 8), value, dtype=torch.float8_e4m3fn),
        weight_scale_2=torch.ones(1, dtype=torch.float32),
        prefix=prefix,
        name="column_parallel_linear",
    )


def test_nan_scale_error_names_module_prefix():
    layer = _layer_with_scale(nan=True, prefix="model.layers.62.mlp.gate_up_proj")
    with pytest.raises(RuntimeError) as exc_info:
        KNvfp4Static().process(layer, Role.WEIGHT)
    # The prefix must identify the layer; the registered layer-type name
    # ('column_parallel_linear') is identical for every linear in the model
    # and cannot.
    assert "model.layers.62.mlp.gate_up_proj" in str(exc_info.value)


def test_nan_scale_error_falls_back_to_layer_name_without_prefix():
    layer = _layer_with_scale(nan=True, prefix=None)
    with pytest.raises(RuntimeError) as exc_info:
        KNvfp4Static().process(layer, Role.WEIGHT)
    assert "column_parallel_linear" in str(exc_info.value)


def test_loaded_scale_passes_check():
    layer = _layer_with_scale(nan=False, prefix="model.layers.0.mlp.gate_up_proj")
    # process() continues past the check and consumes weight_scale_2; give it
    # the attribute it replaces the parameter with.
    KNvfp4Static().process(layer, Role.WEIGHT)
    assert hasattr(layer, "weight_global_scale")
