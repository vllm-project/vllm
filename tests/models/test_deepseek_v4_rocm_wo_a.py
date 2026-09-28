# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.amd.rocm import _wo_a_block_scale_to_e8m0


def test_wo_a_block_scale_to_e8m0_from_float():
    scale = torch.tensor([[0.5, 1.0, 2.0, 4.0]], dtype=torch.float32)

    encoded = _wo_a_block_scale_to_e8m0(scale)

    assert encoded is not None
    torch.testing.assert_close(
        encoded,
        torch.tensor([[126, 127, 128, 129]], dtype=torch.uint8),
    )
    assert encoded.is_contiguous()


def test_wo_a_block_scale_to_e8m0_preserves_encoded_scales():
    raw = torch.tensor([[125, 127, 131]], dtype=torch.uint8)
    encoded = raw.view(torch.float8_e8m0fnu)

    converted = _wo_a_block_scale_to_e8m0(encoded)

    assert converted is not None
    torch.testing.assert_close(converted, raw)


@pytest.mark.parametrize(
    "scale",
    [
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[-1.0, 1.0]]),
        torch.tensor([[0.75, 1.0]]),
        torch.tensor([[float("inf"), 1.0]]),
        torch.ones(1, dtype=torch.int32),
    ],
)
def test_wo_a_block_scale_to_e8m0_rejects_invalid_scales(scale: torch.Tensor):
    assert _wo_a_block_scale_to_e8m0(scale) is None


def test_gateup_preshuffle_uses_fresh_sources_before_stable_restore(monkeypatch):
    from types import SimpleNamespace

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_processing,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )
    from vllm.model_executor.utils import register_derived_buffer
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP

    mlp = object.__new__(DeepseekV4MLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = torch.nn.Linear(128, 16, bias=False)
    mlp.gate_up_proj.register_parameter(
        "weight_scale", torch.nn.Parameter(torch.ones(1, 1), requires_grad=False)
    )
    mlp.gate_up_proj._vllm_defer_weights_reload = True
    mlp.gate_up_proj.quant_method = SimpleNamespace(
        process_weights_after_loading=lambda layer: None
    )
    mlp._gateup = True
    register_derived_buffer(mlp, "_gateup_scale")
    monkeypatch.setattr(
        fp8_utils, "get_fp8_block_weight_scale", lambda layer: layer.weight_scale
    )
    monkeypatch.setattr(
        rocm_aiter_ops, "shuffle_weight", lambda weight, **kwargs: weight + 10
    )
    record_metadata_for_reloading(mlp)
    record_metadata_for_reloading(mlp.gate_up_proj)
    mlp.prepare_gateup_preshuffle()
    weight_ptr = mlp.gate_up_proj.weight.data_ptr()
    scale_ptr = mlp._gateup_scale.data_ptr()
    for value in (1.0, 3.0, 1.0):
        initialize_layerwise_reload(mlp)
        for name, shape in (("weight", (16, 128)), ("weight_scale", (1, 1))):
            param = getattr(mlp.gate_up_proj, name)
            param.weight_loader(param, torch.full(shape, value))
        finalize_layerwise_processing(
            mlp, model_config=SimpleNamespace(dtype=torch.float32)
        )
        assert mlp.gate_up_proj.weight.data_ptr() == weight_ptr
        assert mlp._gateup_scale.data_ptr() == scale_ptr
        torch.testing.assert_close(
            mlp.gate_up_proj.weight, torch.full((16, 128), value + 10)
        )
        torch.testing.assert_close(mlp._gateup_scale, torch.full((1, 1), value))
