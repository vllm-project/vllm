# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests register custom quantization config.

See https://github.com/vllm-project/vllm/issues/11926 for more details.

Run `pytest tests/quantization/test_register_quantization_config.py`.
"""

import logging
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from tests.quantization.utils import load_model_without_vllm_runner
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    LinearBase,  # noqa: E501
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    get_quantization_config,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,  # noqa: E501
)
from vllm.platforms import current_platform


class FakeQuantLinearMethod(UnquantizedLinearMethod):
    """Fake quantization linear method for per-token dynamic quantization."""

    def __init__(self, num_bits: int = 8) -> None:
        """Initialize the quantization method."""
        super().__init__()
        self.num_bits = num_bits

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform fake quantization before the linear layer."""

        # Calculate the scales dynamically
        max_val = torch.amax(x, dim=(0, -1), keepdims=True)
        min_val = torch.amin(x, dim=(0, -1), keepdims=True)
        scales = (max_val - min_val) / (2**self.num_bits - 1)

        # Fake quantize the input
        quant_x = torch.clamp(
            torch.round(x / scales),
            -(2 ** (self.num_bits - 1)),
            2 ** (self.num_bits - 1) - 1,
        )
        dequant_x = quant_x * scales

        return F.linear(dequant_x, layer.weight, bias)


@register_quantization_config("custom_quant")
class CustomQuantConfig(QuantizationConfig):
    """Custom quantization config for per-token dynamic fake quantization."""

    def __init__(self, num_bits: int = 8) -> None:
        """Initialize the quantization config."""
        super().__init__()
        self.num_bits = num_bits

    def get_name(self) -> QuantizationMethods:
        """Name of the quantization method."""
        return "custom_quant"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        """List of supported activation dtypes."""
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU capability to support the quantization method."""
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        """List of filenames to search for in the model directory."""
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CustomQuantConfig":
        """Create a config class from the model's quantization config."""
        return CustomQuantConfig(num_bits=config.get("num_bits", 8))

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> FakeQuantLinearMethod | None:
        """Get the quantize method to use for the quantized layer."""
        if isinstance(layer, LinearBase):
            return FakeQuantLinearMethod(num_bits=self.num_bits)
        return None


def test_register_quantization_config(caplog_vllm):
    """Test register custom quantization config."""

    # The quantization method `custom_quant` should be registered.
    assert get_quantization_config("custom_quant") == CustomQuantConfig

    # The quantization method `custom_quant` is already exists,
    # should raise a debug message when re-registering it.
    with caplog_vllm.at_level(logging.DEBUG, logger="vllm"):
        register_quantization_config("custom_quant")(CustomQuantConfig)

    assert any(
        "The quantization method 'custom_quant' already exists" in message
        for message in caplog_vllm.messages
    ), "Expected a debug message when re-registering custom_quant"


@pytest.mark.parametrize(
    argnames="model",
    argvalues=[
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
)
def test_custom_quant(model, monkeypatch, dist_init, workspace_init):
    """Test infer with the custom quantization method."""
    vllm_model, vllm_config = load_model_without_vllm_runner(
        model,
        quantization="custom_quant",
    )
    layer = vllm_model.model.layers[0]
    qkv_proj = layer.self_attn.qkv_proj

    # Check the quantization method is FakeQuantLinearMethod.
    assert isinstance(qkv_proj.quant_method, FakeQuantLinearMethod)

    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
    device_type = current_platform.device_type
    input_ids = torch.tensor([1, 2, 3, 4], device=device_type)
    positions = torch.arange(input_ids.numel(), device=device_type)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config, num_tokens=input_ids.numel()),
    ):
        hidden_states = vllm_model(input_ids, positions, None)
        logits = vllm_model.compute_logits(hidden_states)
    assert torch.isfinite(logits).all()
