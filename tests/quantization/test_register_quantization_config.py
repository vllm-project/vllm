# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests register custom quantization config.

See https://github.com/vllm-project/vllm/issues/11926 for more details.

Run `pytest tests/quantization/test_register_quantization_config.py`.
"""
from typing import Any, Optional

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import LinearBase  # noqa: E501
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import (
    QuantizationMethods, get_quantization_config, register_quantization_config)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig)


class FakeQuantLinearMethod(UnquantizedLinearMethod):
    """Fake quantization linear method for per-token dynamic quantization."""

    def __init__(self, num_bits: int = 8) -> None:
        """Initialize the quantization method."""
        super().__init__()
        self.num_bits = num_bits

    def apply(self,
              layer: "torch.nn.Module",
              x: "torch.Tensor",
              bias: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """Perform fake quantization before the linear layer."""

        # Calculate the scales dynamically
        max_val = torch.amax(x, dim=(0, -1), keepdims=True)
        min_val = torch.amin(x, dim=(0, -1), keepdims=True)
        scales = (max_val - min_val) / (2**self.num_bits - 1)

        # Fake quantize the input
        quant_x = torch.clamp(torch.round(x / scales), -2**(self.num_bits - 1),
                              2**(self.num_bits - 1) - 1)
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

    def get_supported_act_dtypes(self) -> list["torch.dtype"]:
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

    def get_quant_method(self, layer: "torch.nn.Module",
                         prefix: str) -> Optional["FakeQuantLinearMethod"]:
        """Get the quantize method to use for the quantized layer."""
        if isinstance(layer, LinearBase):
            return FakeQuantLinearMethod(num_bits=self.num_bits)
        return None


def test_register_quantization_config():
    """Test register custom quantization config."""

    # The quantization method `custom_quant` should be registered.
    assert get_quantization_config("custom_quant") == CustomQuantConfig

    # The quantization method `custom_quant` is already exists,
    # should raise an error.
    with pytest.raises(ValueError):
        register_quantization_config("custom_quant")(CustomQuantConfig)


@pytest.mark.parametrize(argnames="model",
                         argvalues=[
                             "meta-llama/Llama-3.2-1B-Instruct",
                         ])
def test_custom_quant(vllm_runner, model, monkeypatch):
    """Test infer with the custom quantization method."""
    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(model_name=model,
                     quantization="custom_quant",
                     enforce_eager=True) as llm:

        def check_model(model):
            layer = model.model.layers[0]
            qkv_proj = layer.self_attn.qkv_proj

            # Check the quantization method is FakeQuantLinearMethod
            assert isinstance(qkv_proj.quant_method, FakeQuantLinearMethod)

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output
