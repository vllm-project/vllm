# SPDX-License-Identifier: Apache-2.0
"""
Test a simple PyTorch model with three sequential parts: 
Linear, SiLU, and Linear.
Each part is implemented as a separate class inheriting from nn.Module.
"""
import os

import pytest
import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CompilationConfig, CompilationLevel, VllmConfig,
                         set_current_vllm_config)


@support_torch_compile
class FirstLinear(nn.Module):
    """First linear layer of the model."""

    def __init__(self,
                 input_size=10,
                 output_size=20,
                 *,
                 vllm_config=None,
                 prefix='',
                 **kwargs):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ActivationLayer(nn.Module):
    """Middle activation layer using SiLU."""

    def __init__(self):
        super().__init__()
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


@support_torch_compile
class SecondLinear(nn.Module):
    """Second linear layer of the model."""

    def __init__(self,
                 input_size=20,
                 output_size=5,
                 *,
                 vllm_config=None,
                 prefix='',
                 **kwargs):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SimpleModel(nn.Module):
    """A simple model with three sequential parts: Linear, SiLU, and Linear."""

    def __init__(self,
                 input_size=10,
                 hidden_size=20,
                 output_size=5,
                 *,
                 vllm_config=None,
                 prefix=''):
        super().__init__()
        self.first_linear = FirstLinear(input_size=input_size,
                                        output_size=hidden_size,
                                        vllm_config=vllm_config,
                                        prefix=f"{prefix}first_linear.")
        self.activation = ActivationLayer()
        self.second_linear = SecondLinear(input_size=hidden_size,
                                          output_size=output_size,
                                          vllm_config=vllm_config,
                                          prefix=f"{prefix}second_linear.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_linear(x)
        x = self.activation(x)
        x = self.second_linear(x)
        return x


@torch.inference_mode
def run_model_test():
    """Run a test with the model using random inputs."""
    # Set up a basic VllmConfig with PIECEWISE compilation level
    compilation_config = CompilationConfig(level=CompilationLevel.PIECEWISE)
    vllm_config = VllmConfig(compilation_config=compilation_config)

    # Create and initialize the model
    with set_current_vllm_config(vllm_config):
        model = SimpleModel(vllm_config=vllm_config, prefix="model.").cuda()

    # Create random input
    batch_size = 2
    input_size = 10
    x = torch.randn(batch_size, input_size, device="cuda")

    # Run the model
    output = model(x)

    # Basic check that output has the expected shape
    assert output.shape == (batch_size, 5)

    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_v0():
    """Test the model with VLLM_USE_V1=0."""
    os.environ["VLLM_USE_V1"] = "0"
    output = run_model_test()
    assert output is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_v1():
    """Test the model with VLLM_USE_V1=1."""
    os.environ["VLLM_USE_V1"] = "1"
    output = run_model_test()
    assert output is not None


if __name__ == "__main__":
    # For manual testing
    test_model_v0()
    test_model_v1()
