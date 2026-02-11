# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

import pytest
import torch

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)


class MyMod(torch.nn.Module):
    def forward(self, x: torch.Tensor, cache: torch.Tensor | None = None):
        if x.size()[0] >= 4:
            return x * 2
        else:
            return x * 100


class MyWrapper(TorchCompileWithNoGuardsWrapper):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        # this is the function to be compiled
        return self.model(x)


@pytest.mark.parametrize("use_bytecode_hook", [True, False])
def test_torch_compile_wrapper(use_bytecode_hook, monkeypatch):
    """Test basic functionality of TorchCompileWithNoGuardsWrapper."""
    # Set the environment variable for this test
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "1" if use_bytecode_hook else "0")

    # Create a proper vLLM config instead of mocking
    vllm_config = VllmConfig()
    vllm_config.compilation_config = CompilationConfig()
    vllm_config.compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE
    vllm_config.compilation_config.backend = "inductor"

    # Test DYNAMO_TRACE_ONCE
    with set_current_vllm_config(vllm_config):
        torch._dynamo.reset()
        mod = MyMod()
        wrapper = MyWrapper(mod)

        # First call should trigger compilation
        x = torch.tensor([1, 2, 3, 4])
        torch._dynamo.mark_dynamic(x, 0)

        result1 = wrapper(x)
        expected1 = torch.tensor([2, 4, 6, 8])
        assert torch.allclose(result1, expected1), (
            f"Expected {expected1}, got {result1}"
        )

        # Second call should use compiled code
        x2 = torch.tensor([1, 2, 3])
        result2 = wrapper(x2)
        expected2 = torch.tensor([2, 4, 6])
        assert torch.allclose(result2, expected2), (
            f"Expected {expected2}, got {result2}"
        )

        # without the wrapper result would be different.
        result3 = mod(x2)
        expected3 = torch.tensor([100, 200, 300])

        assert torch.allclose(result3, expected3), (
            f"Expected {result3}, got {expected3}"
        )

    # with STOCK_TORCH_COMPILE we do not remove guards.
    vllm_config.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
    torch._dynamo.reset()
    with set_current_vllm_config(vllm_config):
        mod = MyMod()
        wrapper = MyWrapper(mod)

        # First call should trigger compilation
        x = torch.tensor([1, 2, 3, 4])
        torch._dynamo.mark_dynamic(x, 0)

        result1 = wrapper(x)
        expected1 = torch.tensor([2, 4, 6, 8])
        assert torch.allclose(result1, expected1), (
            f"Expected {expected1}, got {result1}"
        )

        # Second call should triger another compilation
        x2 = torch.tensor([1, 2, 3])
        result2 = wrapper(x2)
        expected2 = torch.tensor([100, 200, 300])
        assert torch.allclose(result2, expected2), (
            f"Expected {expected2}, got {result2}"
        )

    # NO_COMPILATION level not supported.
    vllm_config.compilation_config.mode = None
    torch._dynamo.reset()
    with set_current_vllm_config(vllm_config):
        torch._dynamo.reset()
        mod = MyMod()

        try:
            wrapper = MyWrapper(mod)
        except Exception:
            return
        raise AssertionError("expected an exception to be raised")


if __name__ == "__main__":
    # Run with both parameter values

    class MockMonkeypatch:
        def setenv(self, name, value):
            os.environ[name] = value

    mp = MockMonkeypatch()

    print("Testing with VLLM_USE_BYTECODE_HOOK=False")
    test_torch_compile_wrapper(False, mp)

    print("Testing with VLLM_USE_BYTECODE_HOOK=True")
    test_torch_compile_wrapper(True, mp)

    print("All tests passed!")
