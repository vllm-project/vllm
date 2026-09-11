# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os
import sys
import types

import pytest
import torch

from vllm.compilation.fx_graph_dump import wrap_backend_with_fx_dump
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


def test_fx_graph_inferrt_backend_tracks_dynamo_recompilation(tmp_path, monkeypatch):
    class ShapeBranch(torch.nn.Module):
        def forward(self, x):
            if x.shape[0] == 2:
                return x + 1
            return x * 2

    calls = []

    def inferrt_backend(gm, example_inputs):
        calls.append(gm)
        return gm.forward

    # Simulate the optional external InferRT package without importing its
    # NPU runtime in the unit-test process.
    ms_inferrt = types.ModuleType("ms_inferrt")
    ms_inferrt_torch = types.ModuleType("ms_inferrt.torch")
    ms_inferrt_fx = types.ModuleType("ms_inferrt.torch.fx_backend")
    ms_inferrt_fx.backend = inferrt_backend
    monkeypatch.setitem(sys.modules, "ms_inferrt", ms_inferrt)
    monkeypatch.setitem(sys.modules, "ms_inferrt.torch", ms_inferrt_torch)
    monkeypatch.setitem(sys.modules, "ms_inferrt.torch.fx_backend", ms_inferrt_fx)

    torch._dynamo.reset()
    backend = wrap_backend_with_fx_dump("inductor", tmp_path, "test/model")
    compiled = torch.compile(
        ShapeBranch(), backend=backend, fullgraph=True, dynamic=False
    )

    assert torch.equal(compiled(torch.ones(2)), torch.full((2,), 2.0))
    assert torch.equal(compiled(torch.ones(2)), torch.full((2,), 2.0))
    assert len(list(tmp_path.glob("fx_graph_test_model_pid*.txt"))) == 1
    assert len(calls) == 1

    assert torch.equal(compiled(torch.ones(3)), torch.full((3,), 2.0))
    dumped = list(tmp_path.glob("fx_graph_test_model_pid*.txt"))
    assert len(dumped) == 2
    assert len(calls) == 2
    assert all(
        "==== raw graph ====" in path.read_text(encoding="utf-8") for path in dumped
    )
    torch._dynamo.reset()


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

        # Second call should trigger another compilation
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
