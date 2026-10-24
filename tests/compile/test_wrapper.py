# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

import pytest
import torch

from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.wrapper import (
    TorchCompileWithNoGuardsWrapper,
    compile_model_with_stock_torch,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
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


@pytest.mark.parametrize("explicit_region", [False, True])
def test_compile_model_with_stock_torch(monkeypatch, explicit_region):
    """The whole model is compiled in place with the configured backend."""
    graphs: list[torch.fx.GraphModule] = []

    def recording_backend(gm: torch.fx.GraphModule, example_inputs):
        graphs.append(gm)
        return gm.forward

    monkeypatch.setattr(
        CompilationConfig,
        "init_backend",
        lambda self, vllm_config, *args, **kwargs: recording_backend,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, x: torch.Tensor):
            return self.linear(x).relu()

    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vllm_config.compilation_config = CompilationConfig(
        mode=CompilationMode.STOCK_TORCH_COMPILE
    )
    model = Model()
    if explicit_region:
        model.get_compilation_modules = lambda: [model]
    x = torch.randn(3, 4)
    expected = model(x)

    torch._dynamo.reset()
    with compilation_counter.expect(stock_torch_compile_count=1):
        compile_model_with_stock_torch(model, vllm_config)
    assert not graphs  # compilation is deferred to the first call
    torch.testing.assert_close(model(x), expected)
    assert len(graphs) == 1


def test_stock_compile_decorated_model_without_regions(monkeypatch):
    """Decorated models must opt in before changing the whole-model compile path."""
    calls = []
    backend = object()
    monkeypatch.setattr(CompilationConfig, "init_backend", lambda *a, **kw: backend)
    monkeypatch.setattr(
        torch.nn.Module, "compile", lambda self, **kw: calls.append((self, kw))
    )

    @support_torch_compile
    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return x + 1

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.compilation_config = CompilationConfig(
        mode=CompilationMode.STOCK_TORCH_COMPILE
    )
    with set_current_vllm_config(config):
        model = Model()
        forward = model.forward
        compile_model_with_stock_torch(model, config)
    assert calls == [(model, {"fullgraph": True, "backend": backend})]
    assert model.forward == forward


def test_stock_compile_regions_preserve_eager_control_and_guards(monkeypatch):
    """Compile decorated regions without tracing orchestration or dropping guards."""
    executed = []

    def backend(graph, inputs):
        def run(*args):
            executed.append(graph)
            return graph.forward(*args)

        return run

    monkeypatch.setattr(CompilationConfig, "init_backend", lambda *a, **kw: backend)

    @support_torch_compile
    class Stage(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x: torch.Tensor):
            return x * self.scale if x.shape[0] >= 4 else x + self.scale

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = Stage(2)
            self.second = Stage(3)
            self.use_second = False

        def get_compilation_modules(self):
            return [self.first, self.second]

        @torch.compiler.disable
        def forward(self, x):
            x = self.first(x)
            return self.second(x) if self.use_second else x

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.compilation_config = CompilationConfig(
        mode=CompilationMode.STOCK_TORCH_COMPILE
    )
    torch._dynamo.reset()
    with (
        set_current_vllm_config(config),
        torch._dynamo.config.patch(recompile_limit=1),
    ):
        model = Model()
        with compilation_counter.expect(stock_torch_compile_count=2):
            compile_model_with_stock_torch(model, config)
        assert not executed
        for size, use_second in [(4, False), (3, True), (8, True), (1, False)]:
            model.use_second = use_second
            x = torch.ones(size)
            expected = x * 2 if size >= 4 else x + 2
            if use_second:
                expected = expected * 3 if size >= 4 else expected + 3
            calls = len(executed)
            torch.testing.assert_close(model(x), expected)
            assert len(executed) - calls == 1 + use_second
            assert torch._dynamo.config.recompile_limit == 1


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
