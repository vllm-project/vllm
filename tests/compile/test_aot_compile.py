# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import multiprocessing
import tempfile
from contextlib import contextmanager

import pytest
import torch

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.utils.torch_utils import is_torch_equal_or_newer


def reference_fn(x: torch.Tensor):
    assert x.shape[0] <= 42
    assert x.shape[0] % 2 == 0
    for _ in range(3000):
        x = x + x.shape[0]
    return x


@support_torch_compile
class CompiledMod(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return reference_fn(x)


def make_vllm_config() -> VllmConfig:
    return VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
        )
    )


@contextmanager
def use_vllm_config(vllm_config: VllmConfig):
    with set_forward_context({}, vllm_config), set_current_vllm_config(vllm_config):
        yield


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
def test_no_dynamo_cache_entry(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        vllm_config = make_vllm_config()
        args = (torch.randn(10, 10),)
        expected = reference_fn(*args)
        with use_vllm_config(vllm_config):
            m.setenv("VLLM_USE_AOT_COMPILE", "0")
            with (
                pytest.raises(RuntimeError, match="Detected recompile"),
                torch.compiler.set_stance("fail_on_recompile"),
            ):
                CompiledMod(vllm_config=vllm_config)(*args)

            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            torch._dynamo.reset()
            with torch.compiler.set_stance("fail_on_recompile"):
                actual = CompiledMod(vllm_config=vllm_config)(*args)
            assert torch.allclose(actual, expected)


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
def test_force_aot_load(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as tmpdirname, monkeypatch.context() as m:
        args = (torch.randn(10, 10),)
        m.setenv("VLLM_USE_AOT_COMPILE", "1")
        m.setenv("VLLM_FORCE_AOT_LOAD", "1")
        m.setenv("VLLM_CACHE_ROOT", tmpdirname)
        vllm_config = make_vllm_config()
        with use_vllm_config(vllm_config), pytest.raises(FileNotFoundError):
            CompiledMod(vllm_config=vllm_config)(*args)


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
def test_save_and_load(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        args = (torch.randn(10, 10),)

        with tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                expected = CompiledMod(vllm_config=vllm_config)(*args)

            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                ret = CompiledMod(vllm_config=vllm_config)(*args)
            assert torch.allclose(ret, expected)


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
def test_shape_env(monkeypatch: pytest.MonkeyPatch):
    """
    Test that the shape environment is correctly serialized and preserved
    when loading from cache.
    """
    with monkeypatch.context() as m:
        args = (torch.randn(10, 10),)

        with tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                compiled_mod(*args)
                artifacts = compiled_mod.aot_compiled_fn._artifacts
                guards_string = artifacts.compiled_fn.shape_env.format_guards()
                assert guards_string == " - s77 <= 42\n - Eq(Mod(s77, 2), 0)"

            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                compiled_mod(*args)
                artifacts = compiled_mod.aot_compiled_fn._artifacts
                guards_string = artifacts.compiled_fn.shape_env.format_guards()
                assert guards_string == " - s77 <= 42\n - Eq(Mod(s77, 2), 0)"


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
@use_vllm_config(make_vllm_config())
def test_gpt2_cache_hit(monkeypatch: pytest.MonkeyPatch):
    """
    Test that compiling gpt2 twice results in a cache hit and
    capture torch dynamic symbol creations to ensure make_symbol
    not called on cache hit.
    """

    import torch.fx.experimental.symbolic_shapes as symbolic_shapes_module
    from torch.utils._sympy.symbol import make_symbol

    from vllm import LLM

    create_symbol_counter = multiprocessing.Value("i", 0)
    original_make_symbol = make_symbol

    @functools.wraps(original_make_symbol)
    def counting_make_symbol(prefix, idx, **kwargs):
        with create_symbol_counter.get_lock():
            create_symbol_counter.value += 1
        return original_make_symbol(prefix, idx, **kwargs)

    symbolic_shapes_module.make_symbol = counting_make_symbol
    try:
        with monkeypatch.context() as m, tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            # First compilation - initialize model and generate
            llm_model = LLM(
                model="gpt2",
                compilation_config=CompilationConfig(
                    mode=CompilationMode.VLLM_COMPILE,
                ),
                max_model_len=256,
            )

            llm_model.generate("Hello, my name is")
            assert create_symbol_counter.value == 2
            create_symbol_counter.value = 0

            # Clean up first model
            del llm_model

            # Second compilation - should hit cache
            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            llm_model = LLM(
                model="gpt2",
                compilation_config=CompilationConfig(
                    mode=CompilationMode.VLLM_COMPILE,
                ),
                max_model_len=256,
            )
            llm_model.generate("Hello, my name is")

            assert create_symbol_counter.value == 0

    finally:
        # Restore original method
        symbolic_shapes_module.make_symbol = original_make_symbol
