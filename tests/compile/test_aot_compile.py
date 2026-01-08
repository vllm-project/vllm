# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import multiprocessing
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import vllm.model_executor.layers.activation
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.envs import disable_envs_cache
from vllm.forward_context import set_forward_context
from vllm.utils.torch_utils import is_torch_equal_or_newer

from ..utils import create_new_process_for_each_test


@pytest.fixture
def vllm_tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fixture that sets VLLM_CACHE_ROOT to a temporary directory."""
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm_cache"))
    return tmp_path


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
            disable_envs_cache()

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
            disable_envs_cache()

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
            disable_envs_cache()

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
def test_partition_wrapper_applied_on_aot_load(
    monkeypatch: pytest.MonkeyPatch, vllm_tmp_cache: Path, mocker
):
    """
    Test that partition wrappers are applied when loading AOT cached functions.

    This test verifies the fix for GitHub issue #31439 where AOT compile
    caused 2x latency regression when use_inductor_graph_partition=True.
    The root cause was that partition wrapper context was bypassed when
    loading from AOT cache.
    """
    from vllm.config import CUDAGraphMode

    args = (torch.randn(10, 10),)
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")

    # Create config with partition enabled
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            use_inductor_graph_partition=True,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )

    # First compilation - save to cache
    with use_vllm_config(vllm_config):
        compiled_mod = CompiledMod(vllm_config=vllm_config)
        compiled_mod(*args)
    disable_envs_cache()

    # Second run - load from cache, verify partition wrapper applied
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            use_inductor_graph_partition=True,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )

    # Use mocker to spy on set_customized_partition_wrappers
    spy = mocker.spy(torch._inductor.utils, "set_customized_partition_wrappers")

    with use_vllm_config(vllm_config):
        compiled_mod = CompiledMod(vllm_config=vllm_config)

        # First call after restart: loads from AOT cache.
        # This tests the fix for the first call after a restart.
        compiled_mod(*args)

        # Verify partition wrapper was called on AOT load.
        assert spy.call_count >= 2, (
            "Expected partition wrapper to be set and cleared on AOT load, "
            f"got {spy.call_count} calls"
        )
        # First call should set a wrapper, last call should clear it
        assert spy.call_args_list[0][0][0] is not None, (
            "First call on AOT load should set a wrapper function"
        )
        assert spy.call_args_list[-1][0][0] is None, (
            "Last call on AOT load should clear the wrapper"
        )

        # Reset for the next check.
        spy.reset_mock()

        # Subsequent call: uses the cached `aot_compiled_fn`.
        # This tests the fix for subsequent calls.
        compiled_mod(*args)

        # Verify partition wrapper was called on the subsequent call.
        assert spy.call_count >= 2, (
            "Expected partition wrapper set and cleared on subsequent "
            f"call, got {spy.call_count} calls"
        )
        assert spy.call_args_list[0][0][0] is not None, (
            "First call on subsequent call should set a wrapper function"
        )
        assert spy.call_args_list[-1][0][0] is None, (
            "Last call on subsequent call should clear the wrapper"
        )


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
@create_new_process_for_each_test("spawn")
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
            disable_envs_cache()
            vllm.model_executor.layers.activation._ACTIVATION_REGISTRY._dict.clear()

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
