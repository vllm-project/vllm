# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from contextlib import contextmanager

import pytest
import torch

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CompilationConfig, CompilationLevel, VllmConfig,
                         set_current_vllm_config)
from vllm.forward_context import set_forward_context


class MyMod(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        for _ in range(3000):
            x = x + x.shape[0]
        return x


def make_vllm_config() -> VllmConfig:
    return VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE, ))


@contextmanager
def use_vllm_config(vllm_config: VllmConfig):
    with set_forward_context(
        {}, vllm_config), set_current_vllm_config(vllm_config):
        yield


def test_no_eval_frame(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        mod = MyMod()
        args = (torch.randn(10, 10), )
        expected = mod(*args)
        CompiledMod = support_torch_compile(MyMod)

        vllm_config = make_vllm_config()
        m.setenv("VLLM_USE_AOT_COMPILE", "0")
        try:
            with use_vllm_config(vllm_config), torch.compiler.set_stance(
                    "fail_on_recompile"):
                CompiledMod(vllm_config=vllm_config)(*args)
        except RuntimeError as e:
            assert "Detected recompile" in str(e)
        else:
            raise AssertionError("Expected exception to be raised")

        m.setenv("VLLM_USE_AOT_COMPILE", "1")
        torch._dynamo.reset()
        with use_vllm_config(vllm_config), torch.compiler.set_stance(
                "fail_on_recompile"):
            ret = CompiledMod(vllm_config=vllm_config)(*args)
            assert torch.allclose(ret, expected)


def test_force_aot_load(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as tmpdirname, monkeypatch.context(
    ) as m:
        args = (torch.randn(10, 10), )
        m.setenv("VLLM_USE_AOT_COMPILE", "1")
        m.setenv("VLLM_FORCE_AOT_LOAD", "1")
        m.setenv("VLLM_CACHE_ROOT", tmpdirname)
        vllm_config = make_vllm_config()
        with use_vllm_config(vllm_config):
            CompiledMod = support_torch_compile(MyMod)
            try:
                CompiledMod(vllm_config=vllm_config)(*args)
            except Exception as e:
                assert isinstance(e, FileNotFoundError)
            else:
                raise AssertionError(
                    "Expected failed aot compilation with clean state.")


def test_basic(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        args = (torch.randn(10, 10), )
        CompiledMod = support_torch_compile(MyMod)

        with tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                expected = CompiledMod(vllm_config=vllm_config)(*args)
                m.setenv("VLLM_FORCE_AOT_LOAD", "1")
                ret = CompiledMod(vllm_config=vllm_config)(*args)
                assert torch.allclose(ret, expected)
