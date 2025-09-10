# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
