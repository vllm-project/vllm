# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import tempfile

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CompilationConfig, CompilationLevel, VllmConfig,
                         set_current_vllm_config)


class MyMod(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        for _ in range(3000):
            x = x + x.shape[0]
        return x


def test_basic(monkeypatch: pytest.MonkeyPatch):
    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE, ))
    mod = MyMod()
    args = (torch.randn(10, 10), )
    expected = mod(*args)
    CompiledMod = support_torch_compile(MyMod)
    with set_current_vllm_config(vllm_config):
        ret = CompiledMod(vllm_config=vllm_config)(*args)
    assert torch.allclose(ret, expected)
    torch._dynamo.reset()

    try:
        with torch.compiler.set_stance(
                "fail_on_recompile"), set_current_vllm_config(vllm_config):
            ret = CompiledMod(vllm_config=vllm_config)(*args)
            assert torch.allclose(ret, expected)
    except RuntimeError as e:
        assert "Detected recompile" in str(e)
    else:
        raise AssertionError("Expected failed recompilation.")

    with tempfile.TemporaryDirectory() as tmpdirname:
        vllm_config.compilation_config.cache_dir = tmpdirname
        with monkeypatch.context() as m, set_current_vllm_config(vllm_config):
            m.setenv("VLLM_USE_TORCH_DYNAMO_CACHING", "1")
            CompiledMod(vllm_config=vllm_config)(*args)
            torch._dynamo.reset()
            with torch.compiler.set_stance("fail_on_recompile"):
                ret = CompiledMod(vllm_config=vllm_config)(*args)
                assert torch.allclose(ret, expected)


@dataclasses.dataclass
class Setting:
    model: str


@pytest.mark.parametrize("test_setting", [
    Setting(model="Qwen/Qwen3-1.7B", ),
])
def test_model(monkeypatch: pytest.MonkeyPatch, test_setting: Setting):
    monkeypatch.setenv('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)

    llm = LLM(model=test_setting.model, gpu_memory_utilization=0.2)
    outputs = llm.generate(prompts, sampling_params)

    torch._dynamo.reset()
    del llm

    try:
        with torch.compiler.set_stance("fail_on_recompile"):
            llm = LLM(model=test_setting.model, gpu_memory_utilization=0.2)
            llm.generate(prompts, sampling_params)
    except RuntimeError as e:
        assert "Detected recompile" in str(e)
    else:
        raise AssertionError("Expected failed recompilation.")

    monkeypatch.setenv("VLLM_USE_TORCH_DYNAMO_CACHING", "1")
    with tempfile.TemporaryDirectory() as tmpdirname:
        monkeypatch.setenv("VLLM_CACHE_ROOT", tmpdirname)
        llm = LLM(model=test_setting.model, gpu_memory_utilization=0.2)
        outputs_save = llm.generate(prompts, sampling_params)
        assert len(outputs) == len(outputs_save)
        for output, output_save in zip(outputs, outputs_save):
            assert output.outputs[0].text == output_save.outputs[0].text
        torch._dynamo.reset()
        del llm
        with torch.compiler.set_stance("fail_on_recompile"):
            llm = LLM(model=test_setting.model, gpu_memory_utilization=0.2)
            outputs_load = llm.generate(prompts, sampling_params)
        assert len(outputs) == len(outputs_load)
        for output, output_load in zip(outputs, outputs_load):
            assert output.outputs[0].text == output_load.outputs[0].text
