# SPDX-License-Identifier: Apache-2.0
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel


def run_model(compilation_config: CompilationConfig):
    prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

    llm = LLM(model="Qwen/Qwen2-1.5B-Instruct",
              compilation_config=compilation_config)

    return llm.generate(prompts, sampling_params)


def test_full_cudagraph(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_FLASH_ATTN_VERSION", "3")

        full_cudagraph_responses = run_model(
            compilation_config=CompilationConfig(
                level=CompilationLevel.FULL_GRAPH,
                use_cudagraph=True,
            ))

        piecewise_responses = run_model(compilation_config=CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            use_cudagraph=True,
        ))

        assert full_cudagraph_responses[0].outputs[
            0].text == piecewise_responses[0].outputs[0].text
