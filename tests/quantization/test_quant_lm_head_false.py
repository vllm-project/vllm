from typing import Optional

import pytest

from vllm import CompletionOutput, SamplingParams

MODEL = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
MAX_TOKENS = 100


@pytest.fixture(scope="session")
def vllm_model(vllm_runner):
    return vllm_runner(
        MODEL,
        enforce_eager=True,
    )


@pytest.mark.skip_global_cleanup
def test_lm_head_false(vllm_model):
    llm_engine = vllm_model.model.llm_engine

    quantization_config = llm_engine.model_config.hf_config.quantization_config
    assert not quantization_config.get("lm_head")

    llm_engine.add_request(
        "id", "A story about life in 1978:\n",
        SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        ), None)

    expected_output = "\nIn 1978, I was a freshman in high school."

    output: Optional[CompletionOutput] = None
    output_text = ""
    while llm_engine.has_unfinished_requests():
        (request_output,) = llm_engine.step()
        (output,) = request_output.outputs

        # Ensure we don't backtrack
        assert output.text.startswith(output_text)
        output_text = output.text

    assert output is not None
    assert output_text.startswith(expected_output)
