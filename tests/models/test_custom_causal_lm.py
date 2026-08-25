"""Unit test for custom causal LM model verification."""
import pytest
from vllm import LLM, SamplingParams

TEST_PROMPTS = [
    "Machine learning systems require efficient memory layouts because",
    "Continuous batching improves overall inference throughput by",
]


@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_custom_causal_lm_execution(dtype: str):
    sampling_params = SamplingParams(
        max_tokens=8,
        temperature=0.0,
    )

    llm = LLM(
        model="meta-llama/Llama-3.2-1B",
        dtype=dtype,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
    )

    outputs = llm.generate(TEST_PROMPTS, sampling_params)
    assert len(outputs) == len(TEST_PROMPTS)
    for out in outputs:
        assert len(out.outputs[0].text) > 0