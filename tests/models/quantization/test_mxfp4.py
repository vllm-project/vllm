# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# flake8: noqa
"""Tests Quark mxfp4 models against ground truth generation
"""
import pytest

from vllm import LLM, SamplingParams

MODELS = ["amd/Llama-2-7b-chat-hf-wmxfp4-amxfp4-kvfp8-scale-uint8"]

EXPECTED_STRS_MAP = {
    "amd/Llama-2-7b-chat-hf-wmxfp4-amxfp4-kvfp8-scale-uint8": [
        '\n### Key Features\n\n* **High-throughput Inference**: vLL',
        '\nArtificial intelligence (AI) has evolved significantly since its inception in the 1',
        'Artificial intelligence (AI) and human intelligence (HI) are two distinct concepts that have been',
        'A neural network is a machine learning model inspired by the structure of the human brain. It consists of',
        '\nTitle: The Dreaming Robot\n\nAs the sun set on the bustling metropol',
        '\nThe COVID-19 pandemic has had a profound impact on global economic structures and business',
        'The Mona Lisa painting, created by Leonardo da Vinci in the early 16th',
        " everybody knows this proverbial saying, but did you know that it's not entirely accurate?",
    ]
}


@pytest.mark.skip(reason="Model to be released in the future")
@pytest.mark.quant_model
@pytest.mark.parametrize("model_name", MODELS)
def test_models(example_prompts, model_name) -> None:
    sampling_params = SamplingParams(max_tokens=20, temperature=0)
    llm = LLM(
        model=model_name,
        kv_cache_dtype="fp8",
        quantization="quark",
    )
    outputs = llm.generate(example_prompts, sampling_params)
    for i, output in enumerate(outputs):
        output_str = output.outputs[0].text
        expected_str = EXPECTED_STRS_MAP[model_name][i]
        assert expected_str == output_str, (
            f"Expected: {expected_str!r}\nvLLM: {output_str!r}")
