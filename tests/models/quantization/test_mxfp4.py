# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# flake8: noqa
"""Tests Quark mxfp4 models against ground truth generation"""

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

MODELS = ["amd/Llama-2-7b-chat-hf-wmxfp4-amxfp4-kvfp8-scale-uint8"]

EXPECTED_STRS_MAP = {
    "amd/Llama-2-7b-chat-hf-wmxfp4-amxfp4-kvfp8-scale-uint8": [
        "\n### Key Features\n\n* **High-throughput Inference**: vLL",
        "\nArtificial intelligence (AI) has evolved significantly since its inception in the 1",
        "Artificial intelligence (AI) and human intelligence (HI) are two distinct concepts that have been",
        "A neural network is a machine learning model inspired by the structure of the human brain. It consists of",
        "\nTitle: The Dreaming Robot\n\nAs the sun set on the bustling metropol",
        "\nThe COVID-19 pandemic has had a profound impact on global economic structures and business",
        "The Mona Lisa painting, created by Leonardo da Vinci in the early 16th",
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
            f"Expected: {expected_str!r}\nvLLM: {output_str!r}"
        )


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(120)
    ),
    reason="Regression for MXFP4 MoE without TMA scatter on SM12x",
)
@pytest.mark.quant_model
def test_gpt_oss_triton_unfused_sm12x(vllm_runner):
    """The unfused backend must serve real MXFP4 weights without TMA scatter."""
    with vllm_runner(
        "openai/gpt-oss-20b",
        moe_backend="triton_unfused",
        attention_config={"backend": "TRITON_ATTN"},
        gpu_memory_utilization=0.3,
        kv_cache_memory_bytes=64 << 20,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        enforce_eager=True,
    ) as llm:
        outputs = llm.get_llm().chat(
            [
                [{"role": "user", "content": "What is 17 times 23? Answer briefly."}],
                [{"role": "user", "content": "What is the capital of France?"}],
            ],
            SamplingParams(temperature=0, max_tokens=128),
            chat_template_kwargs={"reasoning_effort": "low"},
        )
    assert "391" in outputs[0].outputs[0].text
    assert "Paris" in outputs[1].outputs[0].text
