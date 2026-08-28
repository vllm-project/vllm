# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# flake8: noqa
"""Tests Model Optimizer fp8 models against ground truth generation
Note: these tests will only pass on H100
"""

import os

import pytest
from transformers import AutoTokenizer

from tests.quantization.utils import is_quant_method_supported
from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024

MODELS = ["nvidia/Llama-3.1-8B-Instruct-FP8"]

EXPECTED_STRS_MAP = {
    "nvidia/Llama-3.1-8B-Instruct-FP8": [
        "You're referring to VLLM, a high-performance Large Language Model (LLM) inference and",
        "Here are the major milestones in the development of artificial intelligence (AI) from 1950 to ",
        "The comparison between artificial intelligence (AI) and human intelligence in terms of processing information is a complex and",
        'A neural network is a complex system modeled after the human brain, consisting of interconnected nodes or "ne',
        "**The Spark of Imagination**\n\nZeta-5, a sleek and efficient robot, whir",
        "The COVID-19 pandemic has had a profound impact on global economic structures and business models, leading to",
        "The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is one of",
        "Here are the translations:\n\n**Japanese:** 「早起きは早く獲物をとる",
    ]
}


# This test compares against golden strings for exact match since
# there is no baseline implementation to compare against
# and is unstable w.r.t specifics of the fp8 implementation or
# the hardware being run on.
# Disabled to prevent it from breaking the build
@pytest.mark.skip(
    reason="Prevent unstable test based on golden strings from breaking the build."
)
@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="fp8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_name", MODELS)
def test_models(example_prompts, model_name) -> None:
    llm = LLM(
        model=model_name,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        enforce_eager=True,
        quantization="modelopt",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in example_prompts
    ]
    params = SamplingParams(max_tokens=20, temperature=0)
    generations: list[str] = []
    # Note: these need to be run 1 at a time due to numerical precision,
    # since the expected strs were generated this way.
    for prompt in formatted_prompts:
        outputs = llm.generate(prompt, params)
        generations.append(outputs[0].outputs[0].text)
    del llm

    print(model_name, generations)
    expected_strs = EXPECTED_STRS_MAP[model_name]
    for i in range(len(example_prompts)):
        generated_str = generations[i]
        expected_str = expected_strs[i]
        assert expected_str == generated_str, (
            f"Test{i}:\nExpected: {expected_str!r}\nvLLM: {generated_str!r}"
        )
