# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# flake8: noqa
"""Tests Model Optimizer nvfp4 models against ground truth generation
Note: these tests will only pass on B200
"""
import os
from typing import List

import pytest
from transformers import AutoTokenizer

from tests.quantization.utils import is_quant_method_supported
from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024

MODELS = ["nvidia/Llama-3.3-70B-Instruct-FP4"]

EXPECTED_STRS_MAP = {
    "nvidia/Llama-3.3-70B-Instruct-FP4": [
        'vLLM (Vectorized Large Language Model) is indeed a high-throughput and memory-efficient inference',
        'Here are the major milestones in the development of artificial intelligence (AI) from 1950 to ',
        'Artificial intelligence (AI) and human intelligence (HI) are two distinct forms of intelligence that process',
        'A neural network is a type of machine learning model inspired by the structure and function of the human brain',
        'In the heart of a cutting-edge robotics lab, a team of engineers had been working tirelessly to push',
        'The COVID-19 pandemic has had a profound impact on global economic structures and future business models, leading',
        'The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is one of',
        'Here are the translations:\n\n* Japanese: (Sasuga no tori ga miwa o ts'
    ]
}


# This test compares against golden strings for exact match since
# there is no baseline implementation to compare against
# and is unstable w.r.t specifics of the fp4 implementation or
# the hardware being run on.
# Disabled to prevent it from breaking the build
@pytest.mark.skip(
    reason=
    "Prevent unstable test based on golden strings from breaking the build "
    " and test input model being too large and hanging the system.")
@pytest.mark.skipif(not is_quant_method_supported("modelopt_fp4"),
                    reason="modelopt_fp4 is not supported on this GPU type.")
@pytest.mark.parametrize("model_name", MODELS)
def test_models(example_prompts, model_name) -> None:
    model = LLM(
        model=model_name,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        enforce_eager=True,
        quantization="modelopt_fp4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted_prompts = [
        tokenizer.apply_chat_template([{
            "role": "user",
            "content": prompt
        }],
                                      tokenize=False,
                                      add_generation_prompt=True)
        for prompt in example_prompts
    ]
    params = SamplingParams(max_tokens=20, temperature=0)
    generations: List[str] = []
    # Note: these need to be run 1 at a time due to numerical precision,
    # since the expected strs were generated this way.
    for prompt in formatted_prompts:
        outputs = model.generate(prompt, params)
        generations.append(outputs[0].outputs[0].text)
    del model

    print(model_name, generations)
    expected_strs = EXPECTED_STRS_MAP[model_name]
    for i in range(len(example_prompts)):
        generated_str = generations[i]
        expected_str = expected_strs[i]
        assert expected_str == generated_str, (
            f"Test{i}:\nExpected: {expected_str!r}\nvLLM: {generated_str!r}")
