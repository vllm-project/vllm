# flake8: noqa
"""Tests fp8 models against ground truth generation
Note: these tests will only pass on L4 GPU.
"""
import os
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from tests.quantization.utils import is_quant_method_supported
from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024

MODELS = [
    "nm-testing/Meta-Llama-3-8B-Instruct-FP8-KV",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]

EXPECTED_STRS_MAP = {
    "nm-testing/Meta-Llama-3-8B-Instruct-FP8-KV": {
        "auto": [
            'LLaMA is a high-throughput and memory-efficient inference and serving engine for Large Language Models (',
            'Here are the major milestones in the development of artificial intelligence (AI) from 1950 to ',
            'Artificial intelligence (AI) and human intelligence (HI) process information in distinct ways, with both',
            'A neural network is a complex system modeled after the human brain, composed of interconnected nodes or "ne',
            'Zeta-5, a highly advanced robot designed for menial labor, whirred and beep',
            'The COVID-19 pandemic has had a profound impact on global economic structures and future business models. The',
            'The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is one of',
            'Here are the translations:\n\n**Japanese:** (Haya aki no tori, nemuri no'
        ],
        "fp8": [
            'LLM (Large Language Model) is a type of artificial intelligence (AI) model that is trained',
            'Here are the major milestones in the development of artificial intelligence (AI) from 1950 to ',
            'Artificial intelligence (AI) and human intelligence (HI) differ significantly in how they process information.',
            'A neural network is a complex system made up of several basic components that work together to enable it to',
            'Zeta-5, a highly advanced robot designed for menial labor, had never experienced anything like',
            'The COVID-19 pandemic has had a profound impact on global economic structures and future business models. Here',
            'The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is one of',
            'Here are the translations:\n\n**Japanese:** (Haya kotori wa mushi o tsuk'
        ]
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "auto": [
            'LLM (Large Language Model) is a type of artificial intelligence (AI) model that is trained',
            'Here are the major milestones in the development of artificial intelligence (AI) from 1950 to ',
            'Artificial intelligence (AI) and human intelligence (HI) differ significantly in how they process information.',
            'A neural network is a complex system modeled after the human brain, composed of interconnected nodes or "ne',
            'In the vast, sterile laboratory, Robot 3456-Alpha, or "Alpha" for short',
            'The COVID-19 pandemic has had a profound impact on global economic structures and future business models. The',
            'The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is one of',
            'Here are the translations:\n\n**Japanese:** (Haya aki wa mushi o tsukamu'
        ],
        "fp8": [
            'LLM (Large Language Model) is a type of artificial intelligence (AI) model that is trained',
            'Here are the major milestones in the development of artificial intelligence (AI) from 1950 to ',
            'Artificial intelligence (AI) and human intelligence (HI) differ significantly in how they process information.',
            'A neural network is a complex system modeled after the human brain, consisting of interconnected nodes or "ne',
            'In the year 2154, robotics engineer Dr. Rachel Kim had spent years perfecting her latest',
            'The COVID-19 pandemic has had a profound impact on global economic structures and future business models. The',
            'The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is one of',
            'Here are the translations:\n\n**Japanese:** (Haya tori, mushi o tsukamu'
        ]
    },
}


# This test compares against golden strings for exact match since
# there is no baseline implementation to compare against
# and is unstable w.r.t specifics of the fp8 implementation or
# the hardware being run on.
# Disabled to prevent it from breaking the build
@pytest.mark.skip(
    reason=
    "Prevent unstable test based on golden strings from breaking the build.")
@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="fp8 is not supported on this GPU type.")
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_models(example_prompts, model_name, kv_cache_dtype) -> None:
    model = LLM(model=model_name,
                max_model_len=MAX_MODEL_LEN,
                trust_remote_code=True,
                enforce_eager=True,
                quantization="fp8",
                kv_cache_dtype=kv_cache_dtype)

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

    print(model_name, kv_cache_dtype, generations)
    expected_strs = EXPECTED_STRS_MAP[model_name][kv_cache_dtype]
    for i in range(len(example_prompts)):
        generated_str = generations[i]
        expected_str = expected_strs[i]
        assert expected_str == generated_str, (
            f"Test{i}:\nExpected: {expected_str!r}\nvLLM: {generated_str!r}")
