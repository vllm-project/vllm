# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import openai

from tests.evals.gsm8k.gsm8k_eval import (
    assert_min_accuracy,
    evaluate_gsm8k_lm_eval,
)

BASE_URL = "http://localhost:8192/v1"
NUM_CONCURRENT = int(os.getenv("NUM_CONCURRENT", "100"))
# TODO(#43186): Widened from 0.03 to absorb chunk_scan/SSU numeric jitter
# on granite-4.0-h-tiny under NIXL PD; tighten when the kernel divergence
# is fixed.
RTOL = 0.05

# Model-specific expected values
EXPECTED_VALUES = {
    "Qwen/Qwen3-0.6B": 0.41,
    "deepseek-ai/deepseek-vl2-small": 0.59,
    "deepseek-ai/deepseek-vl2-tiny": 0.19,
    "deepseek-ai/DeepSeek-V2-Lite-Chat": 0.65,
    "google/gemma-3-4b-it": 0.74,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": 0.84,
    "ibm-granite/granite-4.0-h-tiny": 0.77,
    "Qwen/Qwen3.5-0.8B": 0.33,
    "google/gemma-4-E2B-it": 0.485,
    "ai21labs/AI21-Jamba2-3B": 0.74,
    "deepseek-ai/DeepSeek-V4-Flash": 0.95,
}

SIMPLE_PROMPT = (
    (
        "The best part about working on vLLM is that I got to meet so many people "
        "across various different organizations like UCB, Google, and Meta which means"
    ),
)

# Get model name from environment variable
MODEL_NAME = os.environ.get("TEST_MODEL", "Qwen/Qwen3-0.6B")


def run_simple_prompt():
    client = openai.OpenAI(api_key="EMPTY", base_url=BASE_URL)
    completion = client.completions.create(model=MODEL_NAME, prompt=SIMPLE_PROMPT)

    print("-" * 50)
    print(f"Completion results for {MODEL_NAME}:")
    print(completion)
    print("-" * 50)


def test_accuracy():
    """Run the end to end accuracy test."""
    run_simple_prompt()

    if "gemma-4" in MODEL_NAME:
        # Gemma4 is quite sensible to having a chat template applied, so we evaluate
        # on chat completions.
        model_args = (
            f"model={MODEL_NAME},"
            f"base_url={BASE_URL}/chat/completions,"
            f"num_concurrent={NUM_CONCURRENT},"
            "tokenizer_backend=huggingface,"
            "trust_remote_code=True"
        )
        result = evaluate_gsm8k_lm_eval(
            model="local-chat-completions",
            model_args=model_args,
            num_fewshot=5,
            apply_chat_template=True,
        )
    else:
        model_args = (
            f"model={MODEL_NAME},"
            f"base_url={BASE_URL}/completions,"
            f"num_concurrent={NUM_CONCURRENT},tokenized_requests=False,"
            "trust_remote_code=True"
        )
        result = evaluate_gsm8k_lm_eval(
            model="local-completions",
            model_args=model_args,
        )

    measured_value = result.accuracy
    expected_value = EXPECTED_VALUES.get(MODEL_NAME)

    print(f"Measured accuracy value: {measured_value}\n")
    if expected_value is None:
        print(
            f"Warning: No expected value found for {MODEL_NAME}. "
            "Skipping accuracy check."
        )
        return

    assert_min_accuracy(result, expected_value, tolerance=RTOL, context=MODEL_NAME)
