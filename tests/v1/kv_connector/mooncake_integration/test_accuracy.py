# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import openai

from tests.evals.gsm8k.gsm8k_eval import (
    assert_min_accuracy,
    evaluate_gsm8k_lm_eval,
)

BASE_URL = "http://localhost:8192/v1"
NUM_CONCURRENT = 100
RTOL = 0.05

# Model-specific expected values
EXPECTED_VALUES = {
    "Qwen/Qwen3-0.6B": 0.41,
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

    model_args = (
        f"model={MODEL_NAME},"
        f"base_url={BASE_URL}/completions,"
        f"num_concurrent={NUM_CONCURRENT},tokenized_requests=False"
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
