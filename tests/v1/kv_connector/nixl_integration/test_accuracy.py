# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import lm_eval
import openai

BASE_URL = "http://localhost:8192/v1"
NUM_CONCURRENT = 100
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03

# Model-specific expected values
EXPECTED_VALUES = {
    "Qwen/Qwen3-0.6B": 0.41,
    "deepseek-ai/deepseek-vl2-small": 0.59
}

SIMPLE_PROMPT = "The best part about working on vLLM is that I got to meet so many people across various different organizations like UCB, Google, and Meta which means",  # noqa: E501

# Get model name from environment variable
MODEL_NAME = os.environ.get("TEST_MODEL", "Qwen/Qwen3-0.6B")


def run_simple_prompt():
    client = openai.OpenAI(api_key="EMPTY", base_url=BASE_URL)
    completion = client.completions.create(model=MODEL_NAME,
                                           prompt=SIMPLE_PROMPT)

    print("-" * 50)
    print(f"Completion results for {MODEL_NAME}:")
    print(completion)
    print("-" * 50)


def test_accuracy():
    """Run the end to end accuracy test."""
    run_simple_prompt()

    model_args = (f"model={MODEL_NAME},"
                  f"base_url={BASE_URL}/completions,"
                  f"num_concurrent={NUM_CONCURRENT},tokenized_requests=False")

    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=TASK,
    )

    measured_value = results["results"][TASK][FILTER]
    expected_value = EXPECTED_VALUES.get(MODEL_NAME)

    if expected_value is None:
        print(f"Warning: No expected value found for {MODEL_NAME}. "
              "Skipping accuracy check.")
        print(f"Measured value: {measured_value}")
        return

    assert (measured_value - RTOL < expected_value
            and measured_value + RTOL > expected_value
            ), f"Expected: {expected_value} | Measured: {measured_value}"
