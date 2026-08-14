# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import openai

from tests.evals.gsm8k.gsm8k_eval import (
    assert_gsm8k_result,
    evaluate_gsm8k_lm_eval,
    load_gsm8k_eval_specs,
)

BASE_URL = "http://localhost:8192/v1"
NUM_CONCURRENT = 100
GSM8K_SPECS = {
    spec.model: spec
    for spec in load_gsm8k_eval_specs("mooncake")
    if spec.model is not None
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
    gsm8k_spec = GSM8K_SPECS.get(MODEL_NAME)
    eval_kwargs = gsm8k_spec.lm_eval_kwargs() if gsm8k_spec else {}

    model_args = (
        f"model={MODEL_NAME},"
        f"base_url={BASE_URL}/completions,"
        f"num_concurrent={NUM_CONCURRENT},tokenized_requests=False"
    )
    result = evaluate_gsm8k_lm_eval(
        model="local-completions",
        model_args=model_args,
        **eval_kwargs,
    )

    measured_value = result.accuracy

    print(f"Measured accuracy value: {measured_value}\n")
    if gsm8k_spec is None:
        print(
            f"Warning: No expected value found for {MODEL_NAME}. "
            "Skipping accuracy check."
        )
        return

    assert_gsm8k_result(result, gsm8k_spec, context=MODEL_NAME)
