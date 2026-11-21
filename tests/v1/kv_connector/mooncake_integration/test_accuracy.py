# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import lm_eval
import openai
import requests

BASE_URL = f"http://localhost:{os.environ.get('PROXY_PORT', 8192)}/v1"
# Comma-separated server URLs to sleep and wake up before the eval.
SLEEP_WAKE_URLS = [
    url for url in os.environ.get("SLEEP_WAKE_URLS", "").split(",") if url
]
NUM_CONCURRENT = 100
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.05

# Model-specific expected values
EXPECTED_VALUES = {
    "Qwen/Qwen3-0.6B": 0.41,
}

SIMPLE_PROMPT = (
    "The best part about working on vLLM is that I got to meet so many people across "
    "various different organizations like UCB, Google, and Meta which means"
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


def sleep_and_wake_up(urls: list[str]):
    for url in urls:
        requests.post(f"{url}/sleep", params={"level": 1}).raise_for_status()
    for url in urls:
        requests.post(f"{url}/wake_up").raise_for_status()


def test_accuracy():
    """Run the end to end accuracy test."""
    run_simple_prompt()
    if SLEEP_WAKE_URLS:
        sleep_and_wake_up(SLEEP_WAKE_URLS)

    model_args = (
        f"model={MODEL_NAME},"
        f"base_url={BASE_URL}/completions,"
        f"num_concurrent={NUM_CONCURRENT},tokenized_requests=False"
    )
    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=TASK,
    )

    measured_value = results["results"][TASK][FILTER]
    expected_value = EXPECTED_VALUES.get(MODEL_NAME)

    print(f"Measured accuracy value: {measured_value}\n")
    if expected_value is None:
        print(
            f"Warning: No expected value found for {MODEL_NAME}. "
            "Skipping accuracy check."
        )
        return

    assert (
        measured_value - RTOL < expected_value
        and measured_value + RTOL > expected_value
    ), f"Expected: {expected_value} | Measured: {measured_value}"
