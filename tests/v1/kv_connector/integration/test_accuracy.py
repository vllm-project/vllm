# SPDX-License-Identifier: Apache-2.0
import lm_eval
import openai

BASE_URL = "http://localhost:8192/v1"
MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_CONCURRENT = 100
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
EXPECTED_VALUE = 0.41

SIMPLE_PROMPT = "The best part about working on vLLM is that I got to meet so many people across various different organizations like UCB, Google, and Meta which means",  # noqa: E501


def run_simple_prompt():
    client = openai.OpenAI(api_key="EMPTY", base_url=BASE_URL)
    completion = client.completions.create(model=MODEL_NAME,
                                           prompt=SIMPLE_PROMPT)

    print("-" * 50)
    print("Completion results:")
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
    assert (measured_value - RTOL < EXPECTED_VALUE
            and measured_value + RTOL > EXPECTED_VALUE
            ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"
