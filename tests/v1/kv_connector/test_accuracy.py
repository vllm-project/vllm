# SPDX-License-Identifier: Apache-2.0
import lm_eval

MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_CONCURRENT = 100
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
EXPECTED_VALUE = 0.41


def test_accuracy():
    """Run the end to end accuracy test."""

    model_args = (f"model={MODEL_NAME},"
                  f"base_url=http://localhost:8192/v1/completions,"
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
