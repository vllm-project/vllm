# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import lm_eval

from ...utils import RemoteOpenAIServer

# arc-easy uses prompt_logprobs=1, logprobs=1
TASK = "arc_easy"
FILTER = "acc_norm,none"
RTOL = 0.03
EXPECTED_VALUE = 0.62

# FIXME(rob): enable prefix caching once supported.
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_ARGS = f"pretrained={MODEL},enforce_eager=True,enable_prefix_caching=False"  # noqa: E501
SERVER_ARGS = [
    "--enforce_eager", "--no_enable_prefix_caching", "--disable-log-requests"
]
NUM_CONCURRENT = 100


def test_prompt_logprobs_e2e():
    results = lm_eval.simple_evaluate(model="vllm",
                                      model_args=MODEL_ARGS,
                                      tasks=TASK,
                                      batch_size="auto")

    measured_value = results["results"][TASK][FILTER]
    assert (measured_value - RTOL < EXPECTED_VALUE
            and measured_value + RTOL > EXPECTED_VALUE
            ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"


def test_prompt_logprobs_e2e_server():
    with RemoteOpenAIServer(MODEL, SERVER_ARGS) as remote_server:
        url = f"{remote_server.url_for('v1')}/completions"

        model_args = (
            f"model={MODEL},"
            f"base_url={url},"
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
