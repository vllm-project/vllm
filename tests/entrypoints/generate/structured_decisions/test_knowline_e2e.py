# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KnowLine served end to end: a tiny Qwen3.5 checkpoint opts in with the
knowline marker, startup selects the strategy and discovers the labels on
the real tokenizer, and /v1/systemone answers through the trained prompt."""

import httpx
import pytest

from tests.utils import RemoteOpenAIServer

MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def server():
    with RemoteOpenAIServer(
        MODEL,
        [
            "--hf-overrides",
            '{"knowline_config": {}}',
            "--enforce-eager",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "8",
        ],
        max_wait_seconds=480,
    ) as server:
        yield server


def test_systemone_answers_a_question(server):
    response = httpx.post(
        server.url_for("v1", "systemone"),
        json={
            "model": MODEL,
            "state": "I was billed twice for the same order this month.",
            "questions": {
                "billed": {
                    "type": "noul",
                    "instructions": "Was the customer billed twice?",
                }
            },
        },
        timeout=120,
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["answers"]["billed"]["noul"] <= 1.0
    assert body["usage"]["input_tokens"] > 0


def test_systemone_answers_130_options_without_flags(server):
    # Above the 128-id cap the read used to mask the labels and needed
    # --logprobs-mode processed_logprobs --max-logprobs 255; one read
    # carries every label now, so no flag is needed.
    criteria = {f"opt{i}": f"outcome {i}" for i in range(130)}
    response = httpx.post(
        server.url_for("v1", "systemone"),
        json={
            "model": MODEL,
            "state": "The customer was billed twice for the same order.",
            "questions": {
                "big": {
                    "type": "choice",
                    "instructions": "Which outcome fits the ticket?",
                    "criteria": criteria,
                }
            },
        },
        timeout=120,
    )
    assert response.status_code == 200
    answer = response.json()["answers"]["big"]
    assert answer["choice"] in criteria
    assert set(answer["probabilities"]) == set(criteria)
