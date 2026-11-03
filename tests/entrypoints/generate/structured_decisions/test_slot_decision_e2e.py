# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""/v1/systemone and /v1/decisions on Qwen3_5ForSlotDecision, served from the
random four-layer checkpoint that test_block_bidirectional.py builds."""

import json

import pytest
import requests

from tests.models.language.generation.test_block_bidirectional import (
    build_tiny_checkpoint,
)
from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="block attention needs a CUDA GPU"
)

MODEL_NAME = "tiny-slot-decision"
# Long enough for the strategy to prefill the shared state once before the reads.
STATE = {
    "ticket": "My card was charged twice for the same order. " * 20,
    "customer": {"tenure_months": 14, "vip": False},
}
QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for money?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "charges", "technical": "bugs", "sales": "pricing"},
    },
    "mood": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["calm", "irritated", "angry"],
    },
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    checkpoint = str(tmp_path_factory.mktemp("slot-decision"))
    build_tiny_checkpoint(checkpoint)
    args = [
        "--dtype",
        "bfloat16",
        "--served-model-name",
        MODEL_NAME,
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.3",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "video": 0}),
    ]
    with RemoteOpenAIServer(checkpoint, args) as remote_server:
        yield remote_server


def systemone(server, **body):
    return requests.post(
        server.url_for("v1/systemone"),
        json={"model": MODEL_NAME, "state": STATE, "questions": QUESTIONS, **body},
    )


def check_probabilities(probabilities: dict[str, float], names: set[str]) -> None:
    assert set(probabilities) == names
    assert all(0.0 <= p <= 1.0 for p in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_systemone(server):
    response = systemone(server)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "structured_decision"
    answers = body["answers"]
    assert list(answers) == list(QUESTIONS)

    # The answers themselves are built by the question types, tested on CPU.
    check_probabilities(answers["refund"]["probabilities"], {"yes", "no"})
    check_probabilities(
        answers["team"]["probabilities"], {"billing", "technical", "sales"}
    )
    check_probabilities(answers["mood"]["probabilities"], {"0", "1", "2"})
    for qid in answers:
        assert 0.0 < body["diagnostics"][qid]["label_mass"] <= 1.0 + 1e-6
    # One sampled token per question.
    assert body["usage"]["output_tokens"] == len(QUESTIONS)

    # The repeat reads the state from the prefix cache.
    again = systemone(server)
    assert again.status_code == 200, again.text
    for qid, answer in again.json()["answers"].items():
        assert answer["probabilities"] == pytest.approx(
            answers[qid]["probabilities"], abs=1e-2
        )


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"images": ["data:image/png;base64,AAAA"]}, "does not take image inputs"),
        ({"instructions": "Be strict."}, "takes no instructions"),
    ],
)
def test_systemone_refusals(server, extra, match):
    response = systemone(server, **extra)
    assert response.status_code == 400
    assert match in response.json()["error"]["message"]


def test_decisions(server):
    response = requests.post(
        server.url_for("v1/decisions"),
        json={
            "model": MODEL_NAME,
            "input": "The package arrived with a broken screen.",
            "questions": [
                {"type": "predicate", "instructions": "Is the screen broken?"},
                {
                    "type": "choice",
                    "instructions": "Which department should handle this?",
                    "choices": [{"value": "returns"}, {"value": "billing"}],
                },
            ],
        },
    )
    assert response.status_code == 200, response.text
    predicate, choice = response.json()["answers"]
    assert 0.0 <= predicate["probability"] <= 1.0
    probabilities = {p["value"]: p["probability"] for p in choice["probabilities"]}
    check_probabilities(probabilities, {"returns", "billing"})
