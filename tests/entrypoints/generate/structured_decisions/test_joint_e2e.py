# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import string
from typing import Any

import pytest
import requests

from tests.utils import RemoteOpenAIServer, large_gpu_mark

MODEL_NAME = "google/diffusiongemma-26B-A4B-it"
CANVAS = 64

pytestmark = large_gpu_mark(min_gb=80)


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.85",
        "--diffusion-config",
        f'{{"canvas_length": {CANVAS}}}',
        "--max-logprobs",
        "32",
        "--enable-prefix-caching",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args, max_wait_seconds=900) as remote_server:
        yield remote_server


def post(server, body):
    return requests.post(server.url_for("v1/systemone"), json=body)


TICKET: dict[str, Any] = {
    "state": "Everything is down since 9am. Your invoice double charged us. "
    "Fix this NOW.",
    "questions": {
        "urgent": {
            "type": "noul",
            "instructions": "Does the customer need a reply within the hour?",
        },
        "bucket": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "payments",
                "outage": "service down",
                "feature": None,
            },
        },
        "tone": {
            "type": "score",
            "instructions": "How upset is the customer?",
            "criteria": ["calm", "annoyed", "furious"],
        },
    },
}


def test_joint_one_canvas(server):
    response = post(server, {**TICKET, "joint": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body["answers"]) == ["urgent", "bucket", "tone"]
    assert body["answers"]["urgent"]["noul"] > 0.5
    assert body["answers"]["tone"]["score"] >= 1
    (canvas,) = body["joint"]["canvases"]
    assert canvas["questions"] == ["urgent", "bucket", "tone"]
    assert canvas["width"] % 16 == 0
    assert len(canvas["slots"]) == 3
    assert canvas["slots"] == sorted(canvas["slots"])
    assert body["usage"]["output_tokens"] == canvas["slots"][-1] + 1
    for qid, answer in body["answers"].items():
        assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
        label_mass = body["diagnostics"][qid]["label_mass"]
        assert 0.5 < label_mass <= 1.0 + 1e-6
    again = post(server, {**TICKET, "joint": True}).json()["answers"]
    for qid, answer in body["answers"].items():
        for label, p in answer["probabilities"].items():
            assert again[qid]["probabilities"][label] == pytest.approx(p, abs=0.02)


def test_joint_matches_per_question_on_a_clear_case(server):
    body = {
        "state": "I was charged twice for order 1182. Please refund one charge.",
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Does the customer ask for a refund?",
            },
            "bucket": TICKET["questions"]["bucket"],
        },
    }
    joint = post(server, {**body, "joint": True}).json()["answers"]
    response = post(server, body).json()
    assert response["joint"] is None
    alone = response["answers"]
    assert joint["refund"]["noul"] > 0.5 and alone["refund"]["noul"] > 0.5
    assert joint["bucket"]["choice"] == alone["bucket"]["choice"] == "billing"


def test_joint_few_questions_keep_label_mass(server):
    fruit = {
        "type": "choice",
        "instructions": "Which fruit did I buy most of?",
        "criteria": dict.fromkeys(["apple", "pear", "banana", "cherry"]),
    }
    cherry = {"type": "noul", "instructions": "Did I buy cherries?"}
    body = {
        "state": "I bought apples, a pear and three bananas.",
        "questions": {"fruit": fruit, "cherry": cherry},
        "joint": True,
    }
    response = post(server, body).json()
    assert len(response["joint"]["canvases"]) == 1
    assert response["answers"]["fruit"]["choice"] == "banana"
    for diagnostics in response["diagnostics"].values():
        assert diagnostics["label_mass"] > 0.8
        assert diagnostics["argmax_is_label"]


def test_joint_answers_do_not_follow_line_numbers(server):
    letters = string.ascii_lowercase
    questions = {
        f"q{i}": {
            "type": "choice",
            "instructions": f"Letter {i % 26 + 1} of the alphabet?",
            "criteria": dict.fromkeys(letters),
        }
        for i in range(64)
    }
    state = (
        "Ticket 4471: our Android app crashes on login since v2.3; enterprise "
        "plan, renewal next month, customer is furious and mentions switching "
        "to a competitor."
    )
    body = {"state": state, "questions": questions, "joint": True}
    response = post(server, body).json()
    assert len(response["joint"]["canvases"]) > 1
    answers = response["answers"]
    assert sum(answers[f"q{i}"]["choice"] == letters[i % 26] for i in range(64)) >= 60
