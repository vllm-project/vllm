# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--enforce-eager",
        "--max-num-seqs",
        "32",
        "--enable-prefix-caching",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def post(server, body):
    return requests.post(server.url_for("v1/systemone"), json=body)


def test_choice_decision(server):
    response = post(
        server,
        {
            "model": MODEL_NAME,
            "state": {"ticket": "My card was charged twice for one order."},
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": {
                        "billing": "payments and refunds",
                        "shipping": "deliveries",
                        "security": "account access",
                    },
                },
                "lang": {
                    "type": "choice",
                    "instructions": "Which language is the ticket written in?",
                    "criteria": {"English": None, "French": None},
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "structured_decision"
    assert list(body["answers"]) == ["team", "lang"]
    team = body["answers"]["team"]
    assert team["type"] == "choice"
    assert set(team["probabilities"]) == {"billing", "shipping", "security"}
    assert sum(team["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert team["confidence"] == max(team["probabilities"].values())
    assert team["choice"] == "billing"
    assert body["answers"]["lang"]["choice"] == "English"
    for diag in body["diagnostics"].values():
        assert 0.0 < diag["label_mass"] <= 1.0 + 1e-6
    assert body["usage"]["output_tokens"] == 2
    assert body["usage"]["input_tokens"] > 0


def test_answers_repeat(server):
    body = {
        "model": MODEL_NAME,
        "state": "The package arrived crushed.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": None, "shipping": None},
            }
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }
    first = post(server, body).json()["answers"]["team"]["probabilities"]
    second = post(server, body).json()["answers"]["team"]["probabilities"]
    for name in first:
        assert first[name] == pytest.approx(second[name], abs=1e-3)


@pytest.mark.parametrize(
    "questions,match",
    [
        ({}, "at least one question"),
        (
            {"q": {"type": "nope", "criteria": {"a": None, "b": None}}},
            "unknown question type",
        ),
        ({"q": {"type": "choice", "criteria": {"a": None}}}, "2 to 26"),
        (
            {
                "q": {
                    "type": "choice",
                    "criteria": {"a": None, "b": None},
                    "depends_on": ["x"],
                }
            },
            "not supported yet",
        ),
    ],
)
def test_validation_errors(server, questions, match):
    response = post(server, {"model": MODEL_NAME, "state": "x", "questions": questions})
    assert response.status_code == 400
    assert match in response.json()["error"]["message"]


def test_unknown_model(server):
    response = post(
        server,
        {
            "model": "not-a-model",
            "state": "x",
            "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}},
        },
    )
    assert response.status_code == 404
