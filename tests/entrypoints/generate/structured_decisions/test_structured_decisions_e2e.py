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
        "--trust-request-chat-template",
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
        ({"q": {"type": "choice", "criteria": {"a": None}}}, "at least 2"),
        (
            {
                "q": {
                    "type": "choice",
                    "criteria": {"a": None, "b": None},
                    "depends_on": ["x"],
                }
            },
            "unknown field(s)",
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


TEMPLATE_BODY = {
    "model": MODEL_NAME,
    "state": "My card was charged twice for one order.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": None, "shipping": None},
        }
    },
    "chat_template_kwargs": {"enable_thinking": False},
}


def test_request_template(server):
    template = (
        "{% macro answer_prefix(question) %}{{ question.id }} ->{% endmacro %}"
        "Classify the ticket.\n"
        "{% for q in questions %}{{ q.instructions }}\n"
        "{% for o in q.options %}{{ o.label }} = {{ o.name }}\n{% endfor %}"
        "{% endfor %}"
        "Reply as: id -> label"
    )
    custom = post(server, dict(TEMPLATE_BODY, decision_template=template))
    assert custom.status_code == 200, custom.text
    default = post(server, TEMPLATE_BODY)
    assert default.status_code == 200, default.text
    custom, default = custom.json(), default.json()
    # The macro's prefix puts the read where the model expects a label.
    assert custom["diagnostics"]["team"]["label_mass"] > 0.5
    # The request's template, not the server's, rendered the prompt.
    assert custom["answers"]["team"]["probabilities"]["billing"] != pytest.approx(
        default["answers"]["team"]["probabilities"]["billing"], abs=1e-3
    )


def test_request_template_that_does_not_compile(server):
    response = post(
        server, dict(TEMPLATE_BODY, decision_template="{% for q in questions %}")
    )
    assert response.status_code == 400
    assert "decision template" in response.json()["error"]["message"]


def test_too_many_questions(server):
    questions = {
        f"q{i}": {"type": "choice", "criteria": {"a": None, "b": None}}
        for i in range(65)
    }
    response = post(server, {"model": MODEL_NAME, "state": "x", "questions": questions})
    assert response.status_code == 400
    assert "at most 64" in response.json()["error"]["message"]
