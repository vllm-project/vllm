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


def choice(instructions, *names):
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": dict.fromkeys(names),
    }


COPY = {
    "team": choice(
        "Which team does the message name?", "billing", "shipping", "security"
    ),
    "lang": choice("Which language does the message name?", "English", "French"),
}


def test_choice_decision(server):
    response = post(
        server,
        {
            "model": MODEL_NAME,
            "state": {"team": "billing", "language": "French"},
            "questions": COPY,
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
    assert team["confidence"] == pytest.approx(
        max(team["probabilities"].values()) * body["diagnostics"]["team"]["label_mass"]
    )
    for diag in body["diagnostics"].values():
        assert 0.0 < diag["label_mass"] <= 1.0 + 1e-6
    assert body["usage"]["output_tokens"] == 2
    assert body["usage"]["input_tokens"] > 0


# Qwen3-0.6B gets a single read of these wrong for some label shuffles, so
# each case averages 16 seeds.
@pytest.mark.parametrize(
    "state,questions,expected",
    [
        (
            {"team": "billing", "language": "French"},
            COPY,
            {"team": "billing", "lang": "French"},
        ),
        (
            "My card was charged twice for one order.",
            {
                "card": choice("Does the message mention a card?", "yes", "no"),
                "mood": choice("How does the customer feel?", "happy", "angry"),
            },
            {"card": "yes", "mood": "angry"},
        ),
        (
            "Hello.",
            {
                "sky": choice(
                    "What color is the sky on a clear day?", "blue", "green", "red"
                )
            },
            {"sky": "blue"},
        ),
    ],
)
def test_answers_averaged_over_seeds(server, state, questions, expected):
    totals: dict[str, dict[str, float]] = {}
    for seed in range(16):
        response = post(
            server,
            {
                "model": MODEL_NAME,
                "state": state,
                "questions": questions,
                "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        assert response.status_code == 200, response.text
        for qid, answer in response.json()["answers"].items():
            for name, p in answer["probabilities"].items():
                totals.setdefault(qid, {}).setdefault(name, 0.0)
                totals[qid][name] += p
    assert {qid: max(t, key=lambda n: t[n]) for qid, t in totals.items()} == expected


@pytest.mark.parametrize(
    "questions,match",
    [
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
        (
            {
                f"q{i}": {"type": "choice", "criteria": {"a": None, "b": None}}
                for i in range(65)
            },
            "at most 64",
        ),
    ],
)
def test_validation_errors(server, questions, match):
    response = post(server, {"model": MODEL_NAME, "state": "x", "questions": questions})
    assert response.status_code == 400
    assert match in response.json()["error"]["message"]
