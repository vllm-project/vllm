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


def test_one_option_choice(server):
    response = post(
        server,
        {
            "model": MODEL_NAME,
            "state": "Team: billing.",
            "questions": {
                "team": choice("Which team does the message name?", "billing")
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    team = body["answers"]["team"]
    assert team["choice"] == "billing"
    assert team["probabilities"] == {"billing": 1.0}
    assert team["confidence"] == pytest.approx(
        body["diagnostics"]["team"]["label_mass"]
    )


def test_one_option_coin_flip(server):
    # A fair coin with only "heads" allowed. The answer is forced, so its
    # confidence must come from the model's probability of the label.
    response = post(
        server,
        {
            "model": MODEL_NAME,
            "state": "I flip a fair coin.",
            "questions": {"coin": choice("How did the coin land?", "heads")},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    coin = body["answers"]["coin"]
    assert coin["probabilities"] == {"heads": 1.0}
    assert coin["confidence"] == pytest.approx(
        body["diagnostics"]["coin"]["label_mass"]
    )


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
def test_single_read_answers(server, state, questions, expected):
    response = post(
        server,
        {
            "model": MODEL_NAME,
            "state": state,
            "questions": questions,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response.status_code == 200, response.text
    answers = response.json()["answers"]
    assert {qid: a["choice"] for qid, a in answers.items()} == expected


def test_option_limit(server):
    body = {
        "model": MODEL_NAME,
        "state": "Pick option 7.",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    ok = post(
        server, {**body, "questions": {"n": choice("Which?", *map(str, range(26)))}}
    )
    assert ok.status_code == 200, ok.text
    assert len(ok.json()["answers"]["n"]["probabilities"]) == 26
    over = post(
        server, {**body, "questions": {"n": choice("Which?", *map(str, range(27)))}}
    )
    assert over.status_code == 400
    assert "at most 26 options" in over.json()["error"]["message"]


def test_thinking_is_refused(server):
    response = post(
        server,
        {
            "model": MODEL_NAME,
            "state": "x",
            "questions": {"q": choice("Which?", "a", "b")},
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    assert response.status_code == 400
    assert "thinking must be off" in response.json()["error"]["message"]


def test_noul_and_score(server):
    response = post(
        server,
        {
            "model": MODEL_NAME,
            "state": {"ticket": "My card was charged twice for one order."},
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is money moving?"},
                "tone": {
                    "type": "score",
                    "instructions": "How upset is the customer?",
                    "criteria": ["calm", "annoyed", "furious"],
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    urgent = body["answers"]["urgent"]
    assert urgent["type"] == "noul"
    assert set(urgent["probabilities"]) == {"yes", "no"}
    assert urgent["probabilities"]["yes"] + urgent["probabilities"]["no"] == (
        pytest.approx(1.0, abs=1e-4)
    )
    assert urgent["noul"] == urgent["probabilities"]["yes"]
    assert urgent["confidence"] == pytest.approx(
        max(urgent["probabilities"].values())
        * body["diagnostics"]["urgent"]["label_mass"]
    )
    tone = body["answers"]["tone"]
    assert tone["type"] == "score"
    assert tone["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}
    assert set(tone["probabilities"]) == {"0", "1", "2"}
    assert sum(tone["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert tone["score"] == pytest.approx(
        sum(i * p for i, p in enumerate(tone["probabilities"].values()))
    )
    assert 0 <= tone["score"] <= 2
    assert body["usage"]["output_tokens"] == 2


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
        ({"q": {"type": "noul", "criteria": ["yes", "no"]}}, "must be an object"),
        ({"q": {"type": "score", "criteria": ["calm"]}}, "ordered list of levels"),
    ],
)
def test_validation_errors(server, questions, match):
    response = post(server, {"model": MODEL_NAME, "state": "x", "questions": questions})
    assert response.status_code == 400
    assert match in response.json()["error"]["message"]
