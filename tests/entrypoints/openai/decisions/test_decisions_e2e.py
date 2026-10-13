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


def test_openai_decisions_text_questions(server):
    response = requests.post(
        server.url_for("v1/decisions"),
        json={
            "model": MODEL_NAME,
            "input": "The package arrived with a broken screen.",
            "questions": [
                {"type": "predicate", "instructions": "Is the screen broken?"},
                {
                    "type": "choice",
                    "name": "department",
                    "instructions": "Which department should handle this?",
                    "choices": [{"value": "returns"}, {"value": "billing"}],
                },
                {
                    "type": "score",
                    "name": "damage",
                    "instructions": "Rate the damage.",
                    "levels": [{"label": "undamaged"}, {"label": "broken"}],
                },
            ],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"model", "answers", "usage"}
    predicate, choice_answer, score = body["answers"]
    assert predicate["type"] == "predicate"
    assert predicate["name"] is None
    assert 0 <= predicate["probability"] <= 1
    assert choice_answer["type"] == "choice"
    assert choice_answer["name"] == "department"
    assert [p["value"] for p in choice_answer["probabilities"]] == [
        "returns",
        "billing",
    ]
    assert (
        choice_answer["choice"]
        == max(choice_answer["probabilities"], key=lambda p: p["probability"])["value"]
    )
    assert sum(
        p["probability"] for p in choice_answer["probabilities"]
    ) == pytest.approx(1)
    assert score["type"] == "score"
    assert score["name"] == "damage"
    assert [p["label"] for p in score["probabilities"]] == ["undamaged", "broken"]
    assert score["score"] == pytest.approx(
        sum(p["value"] * p["probability"] for p in score["probabilities"])
    )
    usage = body["usage"]
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]


def test_openai_decisions_rejects_images(server):
    response = requests.post(
        server.url_for("v1/decisions"),
        json={
            "model": MODEL_NAME,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.png",
                        }
                    ],
                }
            ],
            "questions": [{"type": "predicate", "instructions": "Is this red?"}],
        },
    )
    assert response.status_code == 400
    assert "text input only" in response.json()["error"]["message"]
