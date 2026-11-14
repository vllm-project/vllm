# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import encode_image_url

MODEL_NAME = "Qwen/Qwen3-0.6B"
VISION_MODEL_NAME = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.5",
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
                            "image_url": encode_image_url(
                                ImageAsset("stop_sign").pil_image
                            ),
                        }
                    ],
                }
            ],
            "questions": [{"type": "predicate", "instructions": "Is this red?"}],
        },
    )
    assert response.status_code == 400
    assert "does not support image input" in response.json()["error"]["message"]


@pytest.fixture(scope="module")
def vision_server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.5",
        "--limit-mm-per-prompt",
        '{"image": 1}',
        "--logprobs-mode",
        "raw_logprobs",
    ]
    with RemoteOpenAIServer(VISION_MODEL_NAME, args) as remote_server:
        yield remote_server


def test_openai_decisions_real_image(vision_server):
    """Read all question types from a real photo using the OpenAI Decisions SDK."""
    with vision_server.get_client(timeout=120) as client:
        decision = client.decisions.create(
            model=VISION_MODEL_NAME,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Inspect this photograph."},
                        {
                            "type": "input_image",
                            "image_url": encode_image_url(
                                ImageAsset("stop_sign").pil_image
                            ),
                        },
                    ],
                }
            ],
            questions=[
                {
                    "type": "predicate",
                    "name": "sign_present",
                    "instructions": "Is a traffic sign visible?",
                },
                {
                    "type": "choice",
                    "name": "object",
                    "instructions": "Which object is shown in the photograph?",
                    "choices": [{"value": "stop sign"}, {"value": "coffee cup"}],
                },
                {
                    "type": "score",
                    "name": "visibility",
                    "instructions": "How visible is the traffic sign?",
                    "levels": [{"label": "not visible"}, {"label": "clearly visible"}],
                },
            ],
        )
    body = decision.model_dump()
    predicate, choice, score = body["answers"]
    assert [answer["name"] for answer in body["answers"]] == [
        "sign_present",
        "object",
        "visibility",
    ]
    assert predicate["type"] == "predicate"
    assert 0 <= predicate["probability"] <= 1
    assert choice["type"] == "choice"
    assert choice["choice"] == "stop sign"
    assert [p["value"] for p in choice["probabilities"]] == ["stop sign", "coffee cup"]
    assert score["type"] == "score"
    assert [p["label"] for p in score["probabilities"]] == [
        "not visible",
        "clearly visible",
    ]
    for answer in (choice, score):
        probabilities = [p["probability"] for p in answer["probabilities"]]
        assert all(0 <= probability <= 1 for probability in probabilities)
        assert sum(probabilities) == pytest.approx(1)
    assert score["score"] == pytest.approx(
        sum(p["value"] * p["probability"] for p in score["probabilities"])
    )
    usage = body["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == usage["input_tokens"] + 3
