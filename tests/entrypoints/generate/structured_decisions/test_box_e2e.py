# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from io import BytesIO

import pybase64
import pytest
import requests
from PIL import Image, ImageDraw

from tests.utils import RemoteOpenAIServer

MODEL = os.environ.get("VLLM_BOX_TEST_MODEL", "Qwen/Qwen3.5-0.8B")
SERVED_MODEL = os.environ.get("VLLM_BOX_TEST_SERVED_MODEL", "box-native-e2e")


# The server subprocess owns accelerator state and shuts down on exit.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(scope="module")
def server():
    args = [
        "--host",
        "127.0.0.1",
        "--served-model-name",
        SERVED_MODEL,
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        '{"image": 1}',
    ]
    port = os.environ.get("VLLM_BOX_TEST_PORT")
    if port:
        args.extend(["--port", str(int(port))])
    with RemoteOpenAIServer(
        MODEL, args, auto_port=not bool(port), max_wait_seconds=900
    ) as remote:
        yield remote


@pytest.fixture(scope="module")
def image():
    picture = Image.new("RGB", (384, 384), "white")
    ImageDraw.Draw(picture).rectangle((80, 90, 290, 300), fill="red")
    output = BytesIO()
    picture.save(output, format="PNG")
    return "data:image/png;base64," + pybase64.b64encode(output.getvalue()).decode()


def test_generated_box_and_label_decisions_share_the_served_model(server, image):
    """Exercise the public multimodal path and retain ordinary diagnostics."""
    questions = {
        "present": {"type": "noul", "instructions": "Is a red rectangle visible?"},
        "target": {"type": "box", "instructions": "the red rectangle"},
        "color": {
            "type": "choice",
            "instructions": "What color is the rectangle?",
            "criteria": {"red": None, "blue": None},
        },
    }
    response = requests.post(
        server.url_for("v1/systemone"),
        json={
            "model": SERVED_MODEL,
            "state": "Inspect the attached image.",
            "images": [image],
            "questions": questions,
        },
        timeout=120,
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["model"] == SERVED_MODEL
    assert list(result["answers"]) == list(questions)
    target = result["answers"]["target"]
    assert target["type"] == "box"
    assert target["confidence"] is None
    box = target["box"]
    assert 0 <= box["left"] < box["right"] <= 1
    assert 0 <= box["top"] < box["bottom"] <= 1
    assert set(result["diagnostics"]) == {"present", "color"}
    assert sum(result["answers"]["color"]["probabilities"].values()) == pytest.approx(1)
    assert result["usage"]["output_tokens"] > 2
    assert result["usage"]["input_tokens"] > 0


@pytest.mark.parametrize(
    "override",
    [
        {"images": []},
        {"joint": True},
        {"questions": {"target": {"type": "box", "instructions": ""}}},
        {
            "questions": {
                "target": {"type": "box", "instructions": "rectangle", "criteria": []}
            }
        },
        {
            "questions": {
                str(i): {"type": "box", "instructions": "rectangle"} for i in range(5)
            }
        },
    ],
)
def test_invalid_box_requests_return_client_errors(server, image, override):
    response = requests.post(
        server.url_for("v1/systemone"),
        json={
            "model": SERVED_MODEL,
            "state": "",
            "images": [image],
            "questions": {"target": {"type": "box", "instructions": "red rectangle"}},
            **override,
        },
        timeout=120,
    )
    assert response.status_code == 400, response.text


def test_text_decisions_work_after_image_requests(server):
    """Retain ordinary question behavior and one output token per question."""
    questions = {
        "team": {
            "type": "choice",
            "instructions": "Which team handles this connection error?",
            "criteria": {
                "billing": "charges, invoices and billing disputes",
                "technical": "integration, connectivity and software errors",
            },
        },
        "urgent": {
            "type": "noul",
            "instructions": "The customer explicitly needs a reply today.",
        },
        "tone": {
            "type": "score",
            "instructions": "How upset is the customer?",
            "criteria": ["calm", "annoyed", "furious"],
        },
    }
    response = requests.post(
        server.url_for("v1/systemone"),
        json={
            "model": SERVED_MODEL,
            "state": (
                "The customer cannot connect the payment integration. "
                "They need a reply today and say they are furious."
            ),
            "questions": questions,
        },
        timeout=120,
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["answers"]["team"]["choice"] == "technical"
    assert 0 <= result["answers"]["urgent"]["noul"] <= 1
    assert 0 <= result["answers"]["tone"]["score"] <= 2
    assert set(result["diagnostics"]) == set(questions)
    assert result["usage"]["output_tokens"] == 3
