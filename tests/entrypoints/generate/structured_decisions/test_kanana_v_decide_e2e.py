# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""/v1/systemone on kakaocorp/kanana-2-v-decide-3b-2611 against the
checkpoint's own runtime (`trust_remote_code`), text and image."""

import pytest
import requests
from transformers import AutoModel

from tests.utils import RemoteOpenAIServer
from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import encode_image_url

MODEL = "kakaocorp/kanana-2-v-decide-3b-2611"
REVISION = "12820d654ed9789a0f7f48daeee1703c3483c464"
ATOL = 0.02
TEXT = {
    "state": (
        "The order arrived damaged yesterday. The customer has a receipt and "
        "asks for a replacement today."
    ),
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Refunds, replacements and damaged deliveries",
                "billing": "Payments, invoices and charges",
                "technical": "Product setup and faults",
            },
        },
        "receipt": {
            "type": "noul",
            "instructions": "Does the customer have a receipt?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["Routine", "Soon", "Today"],
        },
    },
}
IMAGE_QUESTIONS = {
    "sign": {
        "type": "choice",
        "instructions": "Which traffic sign is in the image?",
        "criteria": {"stop": None, "yield": None, "speed limit": None},
    },
    "has_sign": {"type": "noul", "instructions": "Is there a traffic sign?"},
}


@pytest.fixture(scope="module")
def bodies():
    image = encode_image_url(ImageAsset("stop_sign").pil_image)
    return {
        "text": TEXT,
        "image": {
            "state": "Describe the attached image.",
            "questions": IMAGE_QUESTIONS,
            "images": [image],
        },
    }


@pytest.fixture(scope="module")
def reference(hf_runner, bodies):
    with hf_runner(
        MODEL,
        dtype="bfloat16",
        auto_cls=AutoModel,
        trust_remote_code=True,
        revision=REVISION,
    ) as hf_model:
        return {
            name: hf_model.model.system_one(**body)["answers"]
            for name, body in bodies.items()
        }


@pytest.fixture(scope="module")
def server(reference):
    args = [
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.6",
        "--revision",
        REVISION,
        "--trust-remote-code",
    ]
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


def systemone(server, body):
    return requests.post(server.url_for("v1/systemone"), json={"model": MODEL, **body})


@pytest.mark.parametrize("name", ["text", "image"])
def test_systemone_matches_the_checkpoint_runtime(server, reference, bodies, name):
    response = systemone(server, bodies[name])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "structured_decision"
    assert body["usage"]["output_tokens"] == 0
    assert body["usage"]["input_tokens"] > 0
    answers = body["answers"]
    assert list(answers) == list(reference[name])
    for qid, expected in reference[name].items():
        answer = answers[qid]
        if expected["type"] == "noul":
            assert answer["noul"] == pytest.approx(expected["noul"], abs=ATOL)
            yes, no = answer["probabilities"]["yes"], answer["probabilities"]["no"]
            assert yes + no == pytest.approx(1.0)
            continue
        assert answer["probabilities"] == pytest.approx(
            expected["probabilities"], abs=ATOL
        )
        if expected["type"] == "choice":
            assert answer["choice"] == expected["choice"]

    if name == "image":
        again = systemone(server, bodies[name])
        assert again.status_code == 200, again.text
        for qid, answer in again.json()["answers"].items():
            assert answer["probabilities"] == pytest.approx(
                answers[qid]["probabilities"], abs=1e-2
            )


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"instructions": "Be strict."}, "takes no instructions"),
        ({"videos": ["data:video/mp4;base64,AAAA"]}, "does not take video inputs"),
        ({"images": ["data:image/png;base64,AAAA"] * 11}, "at most 10"),
    ],
)
def test_systemone_refusals(server, extra, match):
    response = systemone(server, {**TEXT, **extra})
    assert response.status_code == 400
    assert match in response.json()["error"]["message"]
