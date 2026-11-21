# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generative scoring on a server started with --logprobs-mode raw_logits."""

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
        "512",
        "--enforce-eager",
        "--max-num-seqs",
        "32",
        "--logprobs-mode",
        "raw_logits",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.parametrize("apply_softmax", [True, False])
def test_apply_softmax_in_logits_mode(server: RemoteOpenAIServer, apply_softmax):
    """exp(logit) is not a probability, so apply_softmax=false is refused."""
    response = requests.post(
        server.url_for("generative_scoring"),
        json={
            "model": MODEL_NAME,
            "query": "Is Paris the capital of France? Answer Yes or No: ",
            "items": ["Paris is beautiful."],
            "label_token_ids": [9454, 2753],
            "apply_softmax": apply_softmax,
        },
    )
    if apply_softmax:
        assert response.status_code == 200, response.text
        assert 0.0 <= response.json()["data"][0]["score"] <= 1.0
    else:
        assert response.status_code == 400, response.text
        assert "--logprobs-mode" in response.json()["error"]["message"]
