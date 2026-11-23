# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import requests

from tests.utils import RemoteOpenAIServer


def test_clef_probabilities_with_raw_pooling_default():
    model = "Cloudflare/clef-flash"
    args = [
        "--pooler-config",
        '{"use_activation": false}',
        "--load-format",
        "dummy",
        "--language-model-only",
        "--enforce-eager",
        "--max-model-len",
        "256",
        "--max-num-seqs",
        "1",
    ]
    with RemoteOpenAIServer(
        model,
        args,
        override_hf_configs={
            "text_config": {
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "full_attention"],
            }
        },
    ) as server:
        response = requests.post(
            server.url_for("v1/systemone"),
            json={
                "model": model,
                "state": "I want a refund.",
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {"returns": None, "sales": None},
                    }
                },
            },
        )
    assert response.status_code == 200, response.text
    probs = response.json()["answers"]["team"]["probabilities"]
    assert set(probs) == {"returns", "sales"}
    assert all(0 <= p <= 1 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0)
