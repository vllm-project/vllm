# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import requests
from prometheus_client.parser import text_string_to_metric_families

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B"


def test_pause_state_metric():
    # dtype, max-len etc set so that this can run in CI
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "128",
    ]

    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
        env_dict={"VLLM_SERVER_DEV_MODE": "1", "CUDA_VISIBLE_DEVICES": "0"},
    ) as remote_server:
        # "keep" mode freezes all requests -> paused_all.
        response = requests.post(
            remote_server.url_for("pause"), params={"mode": "keep"}
        )
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_paused"))
        assert response.status_code == 200
        assert response.json().get("is_paused") is True

        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        unpaused, paused_new, paused_all = _get_pause_metrics_from_api(response)
        assert unpaused == 0
        assert paused_new == 0
        assert paused_all == 1

        response = requests.post(remote_server.url_for("resume"))
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_paused"))
        assert response.status_code == 200
        assert response.json().get("is_paused") is False

        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        unpaused, paused_new, paused_all = _get_pause_metrics_from_api(response)
        assert unpaused == 1
        assert paused_new == 0
        assert paused_all == 0

        # "abort" mode only blocks new requests -> paused_new.
        response = requests.post(
            remote_server.url_for("pause"), params={"mode": "abort"}
        )
        assert response.status_code == 200

        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        unpaused, paused_new, paused_all = _get_pause_metrics_from_api(response)
        assert unpaused == 0
        assert paused_new == 1
        assert paused_all == 0


def _get_pause_metrics_from_api(response: requests.Response):
    """Return (unpaused, paused_new, paused_all)"""

    unpaused, paused_new, paused_all = None, None, None

    for family in text_string_to_metric_families(response.text):
        if family.name == "vllm:engine_pause_state":
            for sample in family.samples:
                if sample.name == "vllm:engine_pause_state":
                    for label_name, label_value in sample.labels.items():
                        if label_value == "unpaused":
                            unpaused = sample.value
                        elif label_value == "paused_new":
                            paused_new = sample.value
                        elif label_value == "paused_all":
                            paused_all = sample.value

    assert unpaused is not None
    assert paused_new is not None
    assert paused_all is not None

    return unpaused, paused_new, paused_all
