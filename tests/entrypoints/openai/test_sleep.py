# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import requests
from prometheus_client.parser import text_string_to_metric_families

from ...utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B"


def test_sleep_mode():
    # dtype, max-len etc set so that this can run in CI
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "128",
        "--enable-sleep-mode",
    ]

    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
        env_dict={"VLLM_SERVER_DEV_MODE": "1", "CUDA_VISIBLE_DEVICES": "0"},
    ) as remote_server:
        response = requests.post(remote_server.url_for("sleep"), params={"level": "1"})
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is True

        # check sleep metrics
        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        awake, weights_offloaded, discard_all = _get_sleep_metrics_from_api(response)
        assert awake == 0
        assert weights_offloaded == 1
        assert discard_all == 0

        response = requests.post(remote_server.url_for("wake_up"))
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is False

        # check sleep metrics
        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        awake, weights_offloaded, discard_all = _get_sleep_metrics_from_api(response)
        assert awake == 1
        assert weights_offloaded == 0
        assert discard_all == 0

        # test wake up with tags
        response = requests.post(remote_server.url_for("sleep"), params={"level": "1"})
        assert response.status_code == 200

        response = requests.post(
            remote_server.url_for("wake_up"), params={"tags": ["weights"]}
        )
        assert response.status_code == 200

        # is sleeping should be false after waking up any part of the engine
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is True

        response = requests.post(
            remote_server.url_for("wake_up"), params={"tags": ["kv_cache"]}
        )
        assert response.status_code == 200

        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is False

        # check sleep metrics
        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        awake, weights_offloaded, discard_all = _get_sleep_metrics_from_api(response)
        assert awake == 1
        assert weights_offloaded == 0
        assert discard_all == 0


def _get_sleep_metrics_from_api(response: requests.Response):
    """Return (awake, weights_offloaded, discard_all)"""

    awake, weights_offloaded, discard_all = None, None, None

    for family in text_string_to_metric_families(response.text):
        if family.name == "vllm:engine_sleep_state":
            for sample in family.samples:
                if sample.name == "vllm:engine_sleep_state":
                    for label_name, label_value in sample.labels.items():
                        if label_value == "awake":
                            awake = sample.value
                        elif label_value == "weights_offloaded":
                            weights_offloaded = sample.value
                        elif label_value == "discard_all":
                            discard_all = sample.value

    assert awake is not None
    assert weights_offloaded is not None
    assert discard_all is not None

    return awake, weights_offloaded, discard_all
