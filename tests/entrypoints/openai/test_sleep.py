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

    with RemoteOpenAIServer(MODEL_NAME,
                            args,
                            env_dict={
                                "VLLM_SERVER_DEV_MODE": "1",
                                "CUDA_VISIBLE_DEVICES": "0"
                            }) as remote_server:
        response = requests.post(remote_server.url_for("sleep"),
                                 params={"level": "1"})
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is True

        # check sleep metrics
        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        sleep_state, sleep_level = (_get_sleep_metrics_from_api(response))
        assert sleep_state == 1
        assert sleep_level == 1

        response = requests.post(remote_server.url_for("wake_up"))
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is False

        # check sleep metrics
        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        sleep_state, sleep_level = (_get_sleep_metrics_from_api(response))
        assert sleep_state == 0
        assert sleep_level == 0

        # test wake up with tags
        response = requests.post(remote_server.url_for("sleep"),
                                 params={"level": "1"})
        assert response.status_code == 200

        response = requests.post(remote_server.url_for("wake_up"),
                                 params={"tags": ["weights"]})
        assert response.status_code == 200

        # is sleeping should be false after waking up any part of the engine
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is True

        response = requests.post(remote_server.url_for("wake_up"),
                                 params={"tags": ["kv_cache"]})
        assert response.status_code == 200

        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is False

        # check sleep metrics
        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        sleep_state, sleep_level = (_get_sleep_metrics_from_api(response))
        assert sleep_state == 0
        assert sleep_level == 0


def _get_sleep_metrics_from_api(response: requests.Response):
    """Return (sleep_state, sleep_level)"""

    sleep_state, sleep_level = None, None

    for family in text_string_to_metric_families(response.text):
        if family.name == "vllm:engine_sleep_state":
            for sample in family.samples:
                if sample.name == "vllm:engine_sleep_state":
                    sleep_state = sample.value
                    break
        elif family.name == "vllm:engine_sleep_level":
            for sample in family.samples:
                if sample.name == "vllm:engine_sleep_level":
                    sleep_level = sample.value
                    break

    assert sleep_state is not None
    assert sleep_level is not None

    return sleep_state, sleep_level
