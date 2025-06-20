# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import requests

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

        response = requests.post(remote_server.url_for("wake_up"))
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is False

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
