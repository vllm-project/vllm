# SPDX-License-Identifier: Apache-2.0

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
        # Test basic sleep/wake functionality
        response = requests.post(remote_server.url_for("sleep"),
                                 params={"level": "1"})
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is True

        # Test model-dependent endpoints return errors when model is sleeping
        # Test completions endpoint
        completion_payload = {
            "model": MODEL_NAME,
            "prompt": "Hello, world",
            "max_tokens": 5
        }
        response = requests.post(remote_server.url_for("v1/completions"),
                                 json=completion_payload)
        assert response.status_code == 503
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["type"] == "ModelSleepingError"
        assert "sleep mode" in error_data["error"]["message"].lower()

        # Test chat completions endpoint
        chat_payload = {
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": "Hello"
            }],
            "max_tokens": 5
        }
        response = requests.post(remote_server.url_for("v1/chat/completions"),
                                 json=chat_payload)
        assert response.status_code == 503
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["type"] == "ModelSleepingError"

        # Test non-model endpoints still work when model is sleeping
        # Models endpoint should work
        response = requests.get(remote_server.url_for("v1/models"))
        assert response.status_code == 200

        # Health check should work
        response = requests.get(remote_server.url_for("health"))
        assert response.status_code == 200

        # Wake up and verify model-dependent endpoints now work
        response = requests.post(remote_server.url_for("wake_up"))
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is False

        # Verify completions endpoint works after waking up
        response = requests.post(remote_server.url_for("v1/completions"),
                                 json=completion_payload)
        assert response.status_code == 200

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
