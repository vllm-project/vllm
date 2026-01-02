# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time

import requests
from prometheus_client.parser import text_string_to_metric_families

from tests.utils import RemoteOpenAIServer

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


def test_sleep_with_active_requests():
    """Test that sleep endpoint returns error when requests are active."""
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
        # Start a long-running completion in a background thread
        completion_result = []
        
        def make_completion():
            try:
                response = requests.post(
                    remote_server.url_for("v1/completions"),
                    json={
                        "model": MODEL_NAME,
                        "prompt": "Write a long story about",
                        "max_tokens": 200,
                        "temperature": 0.0,
                    },
                    timeout=30,
                )
                completion_result.append(response)
            except Exception as e:
                completion_result.append(e)
        
        completion_thread = threading.Thread(target=make_completion)
        completion_thread.start()
        
        # Poll metrics endpoint to wait for request to be actually running
        # This is more reliable than time.sleep
        max_wait = 5.0  # 5 seconds timeout
        start_time = time.time()
        running_requests = 0
        while running_requests == 0:
            if time.time() - start_time > max_wait:
                raise TimeoutError("Request never started processing")
            try:
                metrics_response = requests.get(remote_server.url_for("metrics"))
                if metrics_response.status_code == 200:
                    # Parse metrics to check for running requests
                    from prometheus_client.parser import text_string_to_metric_families
                    for family in text_string_to_metric_families(metrics_response.text):
                        if family.name == "vllm:num_requests_running":
                            for sample in family.samples:
                                running_requests = int(sample.value)
                                break
            except Exception:
                pass
            time.sleep(0.05)
        
        # Try to sleep while request is active - should return an error
        response = requests.post(remote_server.url_for("sleep"), params={"level": "1"})
        assert response.status_code == 500  # Internal Server Error
        assert "Cannot put engine to sleep while requests are being processed" in response.text
        
        # Wait for completion to finish
        completion_thread.join(timeout=30)
        
        # Verify the completion succeeded
        assert len(completion_result) == 1
        assert isinstance(completion_result[0], requests.Response)
        assert completion_result[0].status_code == 200
        
        # Now sleep should work since no requests are active
        response = requests.post(remote_server.url_for("sleep"), params={"level": "1"})
        assert response.status_code == 200
        
        # Verify the engine is sleeping
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is True
        
        # Wake up for cleanup
        response = requests.post(remote_server.url_for("wake_up"))
        assert response.status_code == 200
