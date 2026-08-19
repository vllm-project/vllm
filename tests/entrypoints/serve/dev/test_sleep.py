# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
        assert _get_sleep_states(response) == ("paused", "offloaded", "discarded")

        response = requests.post(remote_server.url_for("wake_up"))
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is False

        # check sleep metrics
        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        assert _get_sleep_states(response) == ("running", "resident", "resident")

        response = requests.post(remote_server.url_for("release_kv_cache_memory"))
        assert response.status_code == 200

        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        assert _get_sleep_states(response) == ("paused", "resident", "discarded")

        response = requests.post(
            remote_server.url_for("wake_up"), params={"tags": ["kv_cache"]}
        )
        assert response.status_code == 200

        # test wake up with tags
        response = requests.post(remote_server.url_for("sleep"), params={"level": "1"})
        assert response.status_code == 200

        response = requests.post(
            remote_server.url_for("wake_up"), params={"tags": ["weights"]}
        )
        assert response.status_code == 200

        # Partial wake keeps the engine sleeping.
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is True

        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        assert _get_sleep_states(response) == ("paused", "resident", "discarded")

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
        assert _get_sleep_states(response) == ("running", "resident", "resident")


def _get_sleep_states(response: requests.Response):
    """Return scheduler, weights, and KV cache states."""
    values = {}

    for family in text_string_to_metric_families(response.text):
        if family.name == "vllm:engine_sleep_component_state":
            for sample in family.samples:
                key = (sample.labels["component"], sample.labels["state"])
                values[key] = sample.value

    assert len(values) == 7
    active = {component: state for (component, state), value in values.items() if value}
    return active["scheduler"], active["weights"], active["kv_cache"]
