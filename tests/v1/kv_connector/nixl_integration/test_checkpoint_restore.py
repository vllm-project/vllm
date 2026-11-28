# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P/D with NixlConnector across process checkpoint prepare and restore.

Both servers start with --checkpoint-prepare-on-startup, so they come up
prepared, without a NIXL agent. Each restore gives the engine a new KV engine
id, which its peer must handshake with again.
"""

import json
import time

import requests

from tests.models.utils import check_logprobs_close
from tests.utils import RemoteOpenAIServer
from vllm.utils.network_utils import get_open_port

MODEL = "Qwen/Qwen3-0.6B"
# Longer than a KV block, so each P/D request transfers KV.
PROMPT = "Count up by one: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,"


def _server(gpu: str) -> RemoteOpenAIServer:
    kv_transfer_config = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
    return RemoteOpenAIServer(
        MODEL,
        [
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.2",
            "--max-model-len",
            "8192",
            "--enable-sleep-mode",
            "--checkpoint-prepare-on-startup",
            "--kv-transfer-config",
            json.dumps(kv_transfer_config),
        ],
        env_dict={
            "CUDA_VISIBLE_DEVICES": gpu,
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(get_open_port()),
        },
        max_wait_seconds=1200,
    )


def _post(url: str, body: dict | None = None) -> requests.Response:
    response = requests.post(url, json=body, timeout=300)
    response.raise_for_status()
    return response


def _wait_for_state(server: RemoteOpenAIServer, state: str) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        url = server.url_for("checkpoint", "status")
        current = requests.get(url, timeout=30).json()
        assert current["state"] != "failed", current
        if current["state"] == state:
            return
        time.sleep(0.5)
    raise TimeoutError(f"{server.url_root} did not reach {state}")


def _metric(server: RemoteOpenAIServer, name: str) -> float:
    text = requests.get(server.url_for("metrics"), timeout=30).text
    return sum(
        float(line.split()[-1])
        for line in text.splitlines()
        if line.split("{")[0].split(" ")[0] == name
    )


def _output(response: dict) -> tuple[list[str], str, list[dict[str, float]]]:
    choice = response["choices"][0]
    logprobs = choice["logprobs"]
    return logprobs["tokens"], choice["text"], logprobs["top_logprobs"]


def _check_pd(
    prefill: RemoteOpenAIServer, decode: RemoteOpenAIServer, phase: str
) -> str:
    """Send P/D requests and check them against prefill alone. Returns the
    prefill's engine id."""
    transfers = _metric(decode, "vllm:nixl_bytes_transferred_count")
    failed = _metric(decode, "vllm:nixl_num_failed_transfers_total")
    engine_ids = set()
    expected_outputs, pd_outputs = [], []
    for i in range(3):
        # A new prefix per request, so the decode cannot reuse cached KV.
        body = {
            "model": MODEL,
            "prompt": f"{phase} {i}. {PROMPT}",
            "temperature": 0,
            "max_tokens": 16,
            "logprobs": 5,
        }
        prefill_url = prefill.url_for("v1", "completions")
        expected = _post(prefill_url, body).json()
        prefilled = _post(
            prefill_url,
            {
                **body,
                "max_tokens": 1,
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_engine_id": None,
                    "remote_block_ids": None,
                    "remote_host": None,
                    "remote_port": None,
                },
            },
        ).json()
        params = prefilled["kv_transfer_params"]
        engine_ids.add(params["remote_engine_id"])
        decoded = _post(
            decode.url_for("v1", "completions"),
            {**body, "kv_transfer_params": params},
        ).json()
        expected_outputs.append(_output(expected))
        pd_outputs.append(_output(decoded))
    check_logprobs_close(
        outputs_0_lst=expected_outputs,
        outputs_1_lst=pd_outputs,
        name_0="prefill",
        name_1="p/d",
    )
    assert _metric(decode, "vllm:nixl_bytes_transferred_count") > transfers
    assert _metric(decode, "vllm:nixl_num_failed_transfers_total") == failed
    (engine_id,) = engine_ids
    return engine_id


def _restore(server: RemoteOpenAIServer) -> None:
    _post(server.url_for("checkpoint", "restore"))
    _wait_for_state(server, "running")


def test_checkpoint_restore():
    servers: list[RemoteOpenAIServer] = []
    try:
        servers.append(_server("0"))
        servers.append(_server("1"))
        prefill, decode = servers
        for server in servers:
            _wait_for_state(server, "prepared")
            _restore(server)

        # The first restore creates each NIXL agent.
        engine_id = _check_pd(prefill, decode, "first restore")

        # In place, the prefill replaces its agent and engine id, and the
        # decode handshakes with it again.
        _post(prefill.url_for("checkpoint", "prepare"))
        _wait_for_state(prefill, "prepared")
        _restore(prefill)
        restored_engine_id = _check_pd(prefill, decode, "prefill restored")
        assert restored_engine_id != engine_id

        # The decode replaces its agent and handshakes with the prefill again.
        _post(decode.url_for("checkpoint", "prepare"))
        _wait_for_state(decode, "prepared")
        _restore(decode)
        assert _check_pd(prefill, decode, "decode restored") == restored_engine_id
    finally:
        # Each server's own shutdown would wait on the other's GPU memory.
        RemoteOpenAIServer.shutdown_many(servers)
