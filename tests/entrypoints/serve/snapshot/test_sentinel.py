# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import Mock, patch

from vllm.entrypoints.serve.snapshot.sentinel import (
    DEVICE_UNLOCK_TIMEOUT,
    RESUME_TIMEOUT,
    SUSPEND_TIMEOUT,
    SnapshotSentinel,
)


def _sentinel(metadata_path: str) -> SnapshotSentinel:
    return SnapshotSentinel(
        snapshot_metadata=metadata_path,
        port=8000,
        use_tls=False,
        ca_file=None,
    )


def test_suspend_uses_snapshot_metadata(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"model_save_path": "/snapshot/weights"}),
        encoding="utf-8",
    )
    sentinel = _sentinel(str(metadata_path))

    with (
        patch.object(sentinel, "_request") as request,
        patch(
            "vllm.entrypoints.serve.snapshot.sentinel.get_local_ip",
            return_value="10.0.0.1",
        ) as get_local_ip,
    ):
        sentinel._call_suspend()

    get_local_ip.assert_called_once_with()
    request.assert_called_once_with(
        "POST",
        "/suspend",
        SUSPEND_TIMEOUT,
        "10.0.0.1",
        {"model_save_path": "/snapshot/weights"},
    )


def test_checkpoint_unlocks_device_and_stops_on_cold_start(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"checkpoint": "done"}), encoding="utf-8")
    sentinel = _sentinel(str(metadata_path))

    with (
        patch.object(sentinel, "_request") as request,
        patch(
            "vllm.entrypoints.serve.snapshot.sentinel.is_restore",
            return_value=False,
        ),
        patch(
            "vllm.entrypoints.serve.snapshot.sentinel.get_local_ip",
            return_value="10.0.0.1",
        ),
    ):
        sentinel._reach_checkpoint()

    request.assert_called_once_with(
        "POST", "/device_unlock", DEVICE_UNLOCK_TIMEOUT, "10.0.0.1"
    )
    assert sentinel._stop_event.is_set()


def test_resume_uses_snapshot_metadata(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_load_path": "/snapshot/weights",
                "data_parallel_master_ip": "10.0.0.1",
            }
        ),
        encoding="utf-8",
    )
    sentinel = _sentinel(str(metadata_path))

    with (
        patch.object(sentinel, "_request") as request,
        patch(
            "vllm.entrypoints.serve.snapshot.sentinel.get_local_ip",
            return_value="10.0.0.2",
        ) as get_local_ip,
    ):
        sentinel._call_resume()

    get_local_ip.assert_called_once_with()
    request.assert_called_once_with(
        "POST",
        "/resume",
        RESUME_TIMEOUT,
        "10.0.0.2",
        {
            "model_path": "/snapshot/weights",
            "data_parallel_master_ip": "10.0.0.1",
        },
    )


def test_request_uses_provided_host():
    sentinel = _sentinel("/snapshot/metadata.json")
    response = Mock()

    with patch(
        "vllm.entrypoints.serve.snapshot.sentinel.requests.request",
        return_value=response,
    ) as request:
        sentinel._request("POST", "/resume", RESUME_TIMEOUT, "10.0.0.2")

    request.assert_called_once_with(
        "POST",
        "http://10.0.0.2:8000/resume",
        params=None,
        timeout=RESUME_TIMEOUT,
        verify=True,
    )
