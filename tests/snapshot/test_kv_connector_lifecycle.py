# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from vllm.snapshot.kv_connector_lifecycle import (
    refresh_scheduler_handshake_metadata_after_snapshot_restore,
    rotate_engine_id,
    snapshot_scheduler_runtime,
)


@pytest.mark.parametrize(
    "name,extra,enabled,expected",
    [
        ("AscendStoreConnector", {}, True, True),
        ("AscendStoreConnector", {"backend": "memcache"}, True, False),
        ("AscendStoreConnector", {"backend": "yuanrong"}, True, False),
        ("AscendStoreConnector", {}, False, False),
        ("MooncakeConnectorV1", {}, True, False),
        ("MooncakeConnectorV2", {}, True, False),
        ("MooncakePullConnector", {}, True, False),
        (
            "AscendMultiConnector",
            {"connectors": [{"kv_connector": "AscendStoreConnector"}] * 2},
            True,
            True,
        ),
    ],
)
def test_scheduler_runtime_only_snapshots_mooncake_pool(name, extra, enabled, expected):
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            snapshot_config=object() if enabled else None,
            kv_transfer_config=SimpleNamespace(
                kv_connector=name, kv_connector_extra_config=extra
            ),
        )
    )
    runtime = Mock()
    for step in ("Lock", "Backup", "Restore", "Unlock"):
        getattr(runtime, f"aclrtSnapShotProcess{step}").return_value = 0
    with patch(
        "vllm.snapshot.kv_connector_lifecycle.CDLL", return_value=runtime
    ) as load:
        for operation in ("suspend", "device_unlock", "resume"):
            snapshot_scheduler_runtime(engine, operation)
    if expected:
        assert runtime.mock_calls == [
            getattr(call, f"aclrtSnapShotProcess{step}")(os.getpid(), None)
            for step in ("Lock", "Backup", "Unlock", "Restore", "Unlock")
        ]
    else:
        load.assert_not_called()


def test_scheduler_runtime_stops_on_restore_error():
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            snapshot_config=object(),
            kv_transfer_config=SimpleNamespace(
                kv_connector="AscendStoreConnector", kv_connector_extra_config={}
            ),
        )
    )
    runtime = Mock()
    runtime.aclrtSnapShotProcessRestore.return_value = 507899
    with patch("vllm.snapshot.kv_connector_lifecycle.CDLL", return_value=runtime):
        with pytest.raises(RuntimeError, match="Restore status=507899"):
            snapshot_scheduler_runtime(engine, "resume")
    runtime.aclrtSnapShotProcessUnlock.assert_not_called()


def test_rotate_engine_id_preserves_instance_and_dp_rank():
    engine_id = "instance-0123456789abcdef0123456789abcdef_dp3"

    with patch("vllm.snapshot.kv_connector_lifecycle.uuid4") as uuid4:
        uuid4.return_value.hex = "fedcba9876543210fedcba9876543210"
        result = rotate_engine_id(engine_id)

    assert result == "instance-fedcba9876543210fedcba9876543210_dp3"


def test_refresh_scheduler_replaces_worker_handshake_metadata():
    kv_connector = Mock()
    scheduler = SimpleNamespace(get_kv_connector=Mock(return_value=kv_connector))
    model_executor = SimpleNamespace(
        get_kv_connector_handshake_metadata=Mock(
            return_value=[{(0, 0): "rank-0"}, None, {(0, 1): "rank-1"}]
        )
    )
    engine_core = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                is_kv_producer=True,
                kv_connector="MooncakeConnectorV1",
            )
        ),
        scheduler=scheduler,
        model_executor=model_executor,
    )

    refresh_scheduler_handshake_metadata_after_snapshot_restore(engine_core)

    kv_connector.set_xfer_handshake_metadata_pp_aware.assert_called_once_with(
        {(0, 0): "rank-0", (0, 1): "rank-1"}
    )
