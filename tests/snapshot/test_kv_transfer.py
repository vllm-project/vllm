# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm.snapshot.kv_transfer import (
    refresh_scheduler_after_resume,
    refresh_scheduler_handshake_metadata_after_resume,
    rotate_engine_id,
)


def test_rotate_engine_id_preserves_instance_and_dp_rank():
    engine_id = "instance-0123456789abcdef0123456789abcdef_dp3"

    with patch("vllm.snapshot.kv_transfer.uuid4") as uuid4:
        uuid4.return_value.hex = "fedcba9876543210fedcba9876543210"
        result = rotate_engine_id(engine_id)

    assert result == "instance-fedcba9876543210fedcba9876543210_dp3"


def test_refresh_scheduler_updates_kv_identity_and_host():
    connector_scheduler = SimpleNamespace(
        side_channel_host="10.0.0.1",
        engine_id="instance-0123456789abcdef0123456789abcdef_dp0",
    )
    connector = SimpleNamespace(
        connector_scheduler=connector_scheduler,
        engine_id=connector_scheduler.engine_id,
    )
    kv_config = SimpleNamespace(
        is_kv_producer=True,
        is_kv_consumer=False,
        engine_id=connector_scheduler.engine_id,
    )
    engine_core = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=kv_config),
        scheduler=SimpleNamespace(connector=connector),
    )

    with patch(
        "vllm.snapshot.kv_transfer.rotate_engine_id",
        return_value="instance-new",
    ):
        refresh_scheduler_after_resume(engine_core, "10.0.0.2")

    assert connector_scheduler.side_channel_host == "10.0.0.2"
    assert connector_scheduler.engine_id == "instance-new"
    assert connector.engine_id == "instance-new"
    assert kv_config.engine_id == "instance-new"


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

    refresh_scheduler_handshake_metadata_after_resume(engine_core)

    kv_connector.set_xfer_handshake_metadata_pp_aware.assert_called_once_with(
        {(0, 0): "rank-0", (0, 1): "rank-1"}
    )
