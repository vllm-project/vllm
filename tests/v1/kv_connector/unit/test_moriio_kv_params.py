# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MoRI-IO kv_transfer_params integer coercion.

Kept outside test_moriio_connector.py so they run without ROCm/mori.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOConnectorMetadata,
    MoRIIOMode,
    coerce_kv_int_param,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
)


def test_coerce_kv_int_param_absent_and_valid():
    assert coerce_kv_int_param({}, "remote_dp_size", 1) == 1
    assert coerce_kv_int_param({"remote_dp_size": 4}, "remote_dp_size", 1) == 4
    assert coerce_kv_int_param({"remote_dp_size": "8"}, "remote_dp_size", 1) == 8
    assert coerce_kv_int_param({"remote_dp_size": None}, "remote_dp_size", 1) == 1


def test_coerce_kv_int_param_rejects_non_integer():
    with pytest.raises(ValueError, match="remote_dp_size"):
        coerce_kv_int_param({"remote_dp_size": "x"}, "remote_dp_size", 1)
    with pytest.raises(ValueError, match="remote_dp_size"):
        coerce_kv_int_param({"remote_dp_size": ["bad"]}, "remote_dp_size", 1)


def test_write_decode_update_state_rejects_non_numeric_remote_dp_size():
    """Malformed remote_dp_size must not raise into EngineCore.

    A client can set do_remote_prefill with remote_dp_size=\"x\". The WRITE
    decode notify path must refuse KV transfer and clear the flag instead of
    letting bare int() terminate the process.
    """
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.map_request_id = MagicMock()
    scheduler._reqs_need_save = {}
    scheduler._req_kv_params = {}

    request = MagicMock()
    request.request_id = "req-bad-dp-size"
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "remote_dp_size": "x",
        "transfer_id": "xfer-1",
    }
    blocks = MagicMock()

    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=16)

    assert request.kv_transfer_params["do_remote_prefill"] is False
    scheduler.map_request_id.assert_called_once()


def test_add_new_req_skips_non_numeric_remote_dp_size():
    """Metadata registration must skip malformed remote_dp_size without raising."""
    meta = MoRIIOConnectorMetadata()
    zmq_addr = "host:10.0.0.1,handshake:5600,notify:5601"
    request_id = (
        f"___prefill_addr_{zmq_addr}___decode_addr_{zmq_addr}_{uuid.uuid4().hex}"
    )

    meta.add_new_req(
        request_id=request_id,
        local_block_ids=[1, 2, 3],
        kv_transfer_params={
            "transfer_id": "xfer-bad-dp",
            "remote_engine_id": "engine-a",
            "remote_block_ids": [1, 2, 3],
            "remote_dp_size": "x",
        },
        write_mode=False,
    )

    assert meta.reqs_to_recv == {}
    assert meta.reqs_to_save == {}
