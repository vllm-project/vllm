# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import torch

from .utils import create_request


if not hasattr(torch, "npu"):
    torch.npu = types.SimpleNamespace(Event=type("FakeNPUEvent", (), {}))  # type: ignore[attr-defined]

if "mooncake.engine" not in sys.modules:
    fake_mooncake = sys.modules.setdefault("mooncake", types.ModuleType("mooncake"))
    fake_engine = types.ModuleType("mooncake.engine")
    fake_engine.TransferEngine = type("FakeTransferEngine", (), {})
    fake_mooncake.engine = fake_engine
    sys.modules["mooncake.engine"] = fake_engine

mooncake_connector = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector"
)
MooncakeConnectorScheduler = mooncake_connector.MooncakeConnectorScheduler


def _make_scheduler() -> MooncakeConnectorScheduler:
    scheduler = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    scheduler._handle_request_executor = None
    scheduler._handle_request_futures = set()
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_send = {}
    scheduler.is_kv_producer = False
    scheduler.is_kv_consumer = True
    scheduler.use_layerwise = True
    return scheduler


def test_submit_handle_request_uses_thread_pool_and_copies_args():
    scheduler = _make_scheduler()
    scheduler._run_handle_request_task = MagicMock()
    scheduler._log_handle_request_result = MagicMock()

    mock_executor = MagicMock()
    mock_future = MagicMock()
    mock_executor.submit.return_value = mock_future
    scheduler._handle_request_executor = mock_executor

    params = {"remote_engine_id": "engine-1", "transfer_id": "xfer-1"}
    block_ids = [1, 2, 3]

    with patch.object(mooncake_connector.threading, "Thread") as mock_thread:
        scheduler._submit_handle_request(
            "req-1",
            "http://bootstrap",
            params,
            block_ids,
        )

    mock_thread.assert_not_called()
    mock_executor.submit.assert_called_once()
    submit_args = mock_executor.submit.call_args.args
    assert submit_args[0] is scheduler._run_handle_request_task
    assert submit_args[1] == "req-1"
    assert submit_args[2] == "http://bootstrap"
    assert submit_args[3] == params
    assert submit_args[3] is not params
    assert submit_args[4] == block_ids
    assert submit_args[4] is not block_ids
    assert mock_future in scheduler._handle_request_futures
    assert mock_future.add_done_callback.call_count == 2


def test_update_state_after_alloc_layerwise_remote_prefill_submits_to_pool():
    scheduler = _make_scheduler()
    scheduler._submit_handle_request = MagicMock()

    request = create_request(do_remote_prefill=True)
    request.kv_transfer_params.update(
        {
            "remote_bootstrap_addr": "http://bootstrap",
            "transfer_id": "xfer-1",
        }
    )

    blocks = MagicMock()
    blocks.get_unhashed_block_ids.return_value = [10, 11]

    with patch.object(mooncake_connector.threading, "Thread") as mock_thread:
        scheduler.update_state_after_alloc(request, blocks, 2)

    mock_thread.assert_not_called()
    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert scheduler._reqs_need_recv[request.request_id] == (request, [10, 11])
    scheduler._submit_handle_request.assert_called_once_with(
        request.request_id,
        "http://bootstrap",
        request.kv_transfer_params,
        [10, 11],
    )
