# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import sys
import types
from types import SimpleNamespace
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
MooncakeConnectorWorker = mooncake_connector.MooncakeConnectorWorker
MooncakeConnectorMetadata = mooncake_connector.MooncakeConnectorMetadata
SendTask = mooncake_connector.SendTask


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


def _make_worker(
    *,
    layer_chunk_size: int = 4,
    total_layers: int = 8,
    current_layer: int = 0,
) -> MooncakeConnectorWorker:
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.is_kv_producer = True
    worker.is_kv_consumer = False
    worker.use_layerwise = True
    worker.layer_chunk_size = layer_chunk_size
    worker.total_layers = total_layers
    worker.current_layer = current_layer
    worker.pending_layer_send_tasks = []
    worker.layerwise_send_queue = MagicMock()
    worker.sender_loop = MagicMock()
    worker.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=True)
    )
    return worker


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


def test_wait_for_layer_load_flushes_previous_chunk():
    worker = _make_worker(layer_chunk_size=4, total_layers=8, current_layer=3)
    worker.pending_layer_send_tasks = [
        SendTask(layer_idx=0),
        SendTask(layer_idx=1),
        SendTask(layer_idx=2),
    ]

    worker.wait_for_layer_load("layer_3")

    assert worker.sender_loop.call_soon_threadsafe.call_count == 1
    batch_task = worker.sender_loop.call_soon_threadsafe.call_args.args[1]
    assert batch_task.layer_idxs == [0, 1, 2]
    assert batch_task.layer_idx == 2
    assert worker.pending_layer_send_tasks == []


def test_wait_for_layer_load_does_not_flush_non_boundary():
    worker = _make_worker(layer_chunk_size=4, total_layers=8, current_layer=2)
    worker.pending_layer_send_tasks = [
        SendTask(layer_idx=0),
        SendTask(layer_idx=1),
    ]

    worker.wait_for_layer_load("layer_2")

    worker.sender_loop.call_soon_threadsafe.assert_not_called()
    assert [task.layer_idx for task in worker.pending_layer_send_tasks] == [0, 1]


def test_save_kv_layer_flushes_remaining_layers_on_last_layer():
    worker = _make_worker(layer_chunk_size=4, total_layers=5, current_layer=4)
    worker.pending_layer_send_tasks = [
        SendTask(layer_idx=1),
        SendTask(layer_idx=2),
        SendTask(layer_idx=3),
    ]
    connector_metadata = MooncakeConnectorMetadata()
    connector_metadata.send_task.send_request["req-1"] = MagicMock()
    attn_metadata = SimpleNamespace(reshape_cache_event=MagicMock())

    worker.save_kv_layer("layer_4", MagicMock(), attn_metadata, connector_metadata)

    assert worker.sender_loop.call_soon_threadsafe.call_count == 1
    batch_task = worker.sender_loop.call_soon_threadsafe.call_args.args[1]
    assert batch_task.layer_idxs == [1, 2, 3, 4]
    assert batch_task.layer_idx == 4
    assert worker.pending_layer_send_tasks == []
    assert worker.current_layer == 5


def test_transfer_kv_cache_batches_multiple_layers_into_one_transfer():
    worker = _make_worker(layer_chunk_size=4, total_layers=8, current_layer=0)
    worker.tp_rank = 0
    worker.kv_caches_base_addr = [100, 200, 300, 400, 500, 600]
    worker.block_len = 10
    worker.remote_agent_meta = {"engine-1": {0: ("worker-addr", "session-1", [1000, 1100, 1200, 1300, 1400, 1500])}}
    worker.engine = MagicMock()
    worker.engine.batch_transfer_sync_write.return_value = 0
    worker.send_done_send_signal = MagicMock()

    req_meta = mooncake_connector.PushReqMeta(
        d_req_id="d-req-1",
        transfer_id="xfer-1",
        local_block_ids=[1, 2],
        remote_block_ids=[5, 6],
        remote_engine_id="engine-1",
    )
    wait_event_1 = MagicMock()
    wait_event_2 = MagicMock()
    batch_task = SendTask(
        send_request={"req-1": req_meta},
        wait_events=[wait_event_1, wait_event_2],
        layer_idxs=[0, 1],
        layer_idx=1,
    )

    worker._transfer_kv_cache(batch_task)

    wait_event_1.synchronize.assert_called_once()
    wait_event_2.synchronize.assert_called_once()
    worker.engine.batch_transfer_sync_write.assert_called_once()
    call_args = worker.engine.batch_transfer_sync_write.call_args.args
    assert call_args[0] == "session-1"
    assert call_args[1] == [110, 120, 210, 220, 310, 320, 410, 420]
    assert call_args[2] == [1050, 1060, 1150, 1160, 1250, 1260, 1350, 1360]
    assert call_args[3] == [20, 20, 20, 20, 20, 20, 20, 20]
    worker.send_done_send_signal.assert_not_called()
