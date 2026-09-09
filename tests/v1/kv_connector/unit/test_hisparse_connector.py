# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
    HiSparseConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)
from vllm.v1.hisparse.runtime import HiSparseCacheHandle
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def test_hisparse_requires_block_outermost_device_layout():
    assert HiSparseConnector.get_required_kvcache_layout(MagicMock()) == "BLHNC"


def test_no_forward_enqueues_deferred_hisparse_transfers():
    """A zero-token step must still enqueue deferred post-forward transfers."""
    connector = object.__new__(ActiveKVConnector)
    connector._disabled = False
    connector.pre_forward = MagicMock()
    connector.finish_forward = MagicMock()
    connector.post_forward = MagicMock(return_value=None)

    scheduler_output = SimpleNamespace(finished_req_ids=set())
    connector.no_forward(scheduler_output)

    connector.pre_forward.assert_called_once_with(scheduler_output)
    connector.finish_forward.assert_called_once_with()


def test_full_graph_step_prepares_host_mirror_outside_model():
    """Graph replay must restore host-mirror state cleared at step start."""
    runtime = SimpleNamespace(
        is_group_leader=True,
        eager_host_mirror=True,
        begin_forward=MagicMock(),
        invalidate_written_slots=MagicMock(),
    )
    handle = HiSparseCacheHandle(runtime)
    handle.mirror_slot_mapping = torch.tensor([4, 5])
    worker = object.__new__(HiSparseConnectorWorker)
    worker.cache_layer_names = ["layer"]
    worker.cache_handles = [handle]
    worker._group_leaders = (("layer", handle),)
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    worker.is_host_writer = True
    worker._enqueue_row_dma = MagicMock()
    worker.start_step = MagicMock(
        side_effect=lambda *_args: worker._clear_forward_mirror_state()
    )

    connector = object.__new__(HiSparseConnector)
    connector.connector_worker = worker
    connector._get_connector_metadata = MagicMock(
        return_value=HiSparseConnectorMetadata(None, (), (), {}, True)
    )
    req_id_per_token = torch.tensor([0, 1], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_decode_tokens=2,
        num_reqs=2,
        max_query_len=1,
        req_id_per_token=req_id_per_token,
    )

    connector.start_load_kv(
        SimpleNamespace(),
        request_state_indices=None,
        request_ids=[],
        attn_metadata={"layer": attn_metadata},
    )

    worker._enqueue_host_mirror()

    worker._enqueue_row_dma.assert_called_once_with((0,), ready_event=None)
    runtime.invalidate_written_slots.assert_called_once()
