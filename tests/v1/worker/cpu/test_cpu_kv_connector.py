# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import numpy as np
import pytest
import torch

import vllm.v1.worker.cpu.kv_connector as kv_connector_module
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.cpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.cpu.kv_connector import ActiveKVConnector

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)


class FakeKVTransferBackend:
    def __init__(self, events: list[str]):
        self.events = events
        self.load_calls: list[tuple[object, dict[str, Any]]] = []

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.kv_caches = kv_caches

    def set_host_xfer_buffer_ops(self, copy_blocks_op: object) -> None:
        self.copy_blocks_op = copy_blocks_op

    def handle_preemptions(self, metadata: object) -> None:
        self.events.append("handle_preemptions")

    def bind_connector_metadata(self, metadata: object) -> None:
        self.events.append("bind_metadata")

    def start_load_kv(self, context: object, **kwargs: Any) -> None:
        self.events.append("start_load")
        self.load_calls.append((context, kwargs))

    def finish_forward(self) -> None:
        self.events.append("finish_forward")

    def wait_for_save(self) -> None:
        self.events.append("wait_for_save")

    def get_transfer_results(self, finished_req_ids: set[str]):
        assert finished_req_ids == {"finished"}
        return KVConnectorTransferResults(
            finished_sending={"sent"},
            finished_recving={"received"},
            failed_recving={"failed"},
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        return {7}

    def get_kv_connector_stats(self) -> None:
        return None

    def get_kv_connector_kv_cache_events(self) -> None:
        return None

    def build_connector_worker_meta(self) -> None:
        return None

    def clear_connector_metadata(self) -> None:
        self.events.append("clear_metadata")


def _make_connector(
    monkeypatch: pytest.MonkeyPatch, events: list[str]
) -> tuple[ActiveKVConnector, FakeKVTransferBackend, object]:
    backend = FakeKVTransferBackend(events)
    forward_context = object()
    monkeypatch.setattr(kv_connector_module, "get_kv_transfer_group", lambda: backend)
    monkeypatch.setattr(
        kv_connector_module, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(
        kv_connector_module, "get_forward_context", lambda: forward_context
    )
    connector = ActiveKVConnector(VllmConfig(), {"layer": torch.empty(1)})
    return connector, backend, forward_context


def _scheduler_output(has_sync_kv_loads: bool) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.kv_connector_metadata = object()
    output.has_sync_kv_loads = has_sync_kv_loads
    output.finished_req_ids = {"finished"}
    return output


def _input_batch() -> InputBatch:
    buffers = InputBuffers(4, 5, torch.device("cpu"))
    batch = InputBatch.make_dummy(2, 5, buffers)
    batch.req_ids = ["first", "second"]
    batch.idx_mapping = torch.tensor([3, 1])
    batch.idx_mapping_np = np.array([3, 1], dtype=np.intp)
    return batch


@pytest.mark.parametrize("sync_load", [False, True])
def test_cpu_kv_connector_preserves_transfer_phase_order(
    monkeypatch: pytest.MonkeyPatch, sync_load: bool
) -> None:
    events: list[str] = []
    connector, backend, forward_context = _make_connector(monkeypatch, events)
    scheduler_output = _scheduler_output(sync_load)
    input_batch = _input_batch()
    attn_metadata = {"layer": object()}

    connector.pre_forward(scheduler_output, input_batch, attn_metadata)
    connector.finish_forward()
    output = connector.post_forward({"finished"})

    expected = ["handle_preemptions", "bind_metadata"]
    if sync_load:
        expected.append("start_load")
    expected.append("finish_forward")
    if not sync_load:
        expected.append("start_load")
    expected.extend(("wait_for_save", "clear_metadata"))
    assert events == expected

    assert len(backend.load_calls) == 1
    context, load_kwargs = backend.load_calls[0]
    assert context is forward_context
    assert load_kwargs == {
        "scheduler_output": scheduler_output,
        "request_state_indices": input_batch.idx_mapping,
        "request_ids": input_batch.req_ids,
        "num_tokens": 5,
        "attn_metadata": attn_metadata,
    }
    assert output is not None
    assert output.finished_sending == {"sent"}
    assert output.finished_recving == {"received"}
    assert output.failed_recving == {"failed"}
    assert output.invalid_block_ids == {7}

    connector.no_forward(_scheduler_output(False))
    assert backend.load_calls[-1] == (forward_context, {})


def test_cpu_kv_connector_empty_step_does_not_reuse_batch_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    connector, backend, forward_context = _make_connector(monkeypatch, events)
    scheduler_output = _scheduler_output(False)

    output = connector.no_forward(scheduler_output)

    assert events == [
        "handle_preemptions",
        "bind_metadata",
        "finish_forward",
        "start_load",
        "wait_for_save",
        "clear_metadata",
    ]
    assert backend.load_calls == [(forward_context, {})]
    assert output.kv_connector_output is not None
    assert output.kv_connector_output.finished_sending == {"sent"}
