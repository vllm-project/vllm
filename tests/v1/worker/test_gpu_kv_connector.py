# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.kv_connector as kv_connector_module
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def _make_connector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> ActiveKVConnector:
    backend = Mock()
    backend.handle_preemptions.side_effect = lambda _: events.append("handle")
    backend.bind_connector_metadata.side_effect = lambda _: events.append("bind")
    backend.start_load_kv.side_effect = lambda *_args, **_kwargs: events.append("start")
    backend.wait_for_save.side_effect = lambda: events.append("wait")
    backend.get_transfer_results.return_value = KVConnectorTransferResults()
    backend.get_block_ids_with_load_errors.return_value = set()
    backend.get_kv_connector_stats.return_value = None
    backend.get_kv_connector_kv_cache_events.return_value = None
    backend.build_connector_worker_meta.return_value = None
    backend.clear_connector_metadata.side_effect = lambda: events.append("clear")
    monkeypatch.setattr(kv_connector_module, "get_kv_transfer_group", lambda: backend)
    monkeypatch.setattr(
        kv_connector_module, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(kv_connector_module, "get_forward_context", object)

    kv_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_consumer",
        kv_buffer_device="cpu",
    )
    connector = ActiveKVConnector(
        cast(VllmConfig, SimpleNamespace(kv_transfer_config=kv_config)), {}
    )
    events.clear()
    return connector


def _scheduler_output(has_sync_kv_loads: bool) -> SchedulerOutput:
    return cast(
        SchedulerOutput,
        SimpleNamespace(
            kv_connector_metadata=object(),
            finished_req_ids=set(),
            has_sync_kv_loads=has_sync_kv_loads,
        ),
    )


@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
def test_load_start_phase(
    monkeypatch: pytest.MonkeyPatch,
    has_sync_kv_loads: bool,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    output = _scheduler_output(has_sync_kv_loads)

    request_indices = torch.tensor([3, 1])
    request_ids = ["first", "second"]
    attn_metadata = {"layer": object()}
    connector.pre_forward(
        output,
        request_state_indices=request_indices,
        request_ids=request_ids,
        attn_metadata=attn_metadata,
    )
    assert events == (
        ["handle", "bind", "start"] if has_sync_kv_loads else ["handle", "bind"]
    )

    connector.post_forward(set())
    assert events == ["handle", "bind", "start", "wait", "clear"]

    start_load_kv = cast(Mock, connector.kv_connector.start_load_kv)
    kwargs = start_load_kv.call_args.kwargs
    assert kwargs["request_state_indices"] is request_indices
    assert kwargs["request_ids"] is request_ids
    assert kwargs["attn_metadata"] is attn_metadata

    # A subsequent step without a forward must not reuse the prior batch.
    connector.no_forward(_scheduler_output(False))
    assert start_load_kv.call_count == 2
    assert start_load_kv.call_args.kwargs == {}


def test_no_forward_starts_deferred_load_once(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)

    connector.no_forward(_scheduler_output(False))

    assert events == ["handle", "bind", "start", "clear"]
