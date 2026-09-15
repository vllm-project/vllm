# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.kv_connector as kv_connector_module
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def _make_connector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> ActiveKVConnector:
    backend = Mock()
    backend.handle_preemptions.side_effect = lambda _: events.append("handle")
    backend.bind_connector_metadata.side_effect = lambda _: events.append("bind")
    backend.start_load_kv.side_effect = lambda *_args, **_kwargs: events.append("start")
    backend.finish_forward.side_effect = lambda: events.append("finish")
    backend.wait_for_save.side_effect = lambda: events.append("wait")
    backend.get_transfer_results.return_value = KVConnectorTransferResults()
    backend.get_block_ids_with_load_errors.return_value = set()
    backend.get_kv_connector_stats.return_value = None
    backend.get_kv_connector_kv_cache_events.return_value = None
    backend.build_connector_worker_meta.return_value = None
    backend.clear_connector_metadata.side_effect = lambda: events.append("clear")
    monkeypatch.setattr(kv_connector_module, "get_kv_transfer_group", lambda: backend)

    kv_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_consumer",
        kv_buffer_device="cpu",
    )
    connector = ActiveKVConnector(VllmConfig(kv_transfer_config=kv_config), {})
    events.clear()
    return connector


def _scheduler_output(has_sync_kv_loads: bool) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector_metadata=object(),
        finished_req_ids=set(),
        has_sync_kv_loads=has_sync_kv_loads,
    )


@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
@pytest.mark.parametrize("has_forward_context", [False, True])
def test_load_start_phase(
    monkeypatch: pytest.MonkeyPatch,
    has_sync_kv_loads: bool,
    has_forward_context: bool,
):
    """Loads use the target context; save finalization waits until after drafting."""
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    output = _scheduler_output(has_sync_kv_loads)

    request_indices = torch.tensor([3, 1])
    request_ids = ["first", "second"]
    attn_metadata = {"layer": object()}
    with (
        set_forward_context(attn_metadata, connector.vllm_config)
        if has_forward_context
        else nullcontext()
    ):
        connector.pre_forward(  # type: ignore[arg-type]
            output,
            request_state_indices=request_indices,
            request_ids=request_ids,
            attn_metadata=attn_metadata,
        )
        assert events == (
            ["handle", "bind", "start"] if has_sync_kv_loads else ["handle", "bind"]
        )
        events.append("forward")
        connector.finish_forward()
        expected = (
            ["handle", "bind", "start", "forward", "finish"]
            if has_sync_kv_loads
            else ["handle", "bind", "forward", "finish", "start"]
        )
        assert events == expected
        context = connector.kv_connector.start_load_kv.call_args.args[0]
        if has_forward_context:
            assert context is get_forward_context()
            assert context.attn_metadata is attn_metadata
        else:
            assert context.attn_metadata is None

    assert not is_forward_context_available()
    events.append("draft")
    connector.post_forward(set())
    assert events == expected + ["draft", "wait", "clear"]
    assert connector.kv_connector.start_load_kv.call_count == 1

    kwargs = connector.kv_connector.start_load_kv.call_args.kwargs
    assert kwargs["request_state_indices"] is request_indices
    assert kwargs["request_ids"] is request_ids
    assert kwargs["attn_metadata"] is attn_metadata

    # A subsequent step without a forward must not reuse the prior batch.
    connector.no_forward(_scheduler_output(False))  # type: ignore[arg-type]
    assert connector.kv_connector.start_load_kv.call_count == 2
    assert connector.kv_connector.start_load_kv.call_args.kwargs == {}


@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
def test_no_forward_starts_loads_once(
    monkeypatch: pytest.MonkeyPatch, has_sync_kv_loads: bool
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)

    connector.no_forward(_scheduler_output(has_sync_kv_loads))  # type: ignore[arg-type]

    assert events == (
        ["handle", "bind", "start", "finish", "clear"]
        if has_sync_kv_loads
        else ["handle", "bind", "finish", "start", "clear"]
    )


def test_disabled_finish_forward_does_not_start_deferred_loads(monkeypatch):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    connector.pre_forward(_scheduler_output(False))  # type: ignore[arg-type]
    monkeypatch.setattr(
        kv_connector_module.kv_transfer_state, "_KV_CONNECTOR_AGENT", None
    )
    connector.set_disabled(True)

    connector.finish_forward()

    assert events == ["handle", "bind"]
