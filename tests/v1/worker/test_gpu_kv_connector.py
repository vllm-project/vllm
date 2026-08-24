# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.v1.worker.gpu.kv_connector as kv_connector_module
from vllm.config import KVTransferConfig
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def _make_connector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> ActiveKVConnector:
    backend = Mock()
    backend.handle_preemptions.side_effect = lambda _: events.append("handle")
    backend.bind_connector_metadata.side_effect = lambda _: events.append("bind")
    backend.start_load_kv.side_effect = lambda _: events.append("start")
    backend.wait_for_save.side_effect = lambda: events.append("wait")
    backend.get_finished.side_effect = lambda _: (set(), set())
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
    connector = ActiveKVConnector(  # type: ignore[arg-type]
        SimpleNamespace(kv_transfer_config=kv_config), {}
    )
    events.clear()
    return connector


def _scheduler_output(async_load: bool) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector_metadata=object(),
        finished_req_ids=set(),
        async_load=async_load,
    )


@pytest.mark.parametrize("async_load", [False, True])
def test_async_load_start_phase(
    monkeypatch: pytest.MonkeyPatch,
    async_load: bool,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    output = _scheduler_output(async_load)

    connector.pre_forward(output)  # type: ignore[arg-type]
    assert events == (["handle", "bind"] if async_load else ["handle", "bind", "start"])

    connector.post_forward(set(), async_load=output.async_load)
    assert events == ["handle", "bind", "start", "wait", "clear"]


def test_no_forward_starts_deferred_load_once(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)

    connector.no_forward(_scheduler_output(True))  # type: ignore[arg-type]

    assert events == ["handle", "bind", "start", "clear"]


def test_async_load_is_explicit_boolean():
    assert not KVTransferConfig().async_load
    assert KVTransferConfig(async_load=True).async_load
    with pytest.raises(ValueError):
        KVTransferConfig(async_load=None)  # type: ignore[arg-type]
