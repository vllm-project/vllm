# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.v1.worker.gpu.kv_connector as kv_connector_module
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def _make_connector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    connector: str = "NixlConnector",
    module_path: str | None = None,
    role: str = "kv_consumer",
    buffer_device: str = "cuda",
    pipeline_parallel_size: int = 1,
    runner_type: str = "generate",
) -> ActiveKVConnector:
    def get_finished(finished: set[str]) -> tuple[set[str], set[str]]:
        events.append("finished")
        return set(), set()

    backend = Mock()
    backend.handle_preemptions.side_effect = lambda metadata: events.append("handle")
    backend.bind_connector_metadata.side_effect = lambda metadata: events.append("bind")
    backend.start_load_kv.side_effect = lambda context: events.append("start")
    backend.wait_for_save.side_effect = lambda: events.append("wait")
    backend.get_finished.side_effect = get_finished
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

    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector=connector,
            kv_connector_module_path=module_path,
            kv_role=role,
            kv_buffer_device=buffer_device,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pipeline_parallel_size),
        model_config=SimpleNamespace(runner_type=runner_type),
    )
    active = ActiveKVConnector(config, {})  # type: ignore[arg-type]
    events.clear()
    return active


def _scheduler_output() -> SimpleNamespace:
    return SimpleNamespace(kv_connector_metadata=object(), finished_req_ids=set())


@pytest.mark.parametrize(
    ("overrides", "deferred"),
    [
        ({}, True),
        ({"buffer_device": "cpu"}, False),
        ({"role": "kv_both"}, False),
        ({"connector": "MultiConnector"}, False),
        ({"module_path": "custom.connector"}, False),
        ({"pipeline_parallel_size": 2}, False),
        ({"runner_type": "pooling"}, False),
    ],
)
def test_nixl_load_start_phase(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object], deferred: bool
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events, **overrides)  # type: ignore[arg-type]
    scheduler_output = _scheduler_output()

    connector.pre_forward(scheduler_output)  # type: ignore[arg-type]
    assert events == (["handle", "bind"] if deferred else ["handle", "bind", "start"])

    connector.post_forward(scheduler_output)  # type: ignore[arg-type]
    assert events == ["handle", "bind", "start", "wait", "finished", "clear"]


def test_no_forward_starts_deferred_load_once(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)

    connector.no_forward(_scheduler_output())  # type: ignore[arg-type]

    assert events == ["handle", "bind", "start", "finished", "clear"]
