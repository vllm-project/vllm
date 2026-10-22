# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the generic asynchronous connector-init readiness handshake."""

from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    ConnectorInitState,
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRankInitStatus,
    KVConnectorRole,
    NoOpKVConnectorMetadata,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

pytestmark = pytest.mark.cpu_test


class _FakeConnector(KVConnectorBase_V1):
    """Minimal concrete connector: only the init-state hook is meaningful."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state: ConnectorInitState | None = None
        self.requests = []
        self.scheduler_outputs = []

    def on_new_request(self, request):
        self.requests.append(request.request_id)

    def get_connector_init_state(self) -> ConnectorInitState | None:
        return self.state

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        self.scheduler_outputs.append(replace(scheduler_output))
        return KVConnectorMetadata()

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


def _make_connector(
    role: KVConnectorRole,
    rank: int = 0,
    world_size: int = 2,
    backend: str = "mp",
    pipeline_parallel_size: int = 1,
) -> _FakeConnector:
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=True),
        parallel_config=SimpleNamespace(
            rank=rank,
            world_size=world_size,
            distributed_executor_backend=backend,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
    )
    return _FakeConnector(vllm_config, role, SimpleNamespace())


# ---------------------------------------------------------------------------
# KVConnectorRankInitStatus.aggregate — pure union semantics
# ---------------------------------------------------------------------------


def test_rank_init_status_aggregate_unions_ready_ranks():
    """aggregate() must union ready_ranks from both statuses."""
    a = KVConnectorRankInitStatus(ready_ranks={0})
    b = KVConnectorRankInitStatus(ready_ranks={1})
    assert a.aggregate(b).ready_ranks == {0, 1}


# ---------------------------------------------------------------------------
# is_connector_ready
# ---------------------------------------------------------------------------


def test_is_connector_ready_true_when_async_init_disabled():
    """Without enable_async_init(), the connector is ready immediately."""
    connector = _make_connector(KVConnectorRole.WORKER)
    assert connector.is_connector_ready() is True


def test_worker_is_connector_ready_tracks_own_init_state():
    """A worker's readiness must follow get_connector_init_state() directly."""
    connector = _make_connector(KVConnectorRole.WORKER)
    connector.enable_async_init()
    connector.state = ConnectorInitState.INITIALIZING
    assert connector.is_connector_ready() is False
    connector.state = ConnectorInitState.READY
    assert connector.is_connector_ready() is True


# ---------------------------------------------------------------------------
# build_connector_init_status — worker side
# ---------------------------------------------------------------------------


def test_build_connector_init_status_reports_own_rank_when_ready():
    connector = _make_connector(KVConnectorRole.WORKER, rank=1)
    connector.enable_async_init()
    connector.state = ConnectorInitState.READY
    status = connector.build_connector_init_status()
    assert isinstance(status, KVConnectorRankInitStatus)
    assert status.ready_ranks == {1}


def test_build_connector_init_status_empty_while_initializing():
    connector = _make_connector(KVConnectorRole.WORKER, rank=1)
    connector.enable_async_init()
    connector.state = ConnectorInitState.INITIALIZING
    status = connector.build_connector_init_status()
    assert status.ready_ranks == set()


def test_build_connector_init_status_none_once_scheduler_acknowledged():
    """Binding real metadata marks the report acknowledged; no further
    status reports should be built afterwards."""
    connector = _make_connector(KVConnectorRole.WORKER)
    connector.enable_async_init()
    connector.state = ConnectorInitState.READY
    connector.bind_connector_metadata(SimpleNamespace())
    assert connector.build_connector_init_status() is None


# ---------------------------------------------------------------------------
# update_connector_init_status — scheduler side
# ---------------------------------------------------------------------------


def test_update_connector_init_status_ready_once_all_ranks_report():
    connector = _make_connector(KVConnectorRole.SCHEDULER, world_size=2)
    connector.enable_async_init()
    assert connector.is_connector_ready() is False

    connector.update_connector_init_status(KVConnectorRankInitStatus(ready_ranks={0}))
    assert connector.is_connector_ready() is False

    connector.update_connector_init_status(KVConnectorRankInitStatus(ready_ranks={1}))
    assert connector.is_connector_ready() is True


def test_update_connector_init_status_noop_for_worker_role():
    """This method only applies scheduler-side; a worker's readiness must
    keep tracking its own init state instead."""
    connector = _make_connector(KVConnectorRole.WORKER, world_size=1)
    connector.enable_async_init()
    connector.update_connector_init_status(KVConnectorRankInitStatus(ready_ranks={0}))
    assert connector.is_connector_ready() is False


@pytest.mark.parametrize("backend,pp_size", [("external_launcher", 1), ("ray", 2)])
def test_async_init_rejects_executors_without_all_worker_reports(backend, pp_size):
    connector = _make_connector(
        KVConnectorRole.SCHEDULER, backend=backend, pipeline_parallel_size=pp_size
    )
    assert connector.is_connector_ready()
    with pytest.raises(ValueError, match="requires readiness reports"):
        connector.enable_async_init()


def test_worker_readiness_is_latched_and_cannot_be_restarted():
    connector = _make_connector(KVConnectorRole.WORKER)
    connector.enable_async_init()
    connector.state = ConnectorInitState.READY
    assert connector.is_connector_ready()
    connector.state = ConnectorInitState.INITIALIZING
    assert connector.is_connector_ready()
    with pytest.raises(RuntimeError, match="one-shot"):
        connector.enable_async_init()


def test_idle_scheduler_polls_until_all_connector_workers_are_ready():
    scheduler = create_scheduler(use_kv_connector=True)
    connector = _make_connector(KVConnectorRole.SCHEDULER)
    connector.enable_async_init()
    scheduler.connector = connector
    workers = [_make_connector(KVConnectorRole.WORKER, rank=i) for i in range(2)]
    for worker in workers:
        worker.enable_async_init()
        worker.state = ConnectorInitState.INITIALIZING
    aggregator = KVOutputAggregator(expected_finished_count=2)

    for ready_worker in workers:
        assert scheduler.has_requests()
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens == 0
        assert isinstance(output.kv_connector_metadata, NoOpKVConnectorMetadata)
        ready_worker.state = ConnectorInitState.READY
        worker_outputs = [
            ModelRunnerOutput.with_kv_conn_output_only(
                KVConnectorOutput(
                    kv_connector_init_status=worker.build_connector_init_status()
                )
            )
            for worker in workers
        ]
        scheduler.update_from_output(output, aggregator.aggregate(worker_outputs))

    assert connector.is_connector_ready()
    assert not scheduler.has_requests()


@pytest.mark.parametrize("ready_before_finish", [False, True])
def test_excluded_request_stays_hidden_through_abort_and_worker_notification(
    ready_before_finish,
):
    scheduler = create_scheduler(use_kv_connector=True)
    connector = _make_connector(KVConnectorRole.SCHEDULER, world_size=1)
    connector.enable_async_init()
    scheduler.connector = connector
    [request] = create_requests(num_requests=1)
    scheduler.add_request(request)
    scheduler.schedule()
    if ready_before_finish:
        connector.update_connector_init_status(KVConnectorRankInitStatus({0}))

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    assert output.finished_req_ids == {request.request_id}
    if ready_before_finish:
        assert connector.scheduler_outputs[-1].finished_req_ids == set()

    # Reusing the ID after the terminal step must not inherit its exclusion.
    connector.update_connector_init_status(KVConnectorRankInitStatus({0}))
    [replacement] = create_requests(num_requests=1, req_ids=[request.request_id])
    scheduler.add_request(replacement)
    output = scheduler.schedule()
    assert connector.requests == [request.request_id]
    assert connector.scheduler_outputs[-1].num_scheduled_tokens


@pytest.mark.parametrize("load_async", [False, True])
def test_allocation_notifications_include_external_load_destinations(
    monkeypatch, load_async
):
    scheduler = create_scheduler(use_kv_connector=True, block_size=16)
    connector = _make_connector(KVConnectorRole.SCHEDULER)
    monkeypatch.setattr(
        connector, "get_num_new_matched_tokens", Mock(return_value=(32, load_async))
    )
    scheduler.connector = connector
    [request] = create_requests(num_requests=1, num_tokens=64)
    scheduler.add_request(request)

    scheduler.schedule()
    block_state = connector.scheduler_outputs[-1].kv_connector_block_state
    allocated = block_state.allocated_block_ids
    [blocks] = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    assert allocated == set(blocks)
    assert len(allocated) == (2 if load_async else 4)

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()
    next_block_state = connector.scheduler_outputs[-1].kv_connector_block_state
    assert not next_block_state.allocated_block_ids
    assert allocated == set(blocks)


@pytest.mark.parametrize("use_pp", [False, True])
def test_filtering_cached_requests_preserves_alignment_and_original(use_pp):
    cached = CachedRequestData(
        req_ids=["early", "late"],
        resumed_req_ids={"early", "late"},
        new_token_ids=[[10], [20]] if use_pp else [],
        all_token_ids={"early": [10], "late": [20]},
        new_block_ids=[([1],), ([2],)],
        num_computed_tokens=[10, 20],
        num_output_tokens=[0, 1],
    )
    assert cached.is_context_phase("early")
    output = SchedulerOutput.make_empty()
    output.scheduled_cached_reqs = cached
    output.num_scheduled_tokens = {"early": 2, "late": 3}
    output.total_num_scheduled_tokens = 5

    filtered = output.without_requests({"early"})
    assert filtered.scheduled_cached_reqs == CachedRequestData(
        req_ids=["late"],
        resumed_req_ids={"late"},
        new_token_ids=[[20]] if use_pp else [],
        all_token_ids={"late": [20]},
        new_block_ids=[([2],)],
        num_computed_tokens=[20],
        num_output_tokens=[1],
    )
    assert not filtered.scheduled_cached_reqs.is_context_phase("early")
    assert filtered.num_scheduled_tokens == {"late": 3}
    assert filtered.total_num_scheduled_tokens == 3
    assert output.total_num_scheduled_tokens == 5
    assert cached.req_ids == ["early", "late"]


# Fields without_requests() leaves untouched because they are not per-request.
_SCHEDULER_OUTPUT_GLOBAL_FIELDS = {
    "num_common_prefix_blocks",
    "free_encoder_mm_hashes",
    "scheduled_encoder_input_stats",
    "has_structured_output_requests",
    "pending_structured_output_tokens",
    "kv_connector_metadata",
    "has_sync_kv_loads",
    "aux_output_connector_metadata",
    "ec_connector_metadata",
    "ec_manager_metadata",
    "new_block_ids_to_zero",
    "kv_cache_block_copies",
    "num_spec_tokens_to_schedule",
}
_SCHEDULER_OUTPUT_FILTERED_FIELDS = {
    "scheduled_new_reqs",
    "scheduled_cached_reqs",
    "num_scheduled_tokens",
    "total_num_scheduled_tokens",
    "scheduled_spec_decode_tokens",
    "scheduled_encoder_inputs",
    "num_invalid_spec_tokens",
    "finished_req_ids",
    "preempted_req_ids",
    "kv_connector_block_state",
}
_CACHED_REQUEST_DATA_FILTERED_FIELDS = {
    "req_ids",
    "resumed_req_ids",
    "new_token_ids",
    "all_token_ids",
    "new_block_ids",
    "num_computed_tokens",
    "num_output_tokens",
}


def test_without_requests_classifies_every_field():
    """New request-keyed fields must be filtered in without_requests()."""
    assert {f.name for f in fields(SchedulerOutput)} == (
        _SCHEDULER_OUTPUT_GLOBAL_FIELDS | _SCHEDULER_OUTPUT_FILTERED_FIELDS
    )
    assert {
        f.name for f in fields(CachedRequestData)
    } == _CACHED_REQUEST_DATA_FILTERED_FIELDS


def _worker_step(monkeypatch, connector, output, runner):
    if runner == "v1":
        module = "vllm.v1.worker.kv_connector_model_runner_mixin"
        monkeypatch.setattr(f"{module}.get_kv_transfer_group", lambda: connector)
        monkeypatch.setattr(f"{module}.get_forward_context", lambda: None)
        with KVConnectorModelRunnerMixin._get_kv_connector_output(output) as result:
            pass
        return result

    module = "vllm.v1.worker.gpu.kv_connector"
    monkeypatch.setattr(f"{module}.get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(f"{module}.is_forward_context_available", lambda: True)
    monkeypatch.setattr(f"{module}.get_forward_context", lambda: None)
    active = ActiveKVConnector(connector._vllm_config, {})
    return active.no_forward(output).kv_connector_output


@pytest.mark.parametrize("runner", ["v1", "v2"])
@pytest.mark.parametrize("fail", [False, True])
def test_workers_poll_initialization_without_transfer_calls(monkeypatch, runner, fail):
    connector = _make_connector(KVConnectorRole.WORKER, world_size=1)
    connector.enable_async_init()
    connector.state = ConnectorInitState.READY
    for hook in (
        "handle_preemptions",
        "start_load_kv",
        "wait_for_save",
        "get_finished",
    ):
        monkeypatch.setattr(connector, hook, Mock(side_effect=AssertionError(hook)))
    if fail:
        monkeypatch.setattr(
            connector,
            "get_connector_init_state",
            Mock(side_effect=RuntimeError("initialization failed")),
        )
    output = SchedulerOutput.make_empty()
    output.kv_connector_metadata = NoOpKVConnectorMetadata()
    if fail:
        with pytest.raises(RuntimeError, match="initialization failed"):
            _worker_step(monkeypatch, connector, output, runner)
    else:
        result = _worker_step(monkeypatch, connector, output, runner)
        assert result.kv_connector_init_status.ready_ranks == {0}
