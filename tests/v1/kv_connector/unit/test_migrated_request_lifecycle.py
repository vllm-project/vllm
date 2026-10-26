# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle of requests handed off by ``pause_generation(mode="migrate")``."""

import copy
import queue
from dataclasses import replace

import pytest

from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, DraftTokenIds, KVConnectorOutput
from vllm.v1.request import FinishReason, RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)

pytestmark = pytest.mark.cpu_test


def test_migrated_request_keeps_kv_readable_until_sent():
    """A decoding request is exported like a finished remote prefill, and its
    blocks stay allocated until the consumer reports that it read them."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    block_size = vllm_config.cache_config.block_size
    request = create_request(
        request_id=1, block_size=block_size, num_tokens=2 * block_size, max_tokens=8
    )
    scheduler.add_request(request)

    # Prefill, then decode one token. The last sampled token has no KV yet.
    for _ in range(2):
        scheduler_output = scheduler.schedule()
        scheduler.update_from_output(
            scheduler_output, create_model_runner_output(reqs=[request])
        )
    assert request.num_computed_tokens == request.num_tokens - 1

    (output,) = scheduler.migrate_requests()[0].outputs

    assert request.status == RequestStatus.FINISHED_MIGRATED
    assert output.finish_reason == FinishReason.ABORT
    params = output.kv_transfer_params
    assert params["do_remote_prefill"] and not params["do_remote_decode"]
    assert params["remote_request_id"] == request.request_id
    assert params["remote_num_tokens"] == request.num_computed_tokens
    assert len(params["remote_block_ids"][0]) == cdiv(
        request.num_computed_tokens, block_size
    )

    # Workers must track the export, or they drop the consumer's read notice.
    scheduler_output = scheduler.schedule()
    assert request.request_id in scheduler_output.finished_req_ids
    metadata = scheduler_output.kv_connector_metadata
    assert request.request_id in metadata.reqs_in_batch
    assert request.request_id in metadata.reqs_to_send
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)
    assert scheduler.has_requests()

    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_sending={request.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert_scheduler_empty(scheduler)


def test_migrated_request_without_kv_exports_nothing():
    """A request that never ran has no KV cache to hand off."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    request = create_request(
        request_id=1, block_size=vllm_config.cache_config.block_size
    )
    scheduler.add_request(request)

    (output,) = scheduler.migrate_requests()[0].outputs

    assert output.finish_reason == FinishReason.ABORT
    assert output.kv_transfer_params is None
    scheduler.schedule()
    assert_scheduler_empty(scheduler)


def test_migrating_request_awaiting_kv_does_not_export_unread_blocks():
    """An unfinished incoming transfer must not become an outgoing export."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    request = create_request(
        request_id=1,
        block_size=vllm_config.cache_config.block_size,
        num_tokens=32,
        do_remote_prefill=True,
    )
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    (output,) = scheduler.migrate_requests()[0].outputs

    assert output.finish_reason == FinishReason.ABORT
    assert output.kv_transfer_params is None
    scheduler_output = scheduler.schedule()
    assert not scheduler_output.kv_connector_metadata.reqs_to_send
    assert request.request_id in scheduler.requests

    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_recving={request.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert_scheduler_empty(scheduler)


@pytest.mark.parametrize("hybrid", [False, True], ids=["attention", "hybrid"])
def test_migrating_speculative_request_exports_only_committed_state(hybrid):
    """Rejected drafts must not extend the exported KV or recurrent state."""
    vllm_config = create_vllm_config(
        num_speculative_tokens=3,
        disable_hybrid_kv_cache_manager=False,
    )
    kv_cache_config = make_kv_cache_config(16, mamba_enabled=hybrid)
    if hybrid:
        group = kv_cache_config.kv_cache_groups[-1]
        group.kv_cache_spec = replace(group.kv_cache_spec, num_speculative_blocks=3)
        vllm_config.cache_config.mamba_cache_mode = "none"
        vllm_config.cache_config.enable_prefix_caching = False
    scheduler = create_scheduler(vllm_config, kv_cache_config=kv_cache_config)
    request = create_request(
        request_id=1, block_size=vllm_config.cache_config.block_size, num_tokens=30
    )
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output, create_model_runner_output(reqs=[request])
    )
    scheduler.update_draft_token_ids(DraftTokenIds([request.request_id], [[1, 2, 3]]))
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output, create_model_runner_output(reqs=[request])
    )
    assert request.num_computed_tokens == 31

    (output,) = scheduler.migrate_requests()[0].outputs

    params = output.kv_transfer_params
    assert params is not None
    assert params["remote_num_tokens"] == 31
    assert len(params["remote_block_ids"][0]) == 2
    if hybrid:
        assert len(params["remote_block_ids"][-1]) == 1
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_sending={request.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert_scheduler_empty(scheduler)


def test_consumer_loads_only_tokens_the_producer_computed():
    """The producer of a migrated request computed all but the last token of the
    consumer's prompt. Loading that token too would ask for a block the
    producer never filled when the boundary falls on a block edge."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    block_size = vllm_config.cache_config.block_size
    num_computed_tokens = 2 * block_size
    request = create_request(
        request_id=1,
        block_size=block_size,
        num_tokens=num_computed_tokens + 1,
        do_remote_prefill=True,
        num_remote_blocks=2,
    )
    request.kv_transfer_params["remote_num_tokens"] = num_computed_tokens

    assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (
        num_computed_tokens,
        True,
    )


class _Workers:
    """The engine's collective RPC boundary to its workers. ``failing`` names a
    method that raises on the workers, as a GPU-side error would."""

    def __init__(self, failing: str | None = None):
        self.failing = failing
        self.calls: list[tuple[str, tuple]] = []

    def collective_rpc(self, method: str, args: tuple = ()) -> list[None]:
        self.calls.append((method, args))
        if method == self.failing:
            raise RuntimeError(f"{method} failed on a worker")
        return [None]


def _engine_core(vllm_config, scheduler, workers) -> EngineCoreProc:
    """An engine core around a real scheduler and output queue, without the
    client I/O threads that EngineCoreProc.__init__ starts."""
    core = object.__new__(EngineCoreProc)
    core.vllm_config = vllm_config
    core.scheduler = scheduler
    core.model_executor = workers
    core.output_queue = queue.Queue()
    core.batch_queue = None
    core.engines_running = False
    core.reject_new_requests = False
    core.shutdown_state = EngineShutdownState.RUNNING
    core._idle_state_callbacks = []
    return core


def _decode_one_request(vllm_config, scheduler):
    block_size = vllm_config.cache_config.block_size
    request = create_request(
        request_id=1, block_size=block_size, num_tokens=2 * block_size, max_tokens=8
    )
    scheduler.add_request(request)
    for _ in range(2):
        scheduler_output = scheduler.schedule()
        scheduler.update_from_output(
            scheduler_output, create_model_runner_output(reqs=[request])
        )
    return request


def test_migrate_pause_hands_off_requests_and_rejects_new_ones():
    """The engine sends each request's final output with its transfer
    parameters, rejects new requests while paused, and completes the pause once
    the consumer has read the exported blocks."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    request = _decode_one_request(vllm_config, scheduler)
    workers = _Workers()
    core = _engine_core(vllm_config, scheduler, workers)

    future = core.pause_generation(mode="migrate", clear_cache=False)

    _, outputs = core.output_queue.get_nowait()
    (output,) = outputs.outputs
    assert output.finish_reason == FinishReason.ABORT
    assert output.kv_transfer_params["remote_request_id"] == request.request_id
    assert workers.calls == [("prepare_migration", ([request.request_id],))]
    # Exported blocks still wait for the consumer, so the engine is not idle.
    core._notify_idle_state_callbacks()
    assert not future.done()

    new_request = create_request(
        request_id=2, block_size=vllm_config.cache_config.block_size
    )
    core._handle_client_request(EngineCoreRequestType.ADD, (new_request, 0))
    _, rejected = core.output_queue.get_nowait()
    (rejection,) = rejected.outputs
    assert rejection.finish_reason == FinishReason.ERROR
    assert rejection.error.code == "engine_paused" and rejection.error.retryable
    assert not scheduler.has_request(new_request.request_id)

    # The consumer reads the exported blocks; the source frees them.
    scheduler.update_from_output(scheduler.schedule(), EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_sending={request.request_id}
    )
    scheduler.update_from_output(scheduler.schedule(), model_runner_output)
    assert_scheduler_empty(scheduler)
    core._notify_idle_state_callbacks()
    future.result(timeout=0)

    core.resume_scheduler()
    core._handle_client_request(EngineCoreRequestType.ADD, (new_request, 0))
    assert scheduler.has_request(new_request.request_id)
    assert core.output_queue.empty()


def test_migrate_preparation_failure_aborts_requests():
    """Workers that fail to commit speculative state for export may have
    committed part of it: every request gets a terminal abort without transfer
    parameters, its KV cache is freed, and the pause raises."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    request = _decode_one_request(vllm_config, scheduler)
    core = _engine_core(vllm_config, scheduler, _Workers(failing="prepare_migration"))

    future = core.pause_generation(mode="migrate", clear_cache=False)

    with pytest.raises(RuntimeError, match="prepare_migration failed"):
        future.result(timeout=0)
    _, outputs = core.output_queue.get_nowait()
    (output,) = outputs.outputs
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ABORT
    assert output.kv_transfer_params is None
    assert core.output_queue.empty()
    assert request.status == RequestStatus.FINISHED_ABORTED
    scheduler.update_from_output(scheduler.schedule(), EMPTY_MODEL_RUNNER_OUTPUT)
    assert_scheduler_empty(scheduler)


@pytest.mark.parametrize(
    ("use_v2_model_runner", "error"),
    [("0", "requires Model Runner V2"), ("1", "requires mamba_cache_mode='none'")],
    ids=["mrv1", "align"],
)
def test_migrate_rejects_unsupported_hybrid_config_before_pausing(
    monkeypatch, use_v2_model_runner, error
):
    """A hybrid model with speculative decoding can be migrated only where the
    workers can commit its recurrent state: MRV2 without prefix caching (which
    selects the "align" cache mode). Other configurations fail before the
    engine is paused or any request finishes."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", use_v2_model_runner)
    vllm_config = create_vllm_config(
        model="tiny-random/qwen3-next-moe", num_speculative_tokens=1
    )
    assert vllm_config.model_config.is_hybrid
    assert vllm_config.cache_config.mamba_cache_mode == "align"
    scheduler = create_scheduler(vllm_config)
    request = _decode_one_request(vllm_config, scheduler)
    workers = _Workers()
    core = _engine_core(vllm_config, scheduler, workers)

    with pytest.raises(ValueError, match=error):
        core.pause_generation(mode="migrate", clear_cache=False)

    assert scheduler.pause_state == PauseState.UNPAUSED
    assert not core.reject_new_requests
    assert request.status == RequestStatus.RUNNING
    assert workers.calls == [] and core.output_queue.empty()
