# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue as py_queue
from collections import deque
from concurrent.futures import Future
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import CheckpointState
from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState
from vllm.v1.outputs import ModelRunnerOutput


def _make_engine_with_kv_obligation(
    cls: type[EngineCore], *, engines_running: bool = False
) -> tuple[Any, Scheduler]:
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=64,
        num_blocks=5,
        block_size=16,
        enable_prefix_caching=True,
        deterministic_kv_reserve=True,
    )
    assert isinstance(scheduler, Scheduler)
    scheduler.vllm_config.model_config.enable_sleep_mode = True
    core: Any = object.__new__(cls)
    core.vllm_config = scheduler.vllm_config
    core.scheduler = scheduler
    core.model_executor = MagicMock()
    core.model_executor.vllm_config.cache_config.enable_extensible_kv_cache = False
    core.model_executor.is_sleeping = False
    core._checkpoint_state = CheckpointState.RUNNING
    core._checkpoint_error = None
    core._kv_cache_reclaimable = False
    core.batch_queue = None
    core.mm_receiver_cache = None
    core.reject_new_requests = False
    if cls is EngineCoreProc:
        core.engines_running = engines_running
        core._idle_state_callbacks = []
        core.output_queue = MagicMock()
    return core, scheduler


def test_engine_core_kv_obligation_pause_and_release_contract() -> None:
    core, scheduler = _make_engine_with_kv_obligation(EngineCore)
    (req,) = create_requests(
        num_requests=1, num_tokens=16, max_tokens=2, ignore_eos=True, req_ids=["r_keep"]
    )
    scheduler.add_request(req)
    step = scheduler.schedule()
    assert list(step.num_scheduled_tokens) == ["r_keep"]

    with pytest.raises(RuntimeError, match="deterministic KV reservations"):
        core.pause_scheduler(mode="keep", clear_cache=True)
    assert scheduler.pause_state == PauseState.UNPAUSED
    assert scheduler.running == [req]

    assert core.pause_scheduler(mode="keep", clear_cache=False) is None
    assert scheduler.pause_state == PauseState.PAUSED_ALL
    with pytest.raises(RuntimeError, match="deterministic KV reservations"):
        core.release_kv_cache_memory()
    core.model_executor.discard.assert_not_called()
    core.resume_scheduler()
    assert scheduler.pause_state == PauseState.UNPAUSED

    for _ in range(2):
        scheduler.update_from_output(
            step, ModelRunnerOutput(req_ids=["r_keep"], sampled_token_ids=[[0]])
        )
        if scheduler.has_unfinished_requests():
            step = scheduler.schedule()
    assert not scheduler.has_unfinished_requests()
    assert req.num_preemptions == 0
    if scheduler.has_requests():
        scheduler.schedule()
    assert not scheduler.has_requests()

    assert core.pause_scheduler(mode="keep", clear_cache=False) is None
    core.release_kv_cache_memory()
    core.model_executor.discard.assert_called_once_with(("kv_cache",))
    core.resume_scheduler()
    assert scheduler.pause_state == PauseState.UNPAUSED


def test_engine_core_proc_kv_obligation_pause_checkpoint_and_liveness() -> None:
    proc, proc_sched = _make_engine_with_kv_obligation(
        EngineCoreProc, engines_running=True
    )
    proc._deferred_inputs = deque()
    proc.input_queue = py_queue.Queue()
    proc.aborts_queue = py_queue.Queue()
    proc.process_input_queue_block = True
    proc.shutdown_state = EngineShutdownState.RUNNING

    def _fail_blocking_get(*args: object, **kwargs: object) -> object:
        if kwargs.get("block"):
            raise AssertionError("input_queue.get(block=True) reached with has_work")
        raise py_queue.Empty

    proc.input_queue.get = _fail_blocking_get
    (r_proc,) = create_requests(
        num_requests=1, num_tokens=16, max_tokens=2, ignore_eos=True, req_ids=["r_proc"]
    )
    proc_sched.add_request(r_proc)
    proc_sched.schedule()

    with pytest.raises(RuntimeError, match="deterministic KV reservations"):
        proc.pause_scheduler(mode="keep", clear_cache=True)
    assert proc_sched.pause_state == PauseState.UNPAUSED
    assert proc._idle_state_callbacks == []
    assert proc_sched.running == [r_proc]

    refusal = proc.checkpoint_prepare(mode="keep")
    assert refusal is not None and "deterministic KV reservations" in refusal
    assert proc.checkpoint_state() == (CheckpointState.RUNNING, None)
    assert proc_sched.pause_state == PauseState.UNPAUSED
    assert not proc.reject_new_requests

    keep_future = proc.pause_scheduler(mode="keep", clear_cache=False)
    assert isinstance(keep_future, Future)
    assert proc_sched.pause_state == PauseState.PAUSED_ALL
    with pytest.raises(RuntimeError, match="deterministic KV reservations"):
        proc.pause_scheduler(mode="keep", clear_cache=True)
    assert proc_sched.pause_state == PauseState.PAUSED_ALL
    assert len(proc._idle_state_callbacks) == 1

    proc.engines_running = False
    proc._notify_idle_state_callbacks()
    assert keep_future.result(timeout=0) is None
    assert proc_sched.pause_state == PauseState.PAUSED_ALL
    proc.resume_scheduler()
    assert proc_sched.pause_state == PauseState.UNPAUSED

    proc._idle_state_callbacks.append(lambda engine: engine.resume_scheduler())
    proc_sched.set_pause_state(PauseState.PAUSED_ALL)
    assert not proc.has_work()
    proc._process_input_queue()
    assert proc_sched.pause_state == PauseState.UNPAUSED
    assert proc.has_work()

    abort_future = proc.pause_scheduler(mode="abort", clear_cache=True)
    assert isinstance(abort_future, Future)
    assert proc_sched.pause_state == PauseState.PAUSED_NEW
    assert not proc_sched.has_unfinished_requests()
    proc_sched.schedule()
    proc._notify_idle_state_callbacks()
    assert abort_future.result(timeout=0) is None
    proc.resume_scheduler()
    assert proc_sched.pause_state == PauseState.UNPAUSED

    (r_wait,) = create_requests(
        num_requests=1, num_tokens=16, max_tokens=1, ignore_eos=True, req_ids=["r_wait"]
    )
    proc_sched.add_request(r_wait)
    w_step = proc_sched.schedule()
    wait_future = proc.pause_scheduler(mode="wait", clear_cache=True)
    assert isinstance(wait_future, Future)
    proc_sched.update_from_output(
        w_step, ModelRunnerOutput(req_ids=["r_wait"], sampled_token_ids=[[0]])
    )
    if proc_sched.has_requests():
        proc_sched.schedule()
    assert not proc.has_work()
    proc.process_input_queue_block = False
    proc._process_input_queue()
    assert wait_future.result(timeout=0) is None
    proc.resume_scheduler()
    assert proc_sched.pause_state == PauseState.UNPAUSED
