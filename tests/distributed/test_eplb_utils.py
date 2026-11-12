# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import Future
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import ParallelConfig
from vllm.distributed.eplb import async_worker, eplb_state
from vllm.distributed.eplb.async_worker import AsyncEplbExecutor, AsyncEplbRunner
from vllm.distributed.eplb.eplb_state import (
    EplbState,
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.sentinel.eplb_redistribute import reset_eplb_async_state

from .eplb_utils import (
    ASYNC_JOB_TIMEOUT_S,
    async_cycle_running,
    finish_async_cycle,
    make_async_eplb_state,
    start_async_rearrangement,
)


def _make_model_state(
    phy2log: torch.Tensor,
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
    phy2log_storage: torch.Tensor | None = None,
) -> MagicMock:
    """Build a minimal EplbModelState mock with its map tensors."""
    state = MagicMock()
    state.physical_to_logical_map = phy2log
    state.physical_to_logical_map_buffer = (
        phy2log if phy2log_storage is None else phy2log_storage
    )
    state.logical_to_physical_map = log2phy
    state.logical_replica_count = logcnt
    return state


def test_commit_eplb_maps_shape_change():
    """When the number of physical experts changes, resize the active map within its
    preallocated storage.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    # Build current state tensors
    storage = torch.full((num_layers, num_physical + 2), -1, dtype=torch.long)
    model_state = _make_model_state(
        phy2log=storage[:, :num_physical],
        phy2log_storage=storage,
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    # The new map has two more physical experts. These new physical experts will
    # automatically map to the first two logical experts
    new_phy2log_larger = (
        (torch.arange(num_physical + 2, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
    )
    _commit_eplb_maps(model_state, new_phy2log_larger)

    # Check that the number of physical experts has been updated and that the values
    # match
    assert model_state.physical_to_logical_map.shape[1] == num_physical + 2
    assert model_state.physical_to_logical_map.data_ptr() == storage.data_ptr()
    assert torch.equal(model_state.physical_to_logical_map, new_phy2log_larger)


def test_commit_eplb_maps_for_layer_logical_padding():
    """Test that logical_to_physical_map is padded with -1 to fill the
    pre-allocated slots when the new map has fewer replicas than the max.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = (
        (torch.arange(num_physical, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
        .contiguous()
    )
    layer = 0
    _commit_eplb_maps_for_layer(model_state, new_phy2log[layer], layer)

    assert torch.all(model_state.logical_to_physical_map[layer, :, 2] == -1)


def test_commit_eplb_maps_for_layer_shape_assert():
    """Test that a mismatched number of physical experts triggers an assertion error."""
    num_layers, num_logical, num_physical = 2, 4, 6

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full((num_layers, num_logical, 2), -1, dtype=torch.long),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )
    bad_phy2log = torch.zeros(num_layers, num_physical + 1, dtype=torch.long)
    with pytest.raises(AssertionError):
        _commit_eplb_maps_for_layer(model_state, bad_phy2log, layer=0)


def test_commit_eplb_maps():
    """Test that all values are copied correctly into model_state."""
    num_layers, num_logical, num_physical, max_replicas = 2, 3, 4, 2

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps(model_state, new_phy2log)

    assert torch.equal(model_state.physical_to_logical_map, new_phy2log)
    assert torch.equal(model_state.logical_to_physical_map, new_log2phy)
    assert torch.equal(model_state.logical_replica_count, new_logcnt)


def test_commit_eplb_maps_for_layer():
    """Test that only the target layer is updated"""
    num_layers, num_logical, max_replicas = 2, 3, 2

    original_phy2log = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)
    model_state = _make_model_state(
        phy2log=original_phy2log.clone(),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps_for_layer(model_state, new_phy2log[0], layer=0)

    # Layer 0 updated
    assert torch.equal(model_state.physical_to_logical_map[0], new_phy2log[0])
    assert torch.equal(model_state.logical_to_physical_map[0], new_log2phy[0])
    assert torch.equal(model_state.logical_replica_count[0], new_logcnt[0])

    # Layer 1 untouched
    assert torch.equal(model_state.physical_to_logical_map[1], original_phy2log[1])


def _overrun_warning_worker(rank: int, world_size: int, port: int):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    ep_group = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=0,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name="eplb_overrun_warning_test",
    )
    # A rank that skips the collective fails the test instead of hanging it.
    ep_group.cpu_timeout = timedelta(seconds=30)

    interval = 4
    model_state = MagicMock()
    model_state.model_name = "m"
    model_state.model.num_moe_layers = 3
    model_state.num_committed_layers = 1
    state = EplbState(ParallelConfig(), torch.device("cpu"))
    state.is_async = True
    state.expert_rearrangement_step_interval = interval
    state.expert_rearrangement_step = interval - 1
    state.model_states = {"m": model_state}
    with patch.object(async_worker, "get_ep_group", return_value=ep_group):
        runner = AsyncEplbRunner(0, MagicMock(), MagicMock(), False)
    # Rank 0 has finished the transfer of layer 1; rank 1 is still in it.
    runner._layers_left.append((model_state, 1))
    runner._job = Future()
    if rank == 0:
        runner._job.set_result(None)
    state.async_runner = runner

    with (
        patch.object(eplb_state, "get_ep_group", return_value=ep_group),
        patch.object(async_worker, "get_ep_group", return_value=ep_group),
        patch.object(eplb_state.logger, "warning") as warning,
    ):
        state.step()

    if rank == 0:
        warning.assert_called_once()
        message = warning.call_args.args[0] % warning.call_args.args[1:]
        assert message.endswith("slowest EP ranks [1]: transferring layer 1")
    else:
        warning.assert_not_called()
    dist.destroy_process_group()


def test_async_rearrangement_overrun_warning_names_slowest_rank():
    """When an async cycle overruns step_interval, every EP rank must enter
    the progress all-reduce, and only rank 0 logs the rank that is behind.
    """
    mp.spawn(_overrun_warning_worker, args=(2, get_open_port()), nprocs=2, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_commit_eplb_maps_for_layer_does_not_sync_host():
    """Async EPLB commits a layer from the main thread every step, so the commit
    must not synchronize the host with the GPU."""
    num_layers, num_logical, max_replicas = 2, 3, 2
    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, 4, dtype=torch.long, device="cuda"),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas),
            -1,
            dtype=torch.long,
            device="cuda",
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long, device="cuda"),
    )
    new_phy2log = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    prev_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _commit_eplb_maps_for_layer(model_state, new_phy2log, layer=0)
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)


@pytest.fixture
def dist_init_backend() -> str:
    # rearrange() all-reduces device tensors, which the gloo fallback that
    # dist_init uses on XPU cannot do.
    return current_platform.dist_backend


@pytest.fixture
def async_eplb(dist_init, monkeypatch):
    return make_async_eplb_state(
        monkeypatch, torch.device(current_platform.device_type), num_models=2
    )


def test_async_executor_runs_jobs_on_a_daemon_thread():
    """The worker thread must not block interpreter exit when a job hangs."""
    if not torch.accelerator.is_available():
        pytest.skip("Async EPLB needs an accelerator")
    executor = AsyncEplbExecutor(torch.accelerator.current_device_index(), rank=0)
    assert executor.thread.daemon

    stream = executor.submit(lambda stream: stream).result(ASYNC_JOB_TIMEOUT_S)
    assert isinstance(stream, torch.Stream)

    error = RuntimeError("job failed")

    def failing_job(stream):
        raise error

    assert executor.submit(failing_job).exception(ASYNC_JOB_TIMEOUT_S) is error


def test_async_eplb_applies_every_layer_of_every_model_in_order(async_eplb):
    run = async_eplb
    start_async_rearrangement(run.state)
    finish_async_cycle(run.state)

    expected = [(m, layer) for m in range(2) for layer in range(2)]
    assert run.transfers == expected
    assert [meta[1] for meta in run.applied] == expected
    for model_state in run.model_states:
        expected_map = torch.arange(4).roll(1).repeat(2, 1)
        assert torch.equal(model_state.physical_to_logical_map.cpu(), expected_map)


def test_async_rearrangement_interval_starts_after_last_layer_commit(async_eplb):
    """An async cycle longer than step_interval must not start the next
    rearrangement in the step its last layer commits.
    """
    state = async_eplb.state
    start_async_rearrangement(state)
    # Shorter than the cycle, which applies one of its four layers per step.
    state.expert_rearrangement_step_interval = 2
    finish_async_cycle(state)
    assert state.expert_rearrangement_step == 0

    state.step()
    assert not async_cycle_running(state)
    state.step()
    assert async_cycle_running(state)
    state.expert_rearrangement_step_interval = 1000
    finish_async_cycle(state)


def test_async_eplb_failure_raises_until_reset(async_eplb):
    """A failed job reaches step() so fault tolerance can react. No new job is
    scheduled until the reset, after which EPLB resumes."""
    run = async_eplb
    run.fail.add((0, 1))
    start_async_rearrangement(run.state)
    with pytest.raises(RuntimeError) as exc_info:
        finish_async_cycle(run.state)
    assert "injected transfer failure" in str(exc_info.value.__cause__)

    with pytest.raises(RuntimeError) as again:
        run.state.step()
    assert again.value is exc_info.value
    assert run.transfers == [(0, 0)]
    # Elastic EP must not drain a cycle on a failed runner either.
    with pytest.raises(RuntimeError) as on_drain:
        run.state.drain_async()
    assert on_drain.value is exc_info.value

    reset_eplb_async_state(SimpleNamespace(eplb_state=run.state))
    start_async_rearrangement(run.state)
    finish_async_cycle(run.state)
    assert run.transfers[-4:] == [(m, layer) for m in range(2) for layer in range(2)]


def test_reset_waits_for_the_in_flight_job(async_eplb):
    """Fault recovery must not rebuild the groups under a running job, so the
    reset returns only after the job ended. Its result is never applied."""
    run = async_eplb
    gate = threading.Event()
    run.gates[(0, 0)] = gate
    start_async_rearrangement(run.state)  # Submits the gated transfer.

    release = threading.Timer(0.2, gate.set)
    release.start()
    reset_eplb_async_state(SimpleNamespace(eplb_state=run.state))
    assert gate.is_set()
    assert run.transfers == [(0, 0)]
    assert not async_cycle_running(run.state)

    del run.gates[(0, 0)]
    start_async_rearrangement(run.state)
    finish_async_cycle(run.state)

    dropped_transfer = 1
    assert run.applied[0][2] != dropped_transfer
    assert [meta[1] for meta in run.applied] == [
        (m, layer) for m in range(2) for layer in range(2)
    ]


def test_reset_refuses_a_running_job(async_eplb):
    """FT recovery must not reset EPLB state while a transfer still runs.

    A job blocked on a failed peer ends late with an error. If the reset had
    already cleared the error by then, the recovered engine would fault again.
    """
    run = async_eplb
    run.state.parallel_config.cpu_distributed_timeout_seconds = 0
    gate = threading.Event()
    run.gates[(0, 0)] = gate
    run.fail.add((0, 0))
    start_async_rearrangement(run.state)

    runner = SimpleNamespace(eplb_state=run.state)
    with pytest.raises(RuntimeError, match="still in a transfer job"):
        reset_eplb_async_state(runner)

    del run.gates[(0, 0)]
    gate.set()
    run.state.parallel_config.cpu_distributed_timeout_seconds = ASYNC_JOB_TIMEOUT_S
    reset_eplb_async_state(runner)
    start_async_rearrangement(run.state)
    finish_async_cycle(run.state)
    assert run.transfers == [(m, layer) for m in range(2) for layer in range(2)]


def test_drain_runs_the_remaining_transfers_and_applies_none(async_eplb):
    """Elastic EP follows a drain with a full rearrange, so the drained layers
    are transferred to keep the ranks in lockstep but not applied."""
    run = async_eplb
    start_async_rearrangement(run.state)

    run.state.drain_async()
    assert not async_cycle_running(run.state)
    assert run.transfers == [(m, layer) for m in range(2) for layer in range(2)]
    assert run.applied == []
