# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.distributed.eplb import async_worker
from vllm.distributed.eplb import eplb_state as eplb_state_module
from vllm.distributed.eplb.eplb_state import (
    EplbState,
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
)
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.distributed.parallel_state import get_ep_group
from vllm.platforms import current_platform
from vllm.v1.worker.sentinel.eplb_redistribute import reset_eplb_async_state


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


class _WorkerProbe:
    """Reports each point where the async worker blocks, so a test can advance
    in lockstep with it instead of sleeping or polling. The tests then do not
    depend on thread timing, which keeps them from being flaky.

    The worker reports "idle" before waiting for a rearrange signal and
    "published" before waiting for the main thread to consume a layer result.
    Everything the worker did before a report happens-before the test reads it.
    """

    # Only turns a deadlock into a failure; a healthy run never waits this long.
    HANG_TIMEOUT_S = 60

    def __init__(self) -> None:
        self._reports: queue.Queue[str] = queue.Queue()

    def event(self, report: str) -> CpuGpuEvent:
        probe = self

        class _ReportingEvent(CpuGpuEvent):
            def wait(self, stream: torch.Stream | None = None) -> None:
                probe._reports.put(report)
                super().wait(stream)

        return _ReportingEvent()

    def expect(self, report: str) -> None:
        assert self._reports.get(timeout=self.HANG_TIMEOUT_S) == report
        assert self._reports.empty()


def _start_ft_async_eplb(monkeypatch):
    """Start single-rank async EPLB under fault tolerance whose first layer
    transfer raises. On one rank the EP group stands in
    for the EPLB group. Returns once the worker is waiting for a signal.
    """
    num_layers, num_experts, window = 2, 4, 2
    device = torch.device(current_platform.device_type)
    probe = _WorkerProbe()
    monkeypatch.setattr(async_worker, "get_eplb_group", get_ep_group)
    monkeypatch.setattr(
        async_worker,
        "run_rebalance_experts",
        lambda model_state, eplb_state, phy2log_cpu, stream: phy2log_cpu.clone(),
    )
    monkeypatch.setattr(async_worker, "CpuGpuEvent", lambda: probe.event("published"))
    failures: list[Exception] = []
    transferred: list[int] = []

    def transfer_layer(*, layer_idx: int, **kwargs) -> None:
        if not failures:
            failures.append(RuntimeError("injected transfer failure"))
            raise failures[-1]
        transferred.append(layer_idx)

    monkeypatch.setattr(async_worker, "transfer_layer", transfer_layer)
    applied: list[int] = []
    monkeypatch.setattr(
        eplb_state_module,
        "move_from_buffer",
        lambda *, expert_weights, **kwargs: applied.append(expert_weights),
    )

    phy2log = torch.arange(num_experts, device=device).repeat(num_layers, 1)
    model_state = _make_model_state(
        phy2log=phy2log,
        log2phy=phy2log.unsqueeze(-1),
        logcnt=torch.ones(num_layers, num_experts, dtype=torch.long, device=device),
    )
    model_state.model.num_moe_layers = num_layers
    model_state.model.num_logical_experts = num_experts
    model_state.model.expert_weights = list(range(num_layers))
    model_state.rebalanced = False
    model_state.pending_result = None
    model_state.expert_load_pass = torch.zeros(
        num_layers, num_experts, dtype=torch.int32, device=device
    )
    model_state.expert_load_window = torch.zeros(
        window, num_layers, num_experts, dtype=torch.int32, device=device
    )

    parallel_config = ParallelConfig()
    parallel_config.enable_fault_tolerance = True
    state = EplbState(parallel_config, device)
    if state.device_index is None:
        pytest.skip("async EPLB needs a CUDA or XPU device")
    state.is_async = True
    state.expert_load_window_size = window
    # Large enough that consuming layer results never triggers another
    # rearrangement; tests trigger one explicitly with _signal_rearrangement.
    state.expert_rearrangement_step_interval = 1000
    state.model_states = {"m": model_state}
    state.rearrange_event = probe.event("idle")
    state.start_async_loop()
    probe.expect("idle")
    return SimpleNamespace(
        state=state,
        model_state=model_state,
        probe=probe,
        failures=failures,
        transferred=transferred,
        applied=applied,
    )


def _signal_rearrangement(state: EplbState) -> None:
    """Run the step that starts a rearrangement and signals the worker."""
    state.expert_rearrangement_step = state.expert_rearrangement_step_interval - 1
    state.step()
    assert state.model_states["m"].rebalanced


def test_fault_tolerance_recovers_async_eplb_after_worker_failure(
    dist_init, monkeypatch
):
    """Under fault tolerance a worker failure must reach step(), or EPLB stalls
    silently with every rank waiting for the failed layer. The worker thread
    survives, so after the FT reset the next rearrangement runs to completion.
    """
    run = _start_ft_async_eplb(monkeypatch)
    _signal_rearrangement(run.state)
    run.probe.expect("idle")
    with pytest.raises(RuntimeError) as exc_info:
        run.state.step()
    assert exc_info.value.__cause__ is run.failures[0]
    assert run.state.async_worker.is_alive()

    reset_eplb_async_state(SimpleNamespace(eplb_state=run.state))
    assert run.state.async_worker_error is None
    assert not run.model_state.rebalanced

    _signal_rearrangement(run.state)
    num_layers = run.model_state.model.num_moe_layers
    for _ in range(num_layers):
        run.probe.expect("published")
        run.state.step()
    run.probe.expect("idle")

    assert run.transferred == list(range(num_layers))
    assert run.applied == list(range(num_layers))
    assert not run.model_state.rebalanced
    assert run.state.async_worker_error is None
    assert len(run.failures) == 1
