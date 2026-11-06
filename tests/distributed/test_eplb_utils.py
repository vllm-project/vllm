# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.utils import ensure_current_vllm_config
from vllm.config import ParallelConfig
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.eplb import async_worker
from vllm.distributed.eplb.eplb_state import (
    EplbState,
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
)
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
def eplb_dist_init(tmp_path):
    """Like ``dist_init``, but with the platform's device backend (XCCL on XPU).

    ``dist_init`` always requests NCCL, which falls back to gloo on XPU; it is
    shared by many Intel CI tests, so it is left unchanged.
    """
    with ensure_current_vllm_config():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp_path / 'dist_init'}",
            local_rank=0,
            backend=current_platform.dist_backend,
        )
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()


def _start_ft_async_eplb_with_failing_worker(
    monkeypatch, reset_during_transfer: bool = False, start: bool = True
):
    """Start single-rank async EPLB under fault tolerance whose worker fails
    every layer transfer. On one rank the EP group stands in for the EPLB group.
    With ``reset_during_transfer``, an FT reset lands before each failure.
    """
    num_layers, num_experts, interval = 2, 4, 2
    device = torch.device(current_platform.device_type)
    monkeypatch.setattr(async_worker, "get_eplb_group", get_ep_group)
    monkeypatch.setattr(
        async_worker,
        "run_rebalance_experts",
        lambda model_state, eplb_state, phy2log_cpu, stream: phy2log_cpu.clone(),
    )
    failures: list[Exception] = []

    def failing_transfer_layer(**kwargs):
        if reset_during_transfer:
            state.ft_reset_epoch += 1
        failures.append(RuntimeError(f"injected transfer failure {len(failures)}"))
        raise failures[-1]

    monkeypatch.setattr(async_worker, "transfer_layer", failing_transfer_layer)

    phy2log = torch.arange(num_experts, device=device).repeat(num_layers, 1)
    model_state = _make_model_state(
        phy2log=phy2log,
        log2phy=phy2log.unsqueeze(-1),
        logcnt=torch.ones(num_layers, num_experts, dtype=torch.long, device=device),
    )
    model_state.model.num_moe_layers = num_layers
    model_state.model.num_logical_experts = num_experts
    model_state.rebalanced = False
    model_state.pending_result = None
    model_state.expert_load_pass = torch.zeros(
        num_layers, num_experts, dtype=torch.int32, device=device
    )
    model_state.expert_load_window = torch.zeros(
        interval, num_layers, num_experts, dtype=torch.int32, device=device
    )

    parallel_config = ParallelConfig()
    parallel_config.enable_fault_tolerance = True
    state = EplbState(parallel_config, device)
    if state.device_index is None:
        pytest.skip("async EPLB needs a CUDA or XPU device")
    state.is_async = True
    state.expert_load_window_size = interval
    state.expert_rearrangement_step_interval = interval
    state.expert_rearrangement_step = interval - 1
    state.model_states = {"m": model_state}
    if start:
        state.start_async_loop()
    return state, model_state, failures


def _step_until_raise(state: EplbState, max_steps: int = 1000) -> Exception | None:
    for _ in range(max_steps):
        try:
            state.step()
        except RuntimeError as exc:
            return exc
        time.sleep(0.01)
    return None


def test_async_worker_failure_raises_under_fault_tolerance(eplb_dist_init, monkeypatch):
    """Under fault tolerance the async worker survives an exception, but the
    failure must still reach step() so that FT can recover. Otherwise EPLB
    stalls silently with every rank waiting for the failed layer.
    """
    state, _, failures = _start_ft_async_eplb_with_failing_worker(monkeypatch)

    exc = _step_until_raise(state)

    assert exc is not None, "a worker failure under FT never reached step()"
    assert exc.__cause__ is failures[0]
    assert state.async_worker.is_alive()


def test_fault_tolerance_reset_rearms_async_eplb(eplb_dist_init, monkeypatch):
    """After an FT reset, the surviving worker runs the next rearrangement."""
    state, model_state, failures = _start_ft_async_eplb_with_failing_worker(monkeypatch)
    assert _step_until_raise(state) is not None

    reset_eplb_async_state(SimpleNamespace(eplb_state=state))
    assert state.async_worker_error is None
    assert not model_state.rebalanced

    state.expert_rearrangement_step = state.expert_rearrangement_step_interval - 1
    exc = _step_until_raise(state)

    assert exc is not None
    assert exc.__cause__ is failures[1]


def test_fault_tolerance_drops_failure_from_abandoned_cycle(
    eplb_dist_init, monkeypatch
):
    """A failure from a cycle that an FT reset abandoned is stale, so it must
    not fail the rearrangement that runs after recovery.
    """
    handled = threading.Event()
    monkeypatch.setattr(
        async_worker.logger, "exception", lambda *args, **kwargs: handled.set()
    )
    state, _, failures = _start_ft_async_eplb_with_failing_worker(
        monkeypatch, reset_during_transfer=True
    )

    for _ in range(1000):
        state.step()
        if handled.wait(timeout=0.01):
            break

    assert failures, "the injected transfer failure never ran"
    assert state.async_worker_error is None
    state.step()


def test_fault_tolerance_drops_rearrangement_signalled_before_reset(
    eplb_dist_init, monkeypatch
):
    """A rearrangement signalled before an FT reset, but picked up by the
    worker after it, belongs to the abandoned epoch and must not run.
    """
    state, _, _ = _start_ft_async_eplb_with_failing_worker(monkeypatch, start=False)
    cycle_started, release = threading.Event(), threading.Event()
    cycle_epochs: list[int] = []

    def blocking_cycle(state, stream, is_profile, reset_epoch):
        cycle_epochs.append(reset_epoch)
        cycle_started.set()
        release.wait(timeout=30)

    monkeypatch.setattr(async_worker, "_transfer_cycle", blocking_cycle)
    num_waits = 0
    real_wait = state.rearrange_event.wait

    def counting_wait(*args, **kwargs):
        nonlocal num_waits
        num_waits += 1
        real_wait(*args, **kwargs)

    monkeypatch.setattr(state.rearrange_event, "wait", counting_wait)
    state.start_async_loop()

    state.rearrange_event.record()
    assert cycle_started.wait(timeout=30)
    # Queued while the worker is busy, then an FT reset lands before pickup.
    state.rearrange_event.record()
    state.ft_reset_epoch += 1
    release.set()

    deadline = time.monotonic() + 30
    while num_waits < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert num_waits == 3, "the worker never went back to waiting"
    assert cycle_epochs == [0]
