# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.eplb.async_worker import transfer_run_periodically
from vllm.distributed.eplb.eplb_state import (
    EplbState,
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
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
    """
    When the number of physical experts changes, resize the active map within its
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
    """
    Test that logical_to_physical_map is padded with -1 to fill the
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


def test_stop_async_loop_joins_worker_before_return():
    state = EplbState.__new__(EplbState)
    state.is_async = True
    state.model_states = {}
    state.async_worker_stop_event = threading.Event()
    worker_started = threading.Event()
    worker_exited = threading.Event()

    def worker_target():
        worker_started.set()
        state.async_worker_stop_event.wait()
        worker_exited.set()

    worker = threading.Thread(target=worker_target)
    state.async_worker = worker
    worker.start()
    assert worker_started.wait(timeout=1.0)

    # Call stop_async_loop() and ensure that it waits for the worker to exit
    state.stop_async_loop()

    assert worker_exited.is_set()
    assert not worker.is_alive()
    assert state.async_worker is None


def test_stop_async_loop_prevents_group_access_during_teardown():
    wait_started = threading.Event()
    release_wait = threading.Event()
    group_destroyed = threading.Event()
    worker_errors: list[Exception] = []
    group_accesses_after_destroy: list[str] = []

    class ControlledRearrangeEvent:
        def wait(self, *, stream, stop_event):
            wait_started.set()
            assert release_wait.wait(timeout=1.0)
            return True

    class TeardownGroup:
        def rank(self):
            if group_destroyed.is_set():
                group_accesses_after_destroy.append("rank")
            return 0

    state = EplbState.__new__(EplbState)
    state.is_async = True
    state.model_states = {}
    state.rearrange_event = ControlledRearrangeEvent()
    state.async_worker_stop_event = threading.Event()
    eplb_group = TeardownGroup()

    def worker_target():
        try:
            transfer_run_periodically(
                state=state,
                cuda_stream=None,
                stop_event=state.async_worker_stop_event,
                eplb_group=eplb_group,
                eplb_cpu_group=eplb_group,
            )
        except Exception as exc:
            worker_errors.append(exc)

    worker = threading.Thread(target=worker_target)
    state.async_worker = worker
    worker.start()
    assert wait_started.wait(timeout=1.0)

    shutdown = threading.Thread(target=state.stop_async_loop)
    shutdown.start()
    assert state.async_worker_stop_event.wait(timeout=1.0)

    group_destroyed.set()
    release_wait.set()
    shutdown.join(timeout=1.0)

    assert not shutdown.is_alive()
    assert not worker.is_alive()
    assert state.async_worker is None
    assert worker_errors == []
    assert group_accesses_after_destroy == []


def test_stop_async_loop_is_idempotent():
    state = EplbState.__new__(EplbState)
    state.async_worker = MagicMock()
    state.async_worker_stop_event = threading.Event()
    state.drain_async = MagicMock()
    worker = state.async_worker

    state.stop_async_loop()
    state.stop_async_loop()

    state.drain_async.assert_called_once_with()
    worker.join.assert_called_once_with()
    assert state.async_worker_stop_event.is_set()
    assert state.async_worker is None


def test_async_loop_can_restart_after_stop(monkeypatch):
    state = EplbState.__new__(EplbState)
    state.is_async = True
    state.model_states = {}
    state.async_worker_stop_event = threading.Event()
    worker_started = threading.Event()

    def worker_target():
        worker_started.set()
        state.async_worker_stop_event.wait()

    old_worker = threading.Thread(target=worker_target)
    state.async_worker = old_worker
    old_worker.start()
    assert worker_started.wait(timeout=1.0)

    state.stop_async_loop()
    assert not old_worker.is_alive()
    assert state.async_worker_stop_event.is_set()

    new_worker = MagicMock()

    def fake_start_async_worker(started_state, stop_event, is_profile):
        assert started_state is state
        assert stop_event is state.async_worker_stop_event
        assert not stop_event.is_set()
        assert is_profile is False
        return new_worker

    monkeypatch.setattr(
        "vllm.distributed.eplb.eplb_state.start_async_worker",
        fake_start_async_worker,
    )

    state.start_async_loop()

    assert state.async_worker is new_worker


def test_stop_async_loop_drains_inflight_transfer_before_stop():
    calls: list[str] = []
    state = EplbState.__new__(EplbState)
    state.async_worker_stop_event = threading.Event()
    state.async_worker = MagicMock()

    def drain_async():
        assert not state.async_worker_stop_event.is_set()
        calls.append("drain")

    def join_worker():
        assert state.async_worker_stop_event.is_set()
        calls.append("join")

    state.drain_async = drain_async
    state.async_worker.join.side_effect = join_worker

    state.stop_async_loop()

    assert calls == ["drain", "join"]
    assert state.async_worker is None
