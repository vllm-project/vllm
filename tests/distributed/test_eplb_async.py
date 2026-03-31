# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest import mock

import torch

from vllm.distributed.eplb.async_worker import transfer_run_periodically
from vllm.distributed.eplb.eplb_state import (
    EplbModelState,
    EplbStats,
    _move_to_workspace,
)
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent


def make_model_state(physical_to_logical_map: torch.Tensor) -> EplbModelState:
    """Build a minimal EplbModelState with the given physical_to_logical_map."""
    num_layers, num_physical = physical_to_logical_map.shape
    model = SimpleNamespace(
        num_moe_layers=num_layers,
        expert_weights=[[torch.empty(1, device="cuda")] for _ in range(num_layers)],
        model_name="test",
    )
    return EplbModelState(
        physical_to_logical_map=physical_to_logical_map.clone(),
        logical_to_physical_map=torch.zeros(
            num_layers, num_physical, 1, dtype=torch.long, device="cuda"
        ),
        logical_replica_count=torch.ones(
            num_layers, num_physical, dtype=torch.long, device="cuda"
        ),
        expert_load_pass=torch.zeros(num_layers, num_physical, device="cuda"),
        expert_load_window=torch.zeros(1, num_layers, num_physical, device="cuda"),
        model_name="test",
        model=model,
        expert_buffer=[torch.empty(1, device="cuda") for _ in range(num_layers)],
        rebalanced=False,
        eplb_stats=None,
        cuda_device_index=torch.accelerator.current_device_index(),
    )


def make_eplb_stats(num_layers: int, num_physical: int) -> EplbStats:
    return EplbStats(
        global_expert_load_window=torch.zeros(
            1, num_layers, num_physical, device="cuda"
        ),
        num_replicas=1,
        num_groups=1,
        num_nodes=1,
        num_gpus=1,
    )


def make_eplb_state(model_state: EplbModelState) -> SimpleNamespace:
    """Build a minimal EplbState-like namespace with an identity rebalance policy."""
    return SimpleNamespace(
        rearrange_event=CpuGpuEvent(),
        is_async=True,
        model_states={"model": model_state},
        # Identity policy: return the current map unchanged.
        policy=SimpleNamespace(
            rebalance_experts=lambda load, nr, ng, nn, ngpu, p2l: p2l.clone()
        ),
    )


def start_worker_thread(eplb_state: SimpleNamespace) -> threading.Thread:
    """Run transfer_run_periodically in a daemon thread."""

    def run() -> None:
        stream = torch.cuda.Stream()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            transfer_run_periodically(
                state=eplb_state,
                eplb_group=None,  # not used — transfer_layer is mocked
                cuda_stream=stream,
            )
        )

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_worker_snapshots_map_and_load_window_at_wake_time():
    NUM_LAYERS = 1
    NUM_PHYSICAL_EXPERTS = 4

    p2l_v1 = torch.arange(NUM_PHYSICAL_EXPERTS, device="cuda").unsqueeze(0)
    p2l_v2 = torch.tensor([[3, 2, 1, 0]], device="cuda")

    # A recognisable non-zero load window to distinguish from the default zeros.
    load_window = torch.full((1, NUM_LAYERS, NUM_PHYSICAL_EXPERTS), 42.0, device="cuda")

    model_state = make_model_state(p2l_v1)
    eplb_state = make_eplb_state(model_state)

    captured_old_indices = []
    captured_load_windows = []
    transfer_called = threading.Event()

    def capturing_rebalance(load, nr, ng, nn, ngpu, p2l):
        captured_load_windows.append(load.clone())
        return p2l.clone()

    eplb_state.policy.rebalance_experts = capturing_rebalance

    async def mock_transfer(**kwargs):
        captured_old_indices.append(kwargs["old_layer_indices"].clone())
        transfer_called.set()
        return mock.MagicMock(), mock.MagicMock(), mock.MagicMock()

    with mock.patch("vllm.distributed.eplb.async_worker.transfer_layer", mock_transfer):
        start_worker_thread(eplb_state)
        time.sleep(5)

        # Update the map to V2 and set a recognisable load window before
        # unblocking the async worker.
        model_state.physical_to_logical_map.copy_(p2l_v2)
        model_state.eplb_stats = make_eplb_stats(NUM_LAYERS, NUM_PHYSICAL_EXPERTS)
        model_state.eplb_stats.global_expert_load_window = load_window
        model_state.rebalanced = True
        eplb_state.rearrange_event.record()

        assert transfer_called.wait(timeout=5.0), "transfer_layer was not called"

    assert torch.equal(captured_old_indices[0], p2l_v2[0].cpu())
    assert torch.equal(captured_load_windows[0], load_window.cpu())


def test_consumed_event_handshake():
    """
    The worker blocks on consumed_event.wait() after publishing pending_result.
    _move_to_workspace must record consumed_event to unblock it.
    """
    NUM_LAYERS = 1
    NUM_PHYSICAL_EXPERTS = 4

    p2l = torch.arange(NUM_PHYSICAL_EXPERTS, device="cuda").unsqueeze(0)
    model_state = make_model_state(p2l)
    eplb_state = make_eplb_state(model_state)
    ep_group = SimpleNamespace(rank=lambda: 0)

    async def mock_transfer(**kwargs):
        return mock.MagicMock(), mock.MagicMock(), mock.MagicMock()

    with (
        mock.patch("vllm.distributed.eplb.async_worker.transfer_layer", mock_transfer),
        mock.patch("vllm.distributed.eplb.eplb_state.move_from_buffer"),
    ):
        start_worker_thread(eplb_state)
        time.sleep(5)

        model_state.eplb_stats = make_eplb_stats(NUM_LAYERS, NUM_PHYSICAL_EXPERTS)
        model_state.rebalanced = True
        eplb_state.rearrange_event.record()

        # Poll until the worker publishes pending_result.
        deadline = time.monotonic() + 5.0
        while model_state.pending_result is None:
            assert time.monotonic() < deadline, "worker never published pending_result"
            time.sleep(0.001)

        # Worker is now blocked on consumed_event.wait() — event not yet recorded.
        assert not model_state.pending_result.consumed_event._recorded.is_set()

        _move_to_workspace(model_state, ep_group)

    assert model_state.pending_result is None
