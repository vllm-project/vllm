# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from unittest.mock import MagicMock

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import ParallelConfig, SchedulerConfig
from vllm.distributed.utils import stateless_init_torch_distributed_process_group
from vllm.utils.network_utils import get_open_port
from vllm.v1.core.sched.prefill_delayer import (
    MAX_HOLD_STEPS,
    PARTIAL_MAX_HOLD_STEPS,
    STALL_STEPS,
    PrefillDelayer,
    PrefillDelayerState,
)
from vllm.v1.engine.core import DPEngineCoreProc

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

BUDGET = 100
DP_SIZE = 2


def _state(**kwargs) -> PrefillDelayerState:
    """DP-summed state: both ranks prefillable and decoding, underfilled."""
    fields = dict(
        prefillable=DP_SIZE,
        blocked=0,
        unreachable=0,
        pending_tokens=10,
        decodes=4,
        kv_high=0,
        partial=0,
        queue_hot=0,
        prefill_ran=0,
        arrived=0,
    )
    fields.update(kwargs)
    return PrefillDelayerState(**fields)


@pytest.mark.parametrize("interval", [1, 4])
def test_protected_steps_after_prefill(interval: int):
    delayer = PrefillDelayer(DP_SIZE, BUDGET, interval)
    full = _state(pending_tokens=2 * BUDGET)
    for i in range(interval - 1):
        assert delayer.decide(full._replace(prefill_ran=int(i == 0))) == (
            False,
            "protected",
        )
    assert delayer.decide(full._replace(prefill_ran=int(interval == 1))) == (
        True,
        "fill",
    )


@pytest.mark.parametrize(
    "state, reason",
    [
        (_state(prefillable=0, pending_tokens=0), "no_prefill"),
        (_state(decodes=0), "no_decode"),
        (_state(prefillable=1, kv_high=1), "kv_high"),
        (_state(prefillable=1, queue_hot=1), "max_wait"),
        (_state(pending_tokens=int(0.9 * DP_SIZE * BUDGET)), "fill"),
        # A blocked rank counts as aligned.
        (_state(prefillable=1, blocked=1, pending_tokens=BUDGET), "fill"),
    ],
)
def test_fire_rules(state: PrefillDelayerState, reason: str):
    assert PrefillDelayer(DP_SIZE, BUDGET, 1).decide(state) == (True, reason)


@pytest.mark.parametrize(
    "partial, bound, reason",
    [(0, MAX_HOLD_STEPS, "max_hold"), (1, PARTIAL_MAX_HOLD_STEPS, "partial")],
)
def test_unaligned_hold_until_bound(partial: int, bound: int, reason: str):
    """A peer that will finish a decode in time is waited for, even through
    steps in which nothing changes, up to the hold bound."""
    delayer = PrefillDelayer(DP_SIZE, BUDGET, 1)
    state = _state(prefillable=1, partial=partial)
    for _ in range(bound):
        assert delayer.decide(state) == (False, "align")
    assert delayer.decide(state) == (True, reason)


def test_unaligned_stall_fires_only_for_unreachable_peer():
    delayer = PrefillDelayer(DP_SIZE, BUDGET, 1)
    state = _state(prefillable=1, unreachable=1)
    for _ in range(STALL_STEPS):
        assert delayer.decide(state) == (False, "align")
    assert delayer.decide(state) == (True, "stall")


def test_arrivals_restart_stall():
    """Capped pending tokens can stay flat while requests keep arriving."""
    delayer = PrefillDelayer(DP_SIZE, BUDGET, 1)
    for _ in range(3 * STALL_STEPS):
        assert delayer.decide(_state(arrived=1)) == (False, "fill")
    for _ in range(STALL_STEPS - 1):
        assert delayer.decide(_state()) == (False, "fill")
    assert delayer.decide(_state()) == (True, "stall")


def test_reset_ends_hold_episode():
    delayer = PrefillDelayer(DP_SIZE, BUDGET, 1)
    state = _state(prefillable=1)
    for _ in range(MAX_HOLD_STEPS):
        delayer.decide(state)
    delayer.reset()
    assert delayer.decide(state) == (False, "align")
    assert delayer.hold_budget == MAX_HOLD_STEPS - 1


def _delayer_scheduler(**kwargs):
    return create_scheduler(
        data_parallel_size=DP_SIZE,
        prefill_schedule_policy="delayer",
        max_num_batched_tokens=BUDGET,
        **kwargs,
    )


def test_scheduler_state():
    scheduler = _delayer_scheduler()
    assert scheduler.prefill_delayer_enabled
    for request in create_requests(num_requests=2, num_tokens=60, max_tokens=8):
        scheduler.add_request(request)
    state = scheduler.prefill_delayer_state(MAX_HOLD_STEPS)
    assert state.prefillable and not state.blocked and not state.unreachable
    assert state.pending_tokens == BUDGET and state.arrived == 2
    assert state.decodes == 0 and not state.prefill_ran

    output = scheduler.schedule()
    # 60 tokens of the first request and a 40-token chunk of the second.
    assert sum(output.num_scheduled_tokens.values()) == BUDGET
    state = scheduler.prefill_delayer_state(MAX_HOLD_STEPS)
    assert state.prefill_ran and state.partial and state.pending_tokens == 20
    assert state.arrived == 0


def test_scheduler_state_unreachable():
    scheduler = _delayer_scheduler()
    (request,) = create_requests(num_requests=1, num_tokens=8, max_tokens=8)
    scheduler.add_request(request)
    scheduler.schedule()
    scheduler.running[0].is_prefill_chunk = False
    # Its decode finishes within 8 steps: reachable for a longer hold only.
    assert not scheduler.prefill_delayer_state(8).unreachable
    assert scheduler.prefill_delayer_state(7).unreachable


def test_scheduler_hold_defers_only_ranks_with_decodes():
    """A rank with nothing but prefill runs it under a hold: an empty batch
    while its DP peers step can hang the engine."""
    scheduler = _delayer_scheduler()
    first, second = create_requests(num_requests=2, num_tokens=8, req_ids=["a", "b"])
    scheduler.add_request(first)
    output = scheduler.schedule(throttle_prefills=True)
    assert "a" in output.num_scheduled_tokens

    scheduler.running[0].is_prefill_chunk = False
    scheduler.add_request(second)
    output = scheduler.schedule(throttle_prefills=True)
    assert "b" not in output.num_scheduled_tokens


def test_config_rejects_max_wait_without_delayer():
    with pytest.raises(ValueError, match="prefill_max_wait_ms"):
        SchedulerConfig(
            max_num_batched_tokens=8192,
            max_model_len=8192,
            is_encoder_decoder=False,
            prefill_max_wait_ms=100,
        )


NUM_STEPS = 300


def _rank_states(rank: int) -> list[PrefillDelayerState]:
    rng = random.Random(rank)
    return [
        PrefillDelayerState(
            *(rng.randint(0, 1) for _ in range(3)),
            rng.randint(0, BUDGET),
            rng.randint(0, 3),
            *(int(rng.random() < 0.05) for _ in range(3)),
            int(rng.random() < 0.1),
            rng.randint(0, 1),
        )
        for _ in range(NUM_STEPS)
    ]


def _lockstep_worker(rank: int, world_size: int, port: int):
    group = stateless_init_torch_distributed_process_group(
        "127.0.0.1", port, rank, world_size, backend="gloo"
    )
    parallel_config = ParallelConfig()
    delayer = PrefillDelayer(world_size, BUDGET, 3)
    expected_delayer = PrefillDelayer(world_size, BUDGET, 3)
    all_states = [_rank_states(r) for r in range(world_size)]
    for step in range(NUM_STEPS):
        has_unfinished, pause = parallel_config.sync_dp_state(
            group,
            has_unfinished=rank == step % world_size,
            pending_pause=False,
            extra=delayer.load(all_states[rank][step]),
        )
        assert has_unfinished and not pause
        summed = PrefillDelayerState(
            *map(sum, zip(*(states[step] for states in all_states)))
        )
        assert delayer.buffer.tolist() == list(summed)
        assert delayer.step() == expected_delayer.decide(summed)[0]


@pytest.mark.parametrize("world_size", [2, 3])
def test_lockstep_decisions_across_ranks(world_size: int):
    """The delayer state rides on the DP state sync, and every rank takes the
    same decision from the sums."""
    mp.spawn(
        _lockstep_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
        join=True,
    )


def test_buffer_layout():
    delayer = PrefillDelayer(DP_SIZE, BUDGET, 1)
    state = _state(arrived=7)
    assert delayer.load(state).dtype == torch.int32
    assert delayer.buffer.tolist() == list(state)


def test_engine_core_syncs_every_step_with_delayer(monkeypatch):
    """With the delayer, the DP state syncs on every step (not every
    dp_sync_interval steps) and carries the delayer's state."""
    core = object.__new__(DPEngineCoreProc)
    core.enable_fault_tolerance = False
    core.dp_group = object()
    core.vllm_config = MagicMock(parallel_config=ParallelConfig())
    core.dp_sync_interval = 32
    core.step_counter = 0
    core.pending_pause = False
    core.prefill_delayer = PrefillDelayer(DP_SIZE, BUDGET, 1)
    local = _state(prefillable=1, decodes=2, pending_tokens=5)
    core.scheduler = MagicMock(prefill_delayer_state=MagicMock(return_value=local))
    synced: list[list[int]] = []

    def scripted_sync(dp_group, has_unfinished, pending_pause, extra):
        synced.append(extra.tolist())
        extra.mul_(DP_SIZE)  # every rank reports the same state
        return True, False

    monkeypatch.setattr(ParallelConfig, "sync_dp_state", staticmethod(scripted_sync))
    for _ in range(3):
        assert DPEngineCoreProc._has_global_unfinished_reqs(core, True)
        # Both ranks have prefill but it is underfilled: hold for more.
        assert core._should_throttle_prefills()
    assert synced == [list(local)] * 3
