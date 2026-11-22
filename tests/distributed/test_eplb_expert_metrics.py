# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.eplb.eplb_state import (
    EplbModelMetricsState,
    EplbState,
)
from vllm.distributed.eplb.metrics import (
    EPLB_PHASE_DECODE,
    EPLB_PHASE_MIXED,
    EPLB_PHASE_PREFILL,
    EplbExpertTokens,
    EplbLoadSample,
    EplbRearrangementStall,
)

pytestmark = pytest.mark.cpu_test


def test_take_expert_metrics_snapshot_copies_topology_once():
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(enable_expert_metrics=True)
    )
    state._pending_eplb_stalls = []
    state._pending_eplb_loads = []
    state._pending_expert_tokens = [
        EplbExpertTokens(
            model_name="model",
            ep_rank=0,
            num_ep_ranks=2,
            num_local_experts=2,
            tokens=[[1, 2, 3, 4]],
        )
    ]
    model_state = SimpleNamespace(
        metrics=EplbModelMetricsState(
            model_name="model",
            ep_rank=0,
            num_ep_ranks=2,
            num_local_experts=2,
        ),
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]]),
    )
    state.model_states = {"model": model_state}

    snapshot = state._take_expert_metrics_snapshot()

    assert snapshot is not None
    assert snapshot.tokens[0].tokens == [[1, 2, 3, 4]]
    assert snapshot.topology[0].physical_to_logical == [[0, 1, 0, 1]]
    assert state._pending_expert_tokens == []
    assert model_state.metrics.topology_dirty is False

    # Topology is unchanged, and there are no new token samples.
    assert state._take_expert_metrics_snapshot() is None


def test_phase_load_accumulation_separates_prefill_and_decode():
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(enable_expert_metrics=True)
    )
    state.expert_metrics_total_step = 1
    pass_load = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    phase_load = {
        phase: torch.zeros_like(pass_load)
        for phase in (EPLB_PHASE_PREFILL, EPLB_PHASE_DECODE, EPLB_PHASE_MIXED)
    }
    model_state = SimpleNamespace(
        expert_load_pass=pass_load,
        expert_load_pending=torch.zeros_like(pass_load),
        phase_load_accum=phase_load,
        metrics=SimpleNamespace(phase=EPLB_PHASE_PREFILL),
    )
    state.model_states = {"model": model_state}

    state._accumulate_expert_metrics(model_state)

    assert torch.equal(model_state.expert_load_pending, torch.tensor([[1, 2, 3, 4]]))
    assert torch.equal(phase_load[EPLB_PHASE_PREFILL], torch.tensor([[1, 2, 3, 4]]))
    assert torch.equal(
        phase_load[EPLB_PHASE_DECODE], torch.zeros(1, 4, dtype=torch.int32)
    )
    assert torch.equal(
        phase_load[EPLB_PHASE_MIXED], torch.zeros(1, 4, dtype=torch.int32)
    )
    assert torch.count_nonzero(model_state.expert_load_pass) == 0

    model_state.metrics.phase = EPLB_PHASE_DECODE
    model_state.expert_load_pass.fill_(5)
    state._accumulate_expert_metrics(model_state)
    assert torch.equal(model_state.expert_load_pending, torch.tensor([[6, 7, 8, 9]]))
    assert torch.equal(
        phase_load[EPLB_PHASE_DECODE], torch.full((1, 4), 5, dtype=torch.int32)
    )

    state._reset_expert_metrics_accumulators()
    assert torch.count_nonzero(model_state.expert_load_pending) == 0
    assert all(torch.count_nonzero(t) == 0 for t in phase_load.values())


def test_sample_expert_metrics_emits_phase_samples_and_resets():
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(enable_expert_metrics=True, expert_metrics_top_k=2)
    )
    state.expert_metrics_window_step = 7
    state._pending_expert_tokens = []
    state._pending_eplb_loads = []
    phase_load = {
        phase: torch.ones((1, 4), dtype=torch.int32)
        for phase in (EPLB_PHASE_PREFILL, EPLB_PHASE_DECODE, EPLB_PHASE_MIXED)
    }
    model_state = SimpleNamespace(
        phase_load_accum=phase_load,
        metrics=EplbModelMetricsState(
            model_name="model",
            ep_rank=0,
            num_ep_ranks=2,
            num_local_experts=2,
        ),
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]]),
    )
    state.model_states = {"model": model_state}
    state._allreduce_list = lambda tensors: tensors

    state._sample_expert_metrics(rearranged=True)

    assert len(state._pending_expert_tokens) == 3
    assert {sample.phase for sample in state._pending_expert_tokens} == {
        EPLB_PHASE_PREFILL,
        EPLB_PHASE_DECODE,
        EPLB_PHASE_MIXED,
    }
    assert all(sample.steps == 7 for sample in state._pending_expert_tokens)
    assert all(sample.rearranged for sample in state._pending_expert_tokens)
    assert all(sample.timestamp > 0 for sample in state._pending_expert_tokens)
    assert state.expert_metrics_window_step == 0
    assert all(torch.count_nonzero(t) == 0 for t in phase_load.values())


def test_sample_expert_metrics_emits_layer_skew_and_excess():
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(enable_expert_metrics=True, expert_metrics_top_k=2)
    )
    state.expert_metrics_window_step = 4
    state._pending_expert_tokens = []
    state._pending_eplb_loads = []
    phase_load = {
        phase: torch.zeros((2, 4), dtype=torch.int32)
        for phase in (EPLB_PHASE_PREFILL, EPLB_PHASE_DECODE, EPLB_PHASE_MIXED)
    }
    # Layer 0: rank 0 gets 7 tokens, rank 1 gets 1 token.
    # Layer 1: both ranks get 4 tokens.
    phase_load[EPLB_PHASE_DECODE][0] = torch.tensor([3, 4, 1, 0])
    phase_load[EPLB_PHASE_DECODE][1] = torch.tensor([2, 2, 2, 2])
    model_state = SimpleNamespace(
        phase_load_accum=phase_load,
        metrics=EplbModelMetricsState(
            model_name="model",
            ep_rank=0,
            num_ep_ranks=2,
            num_local_experts=2,
        ),
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]]),
    )
    state.model_states = {"model": model_state}
    state._allreduce_list = lambda tensors: tensors

    state._sample_expert_metrics(rearranged=False)

    loads = sorted(
        (load for load in state._pending_eplb_loads if load.phase == EPLB_PHASE_DECODE),
        key=lambda load: load.layer,
    )
    assert [load.layer for load in loads] == [0, 1]

    layer0 = loads[0]
    assert layer0.expert_tokens == [3, 4, 1, 0]
    assert layer0.expert_gini == pytest.approx(0.4375)
    assert layer0.expert_top_k == 2
    assert layer0.expert_topk_share == pytest.approx(0.875)
    assert layer0.active_experts_per_rank == [2, 1]
    assert layer0.expert_excess_tokens == 2
    assert layer0.expert_skew == pytest.approx(2.0)
    assert layer0.rank_tokens == [7, 1]
    assert layer0.rank_excess_tokens == 3
    assert layer0.rank_skew == pytest.approx(7 / 4)
    assert layer0.steps == 4

    layer1 = loads[1]
    assert layer1.expert_tokens == [2, 2, 2, 2]
    assert layer1.expert_gini == pytest.approx(0.0)
    assert layer1.expert_top_k == 2
    assert layer1.expert_topk_share == pytest.approx(0.5)
    assert layer1.active_experts_per_rank == [2, 2]
    assert layer1.expert_excess_tokens == 0
    assert layer1.expert_skew == pytest.approx(1.0)
    assert layer1.rank_tokens == [4, 4]
    assert layer1.rank_excess_tokens == 0
    assert layer1.rank_skew == pytest.approx(1.0)
    # Empty phases must not produce zero-valued load samples.
    assert all(load.phase == EPLB_PHASE_DECODE for load in state._pending_eplb_loads)


def test_sample_expert_metrics_skips_empty_layers():
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(enable_expert_metrics=True, expert_metrics_top_k=2)
    )
    state.expert_metrics_window_step = 4
    state._pending_expert_tokens = []
    state._pending_eplb_loads = []
    phase_load = {
        phase: torch.zeros((2, 4), dtype=torch.int32)
        for phase in (EPLB_PHASE_PREFILL, EPLB_PHASE_DECODE, EPLB_PHASE_MIXED)
    }
    # Only layer 1 of the prefill phase has load; everything else is empty.
    phase_load[EPLB_PHASE_PREFILL][1] = torch.tensor([1, 1, 1, 1])
    model_state = SimpleNamespace(
        phase_load_accum=phase_load,
        metrics=EplbModelMetricsState(
            model_name="model",
            ep_rank=0,
            num_ep_ranks=2,
            num_local_experts=2,
        ),
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]]),
    )
    state.model_states = {"model": model_state}
    state._allreduce_list = lambda tensors: tensors

    state._sample_expert_metrics(rearranged=False)

    assert len(state._pending_eplb_loads) == 1
    sample = state._pending_eplb_loads[0]
    assert (sample.phase, sample.layer) == (EPLB_PHASE_PREFILL, 1)
    assert sample.expert_topk_share == pytest.approx(0.5)
    assert sample.active_experts_per_rank == [2, 2]


def test_rearrangement_stall_is_attributed_to_each_model():
    state = EplbState.__new__(EplbState)
    state._pending_eplb_stalls = []
    state.is_async = False
    state.model_states = {
        "main": SimpleNamespace(
            metrics=SimpleNamespace(model_name="main", phase=EPLB_PHASE_DECODE)
        ),
        "draft": SimpleNamespace(
            metrics=SimpleNamespace(model_name="draft", phase=EPLB_PHASE_MIXED)
        ),
    }

    state._record_rearrangement_stall(1.5, num_requests=32, num_scheduled_tokens=3072)

    assert len(state._pending_eplb_stalls) == 2
    assert all(stall.pause_seconds == 1.5 for stall in state._pending_eplb_stalls)
    assert all(stall.blocked_requests == 32 for stall in state._pending_eplb_stalls)
    assert all(stall.blocked_tokens == 3072 for stall in state._pending_eplb_stalls)
    assert {stall.model_name for stall in state._pending_eplb_stalls} == {
        "main",
        "draft",
    }


def test_take_expert_metrics_snapshot_includes_stalls_and_loads():
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(enable_expert_metrics=True)
    )
    state._pending_expert_tokens = []
    state._pending_eplb_stalls = [
        EplbRearrangementStall(
            model_name="model",
            pause_seconds=0.5,
            blocked_requests=8,
            blocked_tokens=512,
            phase=EPLB_PHASE_DECODE,
            is_async=False,
            timestamp=1.0,
        )
    ]
    state._pending_eplb_loads = [
        EplbLoadSample(
            model_name="model",
            phase=EPLB_PHASE_DECODE,
            layer=0,
            num_ep_ranks=2,
            num_local_experts=2,
            expert_tokens=[5, 3, 3, 1],
            expert_gini=0.25,
            expert_top_k=2,
            expert_topk_share=2 / 3,
            rank_tokens=[8, 4],
            active_experts_per_rank=[2, 2],
            expert_skew=5 / 3,
            expert_excess_tokens=2,
            rank_skew=4 / 3,
            rank_excess_tokens=2,
            timestamp=1.0,
            steps=10,
        )
    ]
    state.model_states = {}

    snapshot = state._take_expert_metrics_snapshot()

    assert snapshot is not None
    assert len(snapshot.stalls) == 1
    assert len(snapshot.loads) == 1
    assert state._pending_eplb_stalls == []
    assert state._pending_eplb_loads == []
