# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
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
)

pytestmark = pytest.mark.cpu_test


def test_take_expert_metrics_snapshot_copies_topology_once():
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            enable_expert_metrics=True,
            expert_metrics_trace=False,
            expert_metrics_trace_interval=1,
            expert_metrics_trace_path="unused.jsonl",
        )
    )
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
        eplb_config=SimpleNamespace(
            enable_expert_metrics=True,
            expert_metrics_trace=False,
            expert_metrics_trace_interval=1,
            expert_metrics_trace_path="unused.jsonl",
        )
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
        eplb_config=SimpleNamespace(enable_expert_metrics=True)
    )
    state.expert_metrics_window_step = 7
    state._pending_expert_tokens = []
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


def test_trace_writes_per_rank_jsonl(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        data_parallel_rank=0,
        eplb_config=SimpleNamespace(
            expert_metrics_trace=True,
            expert_metrics_trace_interval=1,
            expert_metrics_trace_path=str(trace_path),
        ),
    )
    state.expert_metrics_total_step = 3
    state._expert_metrics_trace_files = {}
    model_state = SimpleNamespace(
        metrics=SimpleNamespace(
            model_name="model",
            ep_rank=1,
            num_ep_ranks=2,
            num_local_experts=2,
            phase=EPLB_PHASE_DECODE,
        )
    )

    state._write_expert_trace(model_state, torch.tensor([[1, 2], [3, 4]]))

    output = tmp_path / "trace.jsonl.dp0.ep1.model.jsonl"
    record = json.loads(output.read_text().strip())
    assert record["step"] == 3
    assert record["dp_rank"] == 0
    assert record["ep_rank"] == 1
    assert record["phase"] == EPLB_PHASE_DECODE
    assert record["tokens"] == [[1, 2], [3, 4]]
