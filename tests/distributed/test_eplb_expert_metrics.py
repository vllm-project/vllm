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
)

pytestmark = pytest.mark.cpu_test


def test_take_expert_metrics_snapshot_copies_topology_once():
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(enable_expert_metrics=True)
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
        eplb_config=SimpleNamespace(enable_expert_metrics=True)
    )
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
