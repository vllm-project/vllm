# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.eplb.eplb_state import (
    EplbModelMetricsState,
    EplbState,
)
from vllm.distributed.eplb.metrics import EplbExpertTokens

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
