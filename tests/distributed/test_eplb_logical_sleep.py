# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbState


def test_logical_resize_migrates_once_and_preserves_maximum_mapping(monkeypatch):
    from vllm.distributed.eplb import eplb_state as module

    group = SimpleNamespace(rank=lambda: 1, size=lambda: 2)
    monkeypatch.setattr(
        module, "get_ep_group", lambda: SimpleNamespace(device_group=group)
    )
    monkeypatch.setattr(module, "get_node_count", lambda: 1)
    migrate, commit = Mock(), Mock()
    monkeypatch.setattr(module, "rearrange_expert_weights_inplace", migrate)
    monkeypatch.setattr(module, "_commit_eplb_maps", commit)
    state = EplbState.__new__(EplbState)
    state.is_async = False
    state.num_valid_physical_experts = 4
    state.expert_load_window_size = 1
    state._allreduce_list = lambda values: values
    state.policy = SimpleNamespace(
        rebalance_experts=lambda *args: torch.tensor([[0, 1]])
    )
    state.model_states = {
        "model": SimpleNamespace(
            expert_load_window=torch.ones(1, 1, 4),
            physical_to_logical_map=torch.tensor([[0, 1, 0, 1]]),
            model=SimpleNamespace(
                num_moe_layers=1,
                num_logical_experts=2,
                num_physical_experts=4,
                num_expert_groups=1,
                expert_weights=(),
            ),
            expert_buffer=(),
            communicator=object(),
        )
    }
    state.rearrange(rank_mapping={0: 0, 1: -1}, preserve_world_size=True)
    migrate.assert_called_once()
    commit.assert_called_once()
    assert commit.call_args.kwargs["new_physical_to_logical_map"].tolist() == [
        [0, 1, -1, -1]
    ]


def test_build_logical_sleep_rank_mapping_requires_suffix():
    assert EplbState.build_logical_sleep_rank_mapping(4, [2, 3]) == {
        0: 0,
        1: 1,
        2: -1,
        3: -1,
    }

    with pytest.raises(ValueError, match="sleeping a suffix"):
        EplbState.build_logical_sleep_rank_mapping(4, [1, 3])


@pytest.mark.parametrize("sleeping_ranks", [[], [0, 1, 2, 3], [-1], [4]])
def test_build_logical_sleep_rank_mapping_rejects_invalid_ranks(sleeping_ranks):
    with pytest.raises(ValueError):
        EplbState.build_logical_sleep_rank_mapping(4, sleeping_ranks)
