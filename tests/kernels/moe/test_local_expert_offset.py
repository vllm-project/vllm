# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FusedMoEConfig.local_expert_offset must match the "linear" expert placement
of determine_expert_map, including when EP does not divide the expert count.

The TRT-LLM / FlashInfer MoE kernels take (local_expert_offset,
local_num_experts) instead of an expert map, so a wrong offset makes a rank run
other experts' weights without any error."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)


@pytest.mark.parametrize(
    "num_experts,ep_size",
    [(128, 8), (128, 6), (128, 7), (160, 7), (256, 3), (60, 8), (8, 8)],
)
def test_local_expert_offset_matches_linear_expert_map(num_experts, ep_size):
    covered = []
    for ep_rank in range(ep_size):
        local_num, expert_map, _ = determine_expert_map(
            ep_size, ep_rank, num_experts, "linear"
        )
        assert expert_map is not None
        local_ids = (expert_map >= 0).nonzero().flatten().tolist()
        assert len(local_ids) == local_num
        # linear placement: one contiguous block per rank
        assert local_ids == list(range(local_ids[0], local_ids[0] + local_num))

        cfg = SimpleNamespace(
            num_experts=num_experts, ep_size=ep_size, ep_rank=ep_rank
        )
        assert FusedMoEConfig.local_expert_offset.fget(cfg) == local_ids[0]
        covered += local_ids
    assert covered == list(range(num_experts))


def test_local_expert_offset_differs_from_naive_when_uneven():
    # 128 experts on EP7: ranks 0-1 hold 19 experts, ranks 2-6 hold 18, so
    # rank 2 starts at 38 while ep_rank * num_local_experts gives 36.
    cfg = SimpleNamespace(num_experts=128, ep_size=7, ep_rank=2)
    assert FusedMoEConfig.local_expert_offset.fget(cfg) == 38
