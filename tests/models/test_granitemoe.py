# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.granitemoe import GraniteMoeModel


class _StubModel(torch.nn.Module):
    """Minimal stand-in for `get_expert_mapping`.

    The mapping only reads `config.num_local_experts` and scans
    `named_parameters()` for a LoRA `base_layer.` prefix, so no real weights
    or CUDA are needed.
    """

    def __init__(self, num_local_experts: int):
        super().__init__()
        self.config = SimpleNamespace(num_local_experts=num_local_experts)

    def named_parameters(self, *args, **kwargs):
        return iter([])


def test_granitemoe_expert_mapping_split_experts():
    """Split per-expert checkpoint names map onto the fused FusedMoE slots.

    Regression guard for the FP8 split-experts KeyError: each
    `experts.{e}.{gate,up,down}_proj.` name must route to the right
    `w13`/`w2` shard for its expert. gate->w1 and up->w3 land in `w13_`;
    down->w2 lands in `w2_`.
    """
    mapping = GraniteMoeModel.get_expert_mapping(_StubModel(2))

    assert mapping == [
        ("experts.routed_experts.w13_", "experts.0.gate_proj.", 0, "w1"),
        ("experts.routed_experts.w2_", "experts.0.down_proj.", 0, "w2"),
        ("experts.routed_experts.w13_", "experts.0.up_proj.", 0, "w3"),
        ("experts.routed_experts.w13_", "experts.1.gate_proj.", 1, "w1"),
        ("experts.routed_experts.w2_", "experts.1.down_proj.", 1, "w2"),
        ("experts.routed_experts.w13_", "experts.1.up_proj.", 1, "w3"),
    ]
