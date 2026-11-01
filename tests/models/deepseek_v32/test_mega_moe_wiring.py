# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV2MoE MegaMoE wiring helpers (no GPU needed)."""

from types import SimpleNamespace

from torch import nn

from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.deepseek_v2 import (
    finalize_mega_moe_weights,
    mega_moe_routed_experts_prefix,
)


class _FakeMoE(deepseek_v2.DeepseekV2MoE):
    """A DeepseekV2MoE without __init__: only the attributes the helpers read."""

    def __init__(self, use_mega_moe: bool):
        nn.Module.__init__(self)
        self.use_mega_moe = use_mega_moe
        self.requantized = 0
        self.experts = SimpleNamespace(requantize_block_fp8=self._requantize)

    def _requantize(self):
        self.requantized += 1


def test_routed_experts_prefix_is_empty_for_mega_moe():
    dense = nn.Identity()
    assert mega_moe_routed_experts_prefix([dense, _FakeMoE(False)]) == "routed_experts"
    assert mega_moe_routed_experts_prefix([dense, _FakeMoE(True)]) == ""
    assert mega_moe_routed_experts_prefix(iter([_FakeMoE(True)])) == ""


def test_finalize_requantizes_mega_moe_layers_only():
    mega, fused = _FakeMoE(True), _FakeMoE(False)
    finalize_mega_moe_weights([mega, fused, nn.Identity()])
    assert mega.requantized == 1
    assert fused.requantized == 0
