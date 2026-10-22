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


def test_finalize_skips_experts_loaded_as_is():
    moe = _FakeMoE(True)
    moe.experts = SimpleNamespace()  # NVFP4/BF16 experts: no requantization
    finalize_mega_moe_weights([moe])


class _FakeQuantConfig:
    def __init__(self, name: str, algos: dict[str, str] | None = None):
        self.name, self.algos = name, algos or {}

    def get_name(self) -> str:
        return self.name

    def _resolve_quant_algo(self, prefix: str) -> str | None:
        return self.algos.get(prefix)


def test_mega_moe_weight_format():
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        MEGA_MOE_BF16,
        MEGA_MOE_NVFP4,
    )

    select = deepseek_v2.DeepseekV2MoE._mega_moe_weight_format
    prefix = "model.layers.3.mlp"
    assert select(None, prefix) is MEGA_MOE_BF16
    assert select(_FakeQuantConfig("modelopt_fp4"), prefix) is MEGA_MOE_NVFP4
    mixed = _FakeQuantConfig("modelopt_mixed", {f"{prefix}.experts": "NVFP4"})
    assert select(mixed, prefix) is MEGA_MOE_NVFP4
    # Excluded from quantization (e.g. a BF16 MTP layer).
    assert select(mixed, "model.layers.78.mlp") is MEGA_MOE_BF16
    # Block FP8 experts keep the requantizing path.
    assert select(_FakeQuantConfig("fp8"), prefix) is None
    fp8_mixed = _FakeQuantConfig("modelopt_mixed", {f"{prefix}.experts": "FP8_PB"})
    assert select(fp8_mixed, prefix) is None
