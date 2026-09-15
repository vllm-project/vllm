# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights

RANK, ALPHA = 16, 32
# PEFTHelper stores alpha/sqrt(r) for use_rslora adapters.
RSLORA_SCALING = ALPHA / RANK**0.5


@pytest.fixture(autouse=True)
def should_do_global_cleanup_after_test():
    # Pure-python weight packing; skip GPU/ray cleanup.
    return False


def moe_loras(scaling: float, stacked: bool = False) -> list[LoRALayerWeights]:
    a_shape = (3, RANK, 4) if stacked else (RANK, 4)
    b_shape = (3, 6, RANK) if stacked else (6, RANK)
    return [
        LoRALayerWeights(
            module_name=f"experts.w{i}",
            rank=RANK,
            lora_alpha=ALPHA,
            lora_a=torch.randn(*a_shape),
            lora_b=torch.randn(*b_shape),
            scaling=scaling,
        )
        for i in (1, 2, 3)
    ]


@pytest.mark.parametrize(
    "packer,stacked",
    [
        (PackedLoRALayerWeights.pack_moe, False),
        (PackedLoRALayerWeights.pack_moe_stacked, True),
    ],
)
def test_moe_packing_uses_stored_scaling(packer, stacked: bool):
    packed = packer(moe_loras(RSLORA_SCALING, stacked), "experts")
    assert packed.scaling == pytest.approx([RSLORA_SCALING] * 3)


def test_non_gated_moe_keeps_w3_unscaled():
    packed = PackedLoRALayerWeights.pack_moe(
        moe_loras(RSLORA_SCALING), "experts", is_non_gated_moe=True
    )
    assert packed.scaling[:2] == pytest.approx([RSLORA_SCALING] * 2)
    assert packed.scaling[2] == 1.0
