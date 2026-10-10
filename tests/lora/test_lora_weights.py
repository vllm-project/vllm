# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper

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


@pytest.mark.parametrize(
    "packer,stacked",
    [
        (PackedLoRALayerWeights.pack_moe, False),
        (PackedLoRALayerWeights.pack_moe_stacked, True),
    ],
)
def test_moe_packing_keeps_per_projection_scaling(packer, stacked: bool):
    # e.g. rank_pattern / alpha_pattern giving gate, down and up different scalings
    loras = moe_loras(1.0, stacked)
    for lora, scaling in zip(loras, (2.0, 4.0, 8.0)):
        lora.scaling = scaling
    packed = packer(loras, "experts")
    assert packed.scaling == pytest.approx([2.0, 4.0, 8.0])


def test_moe_packing_keeps_per_expert_scaling():
    """Experts of one projection can have different scaling, e.g. from
    alpha_pattern."""
    loras = moe_loras(RSLORA_SCALING) + moe_loras(RSLORA_SCALING)
    loras[4].scaling = 2 * RSLORA_SCALING  # w2 of expert 1
    expected = [lora.lora_b * lora.scaling for lora in loras]
    packed = PackedLoRALayerWeights.pack_moe(loras, "experts").optimize()
    for i, lora_b in enumerate(packed.lora_b):
        torch.testing.assert_close(lora_b, torch.stack(expected[i::3]))


def test_from_config_uses_checkpoint_module_name():
    peft_helper = PEFTHelper(
        r=RANK,
        lora_alpha=ALPHA,
        target_modules=["q_proj"],
        rank_pattern={r"^model\.layers\.0\.self_attn\.q_proj": 4},
    )
    # the pattern refers to the checkpoint name, not the name after weights mapping
    lora = LoRALayerWeights.from_config(
        "language_model.model.layers.0.self_attn.q_proj",
        peft_helper,
        "model.layers.0.self_attn.q_proj",
    )
    assert lora.rank == 4
    assert lora.scaling == pytest.approx(ALPHA / 4)
    lora = LoRALayerWeights.from_config("model.layers.1.self_attn.q_proj", peft_helper)
    assert lora.rank == RANK
    assert lora.scaling == pytest.approx(ALPHA / RANK)
