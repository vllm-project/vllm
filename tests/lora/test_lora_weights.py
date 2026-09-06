# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights

RANK = 16
ALPHA = 32
# PEFTHelper stores alpha/sqrt(r) for use_rslora adapters; alpha/r otherwise.
RSLORA_SCALING = ALPHA / RANK**0.5
PLAIN_SCALING = ALPHA / RANK


def make_lora(name: str, scaling: float, stacked: bool = False) -> LoRALayerWeights:
    a_shape = (3, RANK, 4) if stacked else (RANK, 4)
    b_shape = (3, 6, RANK) if stacked else (6, RANK)
    return LoRALayerWeights(
        module_name=name,
        rank=RANK,
        lora_alpha=ALPHA,
        lora_a=torch.randn(*a_shape),
        lora_b=torch.randn(*b_shape),
        scaling=scaling,
    )


def make_triplets(scaling: float, stacked: bool = False) -> list[LoRALayerWeights]:
    return [
        make_lora("experts.w1", scaling, stacked),
        make_lora("experts.w2", scaling, stacked),
        make_lora("experts.w3", scaling, stacked),
    ]


@pytest.fixture(autouse=True)
def should_do_global_cleanup_after_test():
    # Pure-python weight-packing test; skip GPU/ray cleanup.
    return False


@pytest.mark.parametrize("scaling", [RSLORA_SCALING, PLAIN_SCALING])
def test_pack_moe_uses_stored_scaling(scaling: float):
    packed = PackedLoRALayerWeights.pack_moe(make_triplets(scaling), "experts")
    assert packed.scaling == pytest.approx([scaling, scaling, scaling])


def test_pack_moe_non_gated_keeps_w3_unscaled():
    packed = PackedLoRALayerWeights.pack_moe(
        make_triplets(RSLORA_SCALING), "experts", is_non_gated_moe=True
    )
    assert packed.scaling[:2] == pytest.approx([RSLORA_SCALING, RSLORA_SCALING])
    assert packed.scaling[2] == 1.0


@pytest.mark.parametrize("scaling", [RSLORA_SCALING, PLAIN_SCALING])
def test_pack_moe_stacked_uses_stored_scaling(scaling: float):
    packed = PackedLoRALayerWeights.pack_moe_stacked(
        make_triplets(scaling, stacked=True), "experts"
    )
    assert packed.scaling == pytest.approx([scaling, scaling, scaling])
