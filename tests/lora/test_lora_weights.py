# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.lora.lora_model import LoRAModel
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
    # PackedLoRALayerWeights stores one scaling per packed projection but
    # inherits the base class's float annotation.
    scaling = packed.scaling
    assert scaling[:2] == pytest.approx([RSLORA_SCALING] * 2)  # type: ignore[index]
    assert scaling[2] == 1.0  # type: ignore[index]


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


def test_moe_packing_rejects_per_expert_rank():
    """Experts of one projection must share a rank to be stacked."""
    loras = moe_loras(1.0) + moe_loras(1.0)
    loras[4].lora_a = torch.randn(RANK // 2, 4)
    loras[4].lora_b = torch.randn(6, RANK // 2)
    with pytest.raises(ValueError, match="different LoRA ranks"):
        PackedLoRALayerWeights.pack_moe(loras, "experts")


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


_FOLD_MODULE = "model.layers.0.self_attn.o_proj"


def _fold_adapter(r: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        f"base_model.model.{_FOLD_MODULE}.lora_A.weight": torch.randn(
            r, 64, generator=g
        ).to(dtype),
        f"base_model.model.{_FOLD_MODULE}.lora_B.weight": (
            torch.randn(256, r, generator=g) * 0.02
        ).to(dtype),
    }


@pytest.mark.parametrize(
    "r,alpha,use_rslora",
    [(8, 16, True), (12, 16, False), (8, 16, False)],
)
def test_from_lora_tensors_folds_scaling_before_cast(r, alpha, use_rslora):
    """A float32 adapter's lora_b must be scaled before it is rounded to
    lora_dtype, not rounded, scaled in lora_dtype and rounded again."""
    tensors = _fold_adapter(r, torch.float32)
    b = tensors[f"base_model.model.{_FOLD_MODULE}.lora_B.weight"].clone()
    helper = PEFTHelper(
        r=r, lora_alpha=alpha, target_modules=["o_proj"], use_rslora=use_rslora
    )
    lora = LoRAModel.from_lora_tensors(
        1, tensors, peft_helper=helper, device="cpu", dtype=torch.bfloat16
    ).loras[_FOLD_MODULE]
    assert lora.scaling == 1
    lora.optimize()
    expected = (b * helper.vllm_lora_scaling_factor).to(torch.bfloat16)
    torch.testing.assert_close(lora.lora_b, expected, rtol=0, atol=0)


def test_from_lora_tensors_does_not_scale_caller_tensors():
    """Tensors that already have lora_dtype must not be scaled in place."""
    tensors = _fold_adapter(8, torch.bfloat16)
    before = {k: v.clone() for k, v in tensors.items()}
    helper = PEFTHelper(r=8, lora_alpha=16, target_modules=["o_proj"])
    model = LoRAModel.from_lora_tensors(
        1, tensors, peft_helper=helper, device=None, dtype=torch.bfloat16
    )
    for lora in model.loras.values():
        lora.optimize()
    for k, v in tensors.items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
