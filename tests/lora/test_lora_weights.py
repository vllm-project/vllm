# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from safetensors.torch import save_file

from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.utils import WeightsMapper

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


@pytest.mark.parametrize("a_rank,b_rank", [(4, 4), (8, 4), (4, 8)])
@pytest.mark.parametrize(
    "module,expected_modules",
    [
        ("model.layers.0.self_attn.q_proj", {"q_proj"}),
        ("model.layers.0.mlp.experts.0.down_proj", {"experts.0.down_proj"}),
    ],
)
def test_checkpoint_rejects_inconsistent_rank(
    tmp_path, a_rank, b_rank, module, expected_modules
):
    """Reject bad factors before rank-buffer padding can hide the mismatch."""
    save_file(
        {
            f"base_model.model.{module}.lora_A.weight": torch.ones(a_rank, 6),
            f"base_model.model.{module}.lora_B.weight": torch.ones(5, b_rank),
        },
        tmp_path / "adapter_model.safetensors",
    )
    target = module.rsplit(".", 1)[-1]
    peft_helper = PEFTHelper(r=8, lora_alpha=16, target_modules=[target])
    with pytest.raises(ValueError, match=f"{target}.*expects rank 8"):
        LoRAModel.from_local_checkpoint(
            str(tmp_path), expected_modules, peft_helper, lora_model_id=1, device="cpu"
        )


@pytest.mark.parametrize("factor", ["A", "B"])
def test_lora_tensors_require_paired_factors(factor):
    peft_helper = PEFTHelper(r=4, lora_alpha=8, target_modules=["q_proj"])
    tensor = torch.ones(4, 6) if factor == "A" else torch.ones(5, 4)
    missing = "B" if factor == "A" else "A"
    with pytest.raises(ValueError, match=f"q_proj.*missing lora_{missing}"):
        LoRAModel.from_lora_tensors(
            1, {f"q_proj.lora_{factor}.weight": tensor}, peft_helper, device="cpu"
        )


def test_dynamic_rank_survives_weights_mapping():
    """Use the original PEFT name's rank, including save_as_lora layouts."""
    module = "model.layers.0.self_attn.q_proj"
    peft_helper = PEFTHelper(
        r=1,
        lora_alpha=1,
        target_modules=["q_proj"],
        rank_pattern={module: 3},
        alpha_pattern={module: 3},
    )
    lora_model = LoRAModel.from_lora_tensors(
        1,
        {
            f"base_model.model.{module}.lora_A.weight": torch.ones(3, 6),
            f"base_model.model.{module}.lora_B.weight": torch.ones(5, 3),
        },
        peft_helper,
        device="cpu",
        weights_mapper=WeightsMapper(orig_to_new_prefix={"model.": "renamed."}),
    )
    lora = lora_model.loras["renamed.layers.0.self_attn.q_proj"]
    assert lora.rank == 3
    torch.testing.assert_close(lora.optimize().lora_b, torch.ones(5, 3))


@pytest.mark.parametrize("module", ["experts", "experts.base_layer"])
def test_fused_moe_flattened_rank_survives_weights_mapping(module):
    """Fused PEFT factors encode rank times experts, before names are mapped."""
    peft_helper = PEFTHelper(r=4, lora_alpha=8, target_modules=["experts"])
    lora_model = LoRAModel.from_lora_tensors(
        1,
        {
            f"{module}.lora_A.weight": torch.ones(12, 6),
            f"{module}.lora_B.weight": torch.ones(5, 12),
        },
        peft_helper,
        device="cpu",
        weights_mapper=WeightsMapper(orig_to_new_prefix={"experts": "fused"}),
    )
    lora = lora_model.loras[module.replace("experts", "fused", 1)]
    assert lora.lora_a.shape == (12, 6)
    assert lora.lora_b.shape == (5, 12)


def test_lora_loader_preserves_stacked_expert_factors():
    peft_helper = PEFTHelper(r=4, lora_alpha=8, target_modules=["w1"])
    lora_model = LoRAModel.from_lora_tensors(
        1,
        {
            "experts.w1.lora_A.weight": torch.ones(3, 4, 6),
            "experts.w1.lora_B.weight": torch.ones(3, 5, 4),
        },
        peft_helper,
        device="cpu",
    )
    assert lora_model.loras["experts.w1"].lora_a.shape == (3, 4, 6)
