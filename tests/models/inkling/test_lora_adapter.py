# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from safetensors.torch import save_file

from vllm.lora.lora_model import LoRAModel, MoEEPLoadSpec
from vllm.lora.peft_helper import PEFTHelper
from vllm.models.inkling.lora import convert_inkling_lora


def _converter(tensors, helper):
    return convert_inkling_lora(tensors, helper, num_experts=3, num_shared_experts=2)


def test_dense_interleave_preserves_gate_and_up_deltas():
    ag = torch.arange(10, dtype=torch.float64).reshape(2, 5)
    au = ag + 1
    bg = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    bu = bg + 1
    tensors = {
        f"model.layers.0.mlp.{name}.lora_{factor}.weight": value
        for name, (a, b) in (("gate_proj", (ag, bg)), ("up_proj", (au, bu)))
        for factor, value in (("A", a), ("B", b))
    }
    helper = PEFTHelper.from_dict(
        {"r": 2, "lora_alpha": 6, "target_modules": ["gate_proj", "up_proj"]}
    )
    packed, converted = _converter(tensors, helper)
    prefix = "model.layers.0.mlp.gate_up_proj"
    actual = packed[prefix + ".lora_B.weight"] @ packed[prefix + ".lora_A.weight"]
    expected = torch.stack((bg @ ag, bu @ au), dim=1).flatten(0, 1)
    torch.testing.assert_close(
        actual * converted.vllm_lora_scaling_factor, expected * 3
    )


def test_expert_factors_preserve_independent_deltas():
    experts, rank = 3, 2
    a = torch.arange(experts * rank * 5, dtype=torch.float64).reshape(experts, rank, 5)
    b = torch.arange(experts * 4 * rank, dtype=torch.float64).reshape(experts, 4, rank)
    tensors = {
        "model.layers.0.mlp.experts.lora_A.weight": a.flatten(0, 1),
        "model.layers.0.mlp.experts.lora_B.weight": b.permute(1, 2, 0)
        .flatten(1)
        .contiguous(),
    }
    helper = PEFTHelper.from_dict(
        {"r": rank, "lora_alpha": 6, "target_modules": ["down_proj"]}
    )
    packed, converted = _converter(tensors, helper)
    for expert in range(experts):
        prefix = f"model.layers.0.mlp.experts.{expert}.down_proj"
        actual = packed[prefix + ".lora_B.weight"] @ packed[prefix + ".lora_A.weight"]
        torch.testing.assert_close(actual, b[expert] @ a[expert])
    assert converted.r == rank


def test_disk_and_tensor_loading_use_same_conversion(tmp_path):
    shared = "model.layers.0.mlp.shared_experts.base_layer.base_layer"
    tensors = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 5),
        "model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2),
        f"{shared}.lora_A.weight": torch.ones(4, 5),
        f"{shared}.lora_B.weight": torch.ones(3, 4),
    }
    helper = PEFTHelper.from_dict(
        {"r": 2, "lora_alpha": 4, "target_modules": ["q_proj", "gate_proj"]}
    )
    save_file(tensors, tmp_path / "adapter_model.safetensors")
    disk = LoRAModel.from_local_checkpoint(
        str(tmp_path),
        {"wq_du", "w1"},
        helper,
        lora_model_id=1,
        device="cpu",
        adapter_converter=_converter,
    )
    memory = LoRAModel.from_lora_tensors(
        2, tensors, helper, device="cpu", adapter_converter=_converter
    )
    assert disk.rank == memory.rank == 4
    assert disk.loras.keys() == memory.loras.keys()
    for name in disk.loras:
        torch.testing.assert_close(disk.loras[name].lora_a, memory.loras[name].lora_a)
        torch.testing.assert_close(disk.loras[name].lora_b, memory.loras[name].lora_b)


def test_disk_conversion_keeps_only_local_expert(tmp_path):
    tensors = {
        "model.layers.0.mlp.experts.lora_A.weight": torch.ones(6, 5),
        "model.layers.0.mlp.experts.lora_B.weight": torch.ones(4, 6),
    }
    helper = PEFTHelper.from_dict(
        {"r": 2, "lora_alpha": 4, "target_modules": ["down_proj"]}
    )
    save_file(tensors, tmp_path / "adapter_model.safetensors")
    model = LoRAModel.from_local_checkpoint(
        str(tmp_path),
        {f"experts.{i}.down_proj" for i in range(3)},
        helper,
        lora_model_id=1,
        device="cpu",
        moe_ep_spec=MoEEPLoadSpec(ep_rank=1, local_num_experts=1, global_num_experts=3),
        adapter_converter=_converter,
    )
    assert set(model.loras) == {"model.layers.0.mlp.experts.1.down_proj"}


def test_incomplete_peft_pair_fails_closed():
    tensors = {"model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 5)}
    helper = PEFTHelper.from_dict(
        {"r": 2, "lora_alpha": 4, "target_modules": ["q_proj"]}
    )
    with pytest.raises(ValueError, match="Incomplete adapter A/B pair"):
        _converter(tensors, helper)
