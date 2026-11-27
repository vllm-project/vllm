# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from safetensors.torch import save_file

from vllm.lora.lora_model import LoRAModel, MoEEPLoadSpec
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.utils import WeightsMapper

lora_lst = ["baichuan7B", "baichuan7B-zero", "baichuan7B-zero-regex", "chatglm3-6b"]
BAICHUAN_LORA_MODULES = [
    "W_pack",
    "o_proj",
    "gate_up_proj",
    "down_proj",
]

MOCK_PACKED_MAPPING = {
    "W_pack": ["W_pack"],
    "gate_up_proj": [
        "gate_proj",
        "up_proj",
    ],
}


@pytest.mark.parametrize("lora_name", lora_lst)
def test_load_checkpoints(
    lora_name,
    baichuan_lora_files,
    baichuan_zero_lora_files,
    baichuan_regex_lora_files,
    chatglm3_lora_files,
):
    expected_lora_lst: list[str] = []
    for module in BAICHUAN_LORA_MODULES:
        if module in MOCK_PACKED_MAPPING:
            expected_lora_lst.extend(MOCK_PACKED_MAPPING[module])
        else:
            expected_lora_lst.append(module)
    expected_lora_modules = set(expected_lora_lst)
    if lora_name == "baichuan7B":
        peft_helper = PEFTHelper.from_local_dir(
            baichuan_lora_files, max_position_embeddings=4096
        )
        # For the baichuan7B model, load it's LoRA,
        # and the test should pass.
        LoRAModel.from_local_checkpoint(
            baichuan_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            model_vocab_size=64000,
        )
    elif lora_name == "baichuan7B-zero":
        # Test that the target_modules contain prefix
        # such as "model.layers.0.self_atten.W_pack", and
        # the test should pass.
        peft_helper = PEFTHelper.from_local_dir(
            baichuan_zero_lora_files, max_position_embeddings=4096
        )
        LoRAModel.from_local_checkpoint(
            baichuan_zero_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            model_vocab_size=64000,
        )
    elif lora_name == "baichuan7B-zero-regex":
        # Test that the `target_modules` in the form of regular expressions,
        # such as `model\\..*(W_pack|o_proj)`, and the test should pass.
        peft_helper = PEFTHelper.from_local_dir(
            baichuan_regex_lora_files, max_position_embeddings=4096
        )
        LoRAModel.from_local_checkpoint(
            baichuan_regex_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            model_vocab_size=64000,
        )
    else:
        # For the baichuan7B model, load chatglm3-6b's LoRA,
        # and the test should raise the following error.
        expected_error = "Please verify that the loaded LoRA module is correct"  # noqa: E501
        peft_helper = PEFTHelper.from_local_dir(
            chatglm3_lora_files, max_position_embeddings=4096
        )
        with pytest.raises(ValueError, match=expected_error):
            LoRAModel.from_local_checkpoint(
                chatglm3_lora_files,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=1,
                device="cpu",
                model_vocab_size=64000,
            )


def test_lora_weights_mapping(baichuan_lora_files):
    expected_lora_lst: list[str] = []
    for module in BAICHUAN_LORA_MODULES:
        if module in MOCK_PACKED_MAPPING:
            expected_lora_lst.extend(MOCK_PACKED_MAPPING[module])
        else:
            expected_lora_lst.append(module)
    expected_lora_modules = set(expected_lora_lst)
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
        },
        orig_to_new_substr={
            ".layers.": ".baichuan_layers.",
        },
    )
    peft_helper = PEFTHelper.from_local_dir(
        baichuan_lora_files, max_position_embeddings=4096
    )
    lora_model = LoRAModel.from_local_checkpoint(
        baichuan_lora_files,
        expected_lora_modules,
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        model_vocab_size=64000,
        weights_mapper=hf_to_vllm_mapper,
    )
    for name in lora_model.loras:
        assert name.startswith(hf_to_vllm_mapper.orig_to_new_prefix["model."])
        assert ".baichuan_layers." in name


def test_gemma4_lora_weights_mapping():
    mapper = Gemma4ForCausalLM.hf_to_vllm_mapper
    name = "base_model.model.model.language_model.layers.9.mlp.down_proj.lora_A.weight"
    assert parse_fine_tuned_lora_name(name, mapper) == (
        "model.layers.9.mlp.down_proj",
        True,
    )


def test_gemma4_moe_lora_weights_mapping():
    mapper = Gemma4ForCausalLM.hf_to_vllm_mapper
    name = (
        "base_model.model.model.language_model.layers.9.experts."
        "gate_up_proj.lora_B.weight"
    )
    assert parse_fine_tuned_lora_name(name, mapper) == (
        "model.layers.9.experts.gate_up_proj",
        False,
    )


@pytest.mark.parametrize("source", ["memory", "disk"])
@pytest.mark.parametrize("unsupported", [False, True])
def test_tensor_adapter_module_validation(tmp_path, source, unsupported):
    """Both sources reject unknown or unwrapped target modules."""
    tensors = {
        f"base_model.model.model.pooler.dense.lora_{ab}.weight": torch.ones(4, 4)
        for ab in ("A", "B")
    }
    peft_helper = PEFTHelper(r=4, lora_alpha=4, target_modules=["dense"])
    expected = {"dense"} if unsupported else {"q_proj"}
    kwargs = dict(
        peft_helper=peft_helper,
        device="cpu",
        unsupported_lora_modules={"model.pooler.dense"} if unsupported else set(),
    )
    with pytest.raises(ValueError, match="Please verify that the loaded LoRA"):
        if source == "memory":
            LoRAModel.from_lora_tensors(
                1, tensors, expected_lora_modules=expected, **kwargs
            )
        else:
            save_file(tensors, tmp_path / "adapter_model.safetensors")
            LoRAModel.from_local_checkpoint(str(tmp_path), expected, **kwargs)


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_tensor_adapter_matches_checkpoint(tmp_path, ep_rank):
    """Memory loading preserves mapping, skips, EP ownership and tensor values."""
    weight = torch.arange(16).reshape(4, 4).float()
    prefix = "base_model.model.model.layers.0.mlp.experts"
    tensors = {
        f"{prefix}.{expert}.{proj}.lora_{ab}.weight": weight + expert
        for expert in range(4)
        for proj in ("gate_proj", "up_proj", "down_proj")
        for ab in ("A", "B")
    }
    tensors["base_model.model.mtp.ignored.lora_A.weight"] = torch.ones(4, 4)
    save_file(tensors, tmp_path / "adapter_model.safetensors")
    peft_helper = PEFTHelper(r=4, lora_alpha=8, target_modules=["gate_proj"])
    expected = {
        f"experts.{expert}.{proj}"
        for expert in range(4)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    kwargs = dict(
        peft_helper=peft_helper,
        device="cpu",
        weights_mapper=WeightsMapper(orig_to_new_prefix={"model.": "language_model."}),
        skip_prefixes=["mtp."],
        moe_ep_spec=MoEEPLoadSpec(ep_rank, 2, 4),
    )
    disk = LoRAModel.from_local_checkpoint(str(tmp_path), expected, **kwargs)
    memory = LoRAModel.from_lora_tensors(
        2, tensors, expected_lora_modules=expected, **kwargs
    )
    assert memory.loras.keys() == disk.loras.keys()
    assert len(memory.loras) == 6
    for name, weights in memory.loras.items():
        assert name.startswith("language_model.")
        expert = int(name.split(".experts.")[1].split(".")[0])
        assert expert // 2 == ep_rank
        torch.testing.assert_close(
            weights.lora_a, disk.loras[name].lora_a, rtol=0, atol=0
        )
        torch.testing.assert_close(
            weights.lora_b, disk.loras[name].lora_b, rtol=0, atol=0
        )
        assert weights.scaling == disk.loras[name].scaling
