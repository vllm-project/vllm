# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.baichuan import BaiChuanBaseForCausalLM
from vllm.model_executor.models.mistral import MistralForCausalLM
from vllm.model_executor.models.utils import WeightsMapper

lora_lst = ["baichuan7B", "baichuan7B-zero", "baichuan7B-zero-regex", "chatglm3-6b"]
BAICHUAN_LORA_MODULES = [
    "W_pack",
    "o_proj",
    "gate_up_proj",
    "down_proj",
]


@pytest.mark.parametrize("lora_name", lora_lst)
def test_load_checkpoints(
    lora_name,
    baichuan_lora_files,
    baichuan_zero_lora_files,
    baichuan_regex_lora_files,
    chatglm3_lora_files,
):
    packed_modules_mapping = BaiChuanBaseForCausalLM.packed_modules_mapping

    expected_lora_lst: list[str] = []
    for module in BAICHUAN_LORA_MODULES:
        if module in packed_modules_mapping:
            expected_lora_lst.extend(packed_modules_mapping[module])
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
    packed_modules_mapping = BaiChuanBaseForCausalLM.packed_modules_mapping

    expected_lora_lst: list[str] = []
    for module in BAICHUAN_LORA_MODULES:
        if module in packed_modules_mapping:
            expected_lora_lst.extend(packed_modules_mapping[module])
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


def test_mistral_lora_format_weights_mapping():
    """Test that Mistral-format LoRA weight keys are correctly remapped.

    Mistral consolidated checkpoints use different layer names than the
    HuggingFace / vLLM convention, e.g.:
      layers.7.attention.wk  ->  model.layers.7.self_attn.k_proj
      layers.7.feed_forward.w1  ->  model.layers.7.mlp.gate_proj

    MistralForCausalLM.hf_to_vllm_mapper must translate these keys so that
    LoRA weights saved in the Mistral format can be loaded without manual
    key renaming.
    """
    rank = 8
    hidden_size = 64
    intermediate_size = 256

    # Synthetic LoRA tensors using Mistral-format key names (no base_model prefix).
    # Shapes follow the LoRA convention: lora_A=(rank, in_features),
    # lora_B=(out_features, rank).
    mistral_format_tensors: dict[str, torch.Tensor] = {}
    mistral_keys = [
        # (module_key, out_features, in_features)
        ("layers.0.attention.wq", hidden_size, hidden_size),
        ("layers.0.attention.wk", hidden_size, hidden_size),
        ("layers.0.attention.wv", hidden_size, hidden_size),
        ("layers.0.attention.wo", hidden_size, hidden_size),
        ("layers.0.feed_forward.w1", intermediate_size, hidden_size),
        ("layers.0.feed_forward.w2", hidden_size, intermediate_size),
        ("layers.0.feed_forward.w3", intermediate_size, hidden_size),
    ]
    for module_key, out_features, in_features in mistral_keys:
        mistral_format_tensors[f"{module_key}.lora_A.weight"] = torch.zeros(
            rank, in_features
        )
        mistral_format_tensors[f"{module_key}.lora_B.weight"] = torch.zeros(
            out_features, rank
        )

    peft_helper = PEFTHelper.from_dict(
        {
            "r": rank,
            "lora_alpha": rank,
            "target_modules": ["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
        }
    )

    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=1,
        tensors=mistral_format_tensors,
        peft_helper=peft_helper,
        device="cpu",
        weights_mapper=MistralForCausalLM.hf_to_vllm_mapper,
    )

    # All loaded module names must use vLLM's naming convention
    expected_module_names = {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.up_proj",
    }
    assert set(lora_model.loras.keys()) == expected_module_names

    # Backward compat: standard PEFT-format keys (after stripping
    # "base_model.model.") must pass through the mapper unchanged.
    hf_format_keys = [
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.k_proj.lora_A.weight",
        "model.layers.0.mlp.gate_proj.lora_A.weight",
        "model.layers.0.mlp.down_proj.lora_A.weight",
    ]
    mapper = MistralForCausalLM.hf_to_vllm_mapper
    for key in hf_format_keys:
        assert mapper._map_name(key) == key, (
            f"hf_to_vllm_mapper must not modify standard HF-format keys, "
            f"but '{key}' was changed to '{mapper._map_name(key)}'"
        )
