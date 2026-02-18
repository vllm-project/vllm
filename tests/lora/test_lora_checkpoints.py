# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.baichuan import BaiChuanBaseForCausalLM
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


def test_lora_module_prefix_mismatch_detected(baichuan_lora_files):
    """LoRA adapters with mismatched module name prefixes should raise
    ValueError instead of silently being ignored (issue #34186)."""
    peft_helper = PEFTHelper.from_local_dir(
        baichuan_lora_files, max_position_embeddings=4096
    )
    # Simulate a model that wraps the LM in a language_model submodule:
    # the LoRA adapter has "model.layers.0..." keys but the model expects
    # "language_model.model.layers.0..." full paths.
    model_module_names = {
        "language_model.model.layers.0.self_attn.W_pack",
        "language_model.model.layers.0.self_attn.o_proj",
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
        "language_model.model.layers.0.mlp.down_proj",
    }
    expected_error = "none of the LoRA adapter's module names matched"
    with pytest.raises(ValueError, match=expected_error):
        LoRAModel.from_local_checkpoint(
            baichuan_lora_files,
            {"W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"},
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            model_vocab_size=64000,
            model_module_names=model_module_names,
        )


def test_lora_module_prefix_match_passes(baichuan_lora_files):
    """LoRA adapters with matching module name prefixes should load fine."""
    peft_helper = PEFTHelper.from_local_dir(
        baichuan_lora_files, max_position_embeddings=4096
    )
    # The baichuan adapter keys parse to "model.layers.N.self_attn.W_pack"
    # etc. Provide model_module_names that match.
    model_module_names = {
        "model.layers.0.self_attn.W_pack",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    }
    # Should not raise
    LoRAModel.from_local_checkpoint(
        baichuan_lora_files,
        {"W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"},
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        model_vocab_size=64000,
        model_module_names=model_module_names,
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
