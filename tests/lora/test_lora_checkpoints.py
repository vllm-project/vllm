from typing import List

import pytest

from vllm.lora.models import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.baichuan import BaiChuanBaseForCausalLM
from vllm.model_executor.models.utils import WeightsMapper

lora_lst = [
    "baichuan7B", "baichuan7B-zero", "baichuan7B-zero-regex", "chatglm3-6b"
]


@pytest.mark.parametrize("lora_name", lora_lst)
def test_load_checkpoints(
    lora_name,
    baichuan_lora_files,
    baichuan_zero_lora_files,
    baichuan_regex_lora_files,
    chatglm3_lora_files,
):
    supported_lora_modules = BaiChuanBaseForCausalLM.supported_lora_modules
    packed_modules_mapping = BaiChuanBaseForCausalLM.packed_modules_mapping
    embedding_modules = BaiChuanBaseForCausalLM.embedding_modules
    embed_padding_modules = BaiChuanBaseForCausalLM.embedding_padding_modules
    expected_lora_modules: List[str] = []
    for module in supported_lora_modules:
        if module in packed_modules_mapping:
            expected_lora_modules.extend(packed_modules_mapping[module])
        else:
            expected_lora_modules.append(module)
    if lora_name == "baichuan7B":
        peft_helper = PEFTHelper.from_local_dir(baichuan_lora_files,
                                                max_position_embeddings=4096)
        # For the baichuan7B model, load it's LoRA,
        # and the test should pass.
        LoRAModel.from_local_checkpoint(
            baichuan_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            embedding_modules=embedding_modules,
            embedding_padding_modules=embed_padding_modules)
    elif lora_name == "baichuan7B-zero":
        # Test that the target_modules contain prefix
        # such as "model.layers.0.self_atten.W_pack", and
        # the test should pass.
        peft_helper = PEFTHelper.from_local_dir(baichuan_zero_lora_files,
                                                max_position_embeddings=4096)
        LoRAModel.from_local_checkpoint(
            baichuan_zero_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            embedding_modules=embedding_modules,
            embedding_padding_modules=embed_padding_modules)
    elif lora_name == "baichuan7B-zero-regex":
        # Test that the `target_modules` in the form of regular expressions,
        # such as `model\\..*(W_pack|o_proj)`, and the test should pass.
        peft_helper = PEFTHelper.from_local_dir(baichuan_regex_lora_files,
                                                max_position_embeddings=4096)
        LoRAModel.from_local_checkpoint(
            baichuan_regex_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            embedding_modules=embedding_modules,
            embedding_padding_modules=embed_padding_modules)
    else:
        # For the baichuan7B model, load chatglm3-6b's LoRA,
        # and the test should raise the following error.
        expected_error = "Please verify that the loaded LoRA module is correct"  # noqa: E501
        peft_helper = PEFTHelper.from_local_dir(chatglm3_lora_files,
                                                max_position_embeddings=4096)
        with pytest.raises(ValueError, match=expected_error):
            LoRAModel.from_local_checkpoint(
                chatglm3_lora_files,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=1,
                device="cpu",
                embedding_modules=embedding_modules,
                embedding_padding_modules=embed_padding_modules)


def test_lora_weights_mapping(baichuan_lora_files):
    supported_lora_modules = BaiChuanBaseForCausalLM.supported_lora_modules
    packed_modules_mapping = BaiChuanBaseForCausalLM.packed_modules_mapping
    embedding_modules = BaiChuanBaseForCausalLM.embedding_modules
    embed_padding_modules = BaiChuanBaseForCausalLM.embedding_padding_modules
    expected_lora_modules: List[str] = []
    for module in supported_lora_modules:
        if module in packed_modules_mapping:
            expected_lora_modules.extend(packed_modules_mapping[module])
        else:
            expected_lora_modules.append(module)

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
        },
        orig_to_new_substr={
            ".layers.": ".baichuan_layers.",
        },
    )
    peft_helper = PEFTHelper.from_local_dir(baichuan_lora_files,
                                            max_position_embeddings=4096)
    lora_model = LoRAModel.from_local_checkpoint(
        baichuan_lora_files,
        expected_lora_modules,
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        embedding_modules=embedding_modules,
        embedding_padding_modules=embed_padding_modules,
        weights_mapper=hf_to_vllm_mapper,
    )
    for name in lora_model.loras:
        assert name.startswith(hf_to_vllm_mapper.orig_to_new_prefix["model."])
        assert ".baichuan_layers." in name
