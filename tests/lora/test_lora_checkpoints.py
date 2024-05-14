import pytest

from vllm.lora.models import LoRAModel
from vllm.model_executor.models.baichuan import BaiChuanBaseForCausalLM

lora_lst = ["baichuan7B", "baichuan7B-zero", "chatglm3-6b"]


@pytest.mark.parametrize("lora_name", lora_lst)
def test_load_checkpoints(
    lora_name,
    baichuan_lora_files,
    baichuan_zero_lora_files,
    chatglm3_lora_files,
):
    supported_lora_modules = BaiChuanBaseForCausalLM.supported_lora_modules
    packed_modules_mapping = BaiChuanBaseForCausalLM.packed_modules_mapping
    embedding_modules = BaiChuanBaseForCausalLM.embedding_modules
    embed_padding_modules = BaiChuanBaseForCausalLM.embedding_padding_modules
    expected_lora_modules = []
    for module in supported_lora_modules:
        if module in packed_modules_mapping:
            expected_lora_modules.extend(packed_modules_mapping[module])
        else:
            expected_lora_modules.append(module)
    if lora_name == "baichuan7B":
        # For the baichuan7B model, load it's LoRA,
        # and the test should pass.
        LoRAModel.from_local_checkpoint(
            baichuan_lora_files,
            expected_lora_modules,
            lora_model_id=1,
            device="cpu",
            embedding_modules=embedding_modules,
            embedding_padding_modules=embed_padding_modules)
    elif lora_name == "baichuan7B-zero":
        #Test that the target_modules contain prefix
        # such as "model.layers.0.self_atten.W_pack", and
        # the test should pass.
        LoRAModel.from_local_checkpoint(
            baichuan_zero_lora_files,
            expected_lora_modules,
            lora_model_id=1,
            device="cpu",
            embedding_modules=embedding_modules,
            embedding_padding_modules=embed_padding_modules)
    else:
        # For the baichuan7B model, load chatglm3-6b's LoRA,
        # and the test should raise the following error.
        expected_error = "Please verify that the loaded LoRA module is correct"  # noqa: E501
        with pytest.raises(ValueError, match=expected_error):
            LoRAModel.from_local_checkpoint(
                chatglm3_lora_files,
                expected_lora_modules,
                lora_model_id=1,
                device="cpu",
                embedding_modules=embedding_modules,
                embedding_padding_modules=embed_padding_modules)
