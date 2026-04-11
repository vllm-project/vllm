# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.baichuan import BaiChuanBaseForCausalLM
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
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

# ---------------------------------------------------------------------------
# Gemma4ForCausalLM LoRA weight mapping tests
# ---------------------------------------------------------------------------


def test_gemma4_lora_weights_mapping():
    """Test that LoRA adapter names from the conditional wrapper are
    correctly remapped to the text-only model namespace via the
    ``hf_to_vllm_mapper`` on ``Gemma4ForCausalLM``."""
    mapper = Gemma4ForCausalLM.hf_to_vllm_mapper
    name = (
        "base_model.model.model.language_model.layers.9"
        ".mlp.down_proj.lora_A.weight"
    )
    assert parse_fine_tuned_lora_name(name, mapper) == (
        "model.layers.9.mlp.down_proj",
        True,
    )


def test_gemma4_attention_lora_weights_mapping():
    """Test that attention projection LoRA names are mapped correctly."""
    mapper = Gemma4ForCausalLM.hf_to_vllm_mapper
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        name = (
            f"base_model.model.model.language_model.layers.3"
            f".self_attn.{proj}.lora_A.weight"
        )
        expected_module = f"model.layers.3.self_attn.{proj}"
        result = parse_fine_tuned_lora_name(name, mapper)
        assert result == (expected_module, True), (
            f"Unexpected mapping for {proj}: {result}"
        )


def test_gemma4_gate_up_proj_lora_weights_mapping():
    """Test that gate_up_proj LoRA names are mapped correctly."""
    mapper = Gemma4ForCausalLM.hf_to_vllm_mapper
    name = (
        "base_model.model.model.language_model.layers.5"
        ".mlp.gate_up_proj.lora_B.weight"
    )
    assert parse_fine_tuned_lora_name(name, mapper) == (
        "model.layers.5.mlp.gate_up_proj",
        False,
    )


def test_gemma4_moe_lora_weights_mapping():
    """Test that MoE expert adapter names are remapped: the checkpoint
    uses ``moe.experts.gate_up_proj`` while the text-only model uses
    ``moe.gate_up_proj``."""
    mapper = Gemma4ForCausalLM.hf_to_vllm_mapper
    name = (
        "base_model.model.model.language_model.layers.9.moe.experts."
        "gate_up_proj.lora_B.weight"
    )
    assert parse_fine_tuned_lora_name(name, mapper) == (
        "model.layers.9.moe.gate_up_proj",
        False,
    )


def test_gemma4_moe_down_proj_lora_weights_mapping():
    """Test that MoE down_proj expert adapter names are remapped."""
    mapper = Gemma4ForCausalLM.hf_to_vllm_mapper
    name = (
        "base_model.model.model.language_model.layers.12.moe.experts."
        "down_proj.lora_A.weight"
    )
    assert parse_fine_tuned_lora_name(name, mapper) == (
        "model.layers.12.moe.down_proj",
        True,
    )


# ---------------------------------------------------------------------------
# Gemma4ForConditionalGeneration LoRA support tests
# ---------------------------------------------------------------------------


def test_gemma4_conditional_supports_lora():
    """Verify that the multimodal wrapper exposes LoRA support."""
    assert hasattr(Gemma4ForConditionalGeneration, "supports_lora")
    assert Gemma4ForConditionalGeneration.supports_lora is True


def test_gemma4_conditional_packed_modules():
    """Verify packed_modules_mapping on the multimodal model."""
    mapping = Gemma4ForConditionalGeneration.packed_modules_mapping
    assert "qkv_proj" in mapping
    assert set(mapping["qkv_proj"]) == {"q_proj", "k_proj", "v_proj"}
    assert "gate_up_proj" in mapping
    assert set(mapping["gate_up_proj"]) == {"gate_proj", "up_proj"}


def test_gemma4_conditional_hf_to_vllm_mapper():
    """Verify that the multimodal model exposes a weights mapper."""
    mapper = Gemma4ForConditionalGeneration.hf_to_vllm_mapper
    assert isinstance(mapper, WeightsMapper)
    # Should map language_model prefix
    assert "model.language_model." in mapper.orig_to_new_prefix


def test_gemma4_causal_lm_supports_lora():
    """Verify that the text-only model exposes LoRA support."""
    assert hasattr(Gemma4ForCausalLM, "supports_lora")
    assert Gemma4ForCausalLM.supports_lora is True


def test_gemma4_causal_packed_modules():
    """Verify packed_modules_mapping on the text-only model."""
    mapping = Gemma4ForCausalLM.packed_modules_mapping
    assert "qkv_proj" in mapping
    assert set(mapping["qkv_proj"]) == {"q_proj", "k_proj", "v_proj"}
    assert "gate_up_proj" in mapping
    assert set(mapping["gate_up_proj"]) == {"gate_proj", "up_proj"}
