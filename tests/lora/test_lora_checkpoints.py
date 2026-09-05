# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import parse_fine_tuned_lora_name
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
        "base_model.model.model.language_model.layers.9.moe.experts."
        "gate_up_proj.lora_B.weight"
    )
    assert parse_fine_tuned_lora_name(name, mapper) == (
        "model.layers.9.moe.gate_up_proj",
        False,
    )


@pytest.mark.parametrize(
    "model_cls,prefix",
    [
        (Gemma4ForCausalLM, "model"),
        (Gemma4ForConditionalGeneration, "language_model.model"),
    ],
)
def test_gemma4_stacked_expert_lora_weights_mapping(model_cls, prefix):
    """Stacked (PEFT ``target_parameters``) expert LoRA tensors must be
    rewritten onto the ``.moe.experts`` parent module.

    These adapters store the expert deltas as two tensor pairs per layer that
    name the ``experts`` module itself rather than a child module:
    ``...experts.base_layer.lora_{A,B}`` (gate_up_proj) and
    ``...experts.lora_{A,B}`` (down_proj). ``_convert_3d_to_2d_moe_lora``
    looks these up under the registered module name, which carries the
    ``.moe.`` segment, so the mapper has to add it here as well.
    """
    mapper = model_cls.hf_to_vllm_mapper
    base = "base_model.model.model.language_model.layers.9.experts"
    assert parse_fine_tuned_lora_name(f"{base}.base_layer.lora_A.weight", mapper) == (
        f"{prefix}.layers.9.moe.experts.base_layer",
        True,
    )
    assert parse_fine_tuned_lora_name(f"{base}.base_layer.lora_B.weight", mapper) == (
        f"{prefix}.layers.9.moe.experts.base_layer",
        False,
    )
    assert parse_fine_tuned_lora_name(f"{base}.lora_A.weight", mapper) == (
        f"{prefix}.layers.9.moe.experts",
        True,
    )
    assert parse_fine_tuned_lora_name(f"{base}.lora_B.weight", mapper) == (
        f"{prefix}.layers.9.moe.experts",
        False,
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        # ModelOpt `quantized_layers` entries name the parent module exactly.
        ("model.language_model.layers.9.experts", "model.layers.9.moe.experts"),
        # Already-normalized names are not rewritten twice.
        ("model.layers.9.moe.experts", "model.layers.9.moe.experts"),
        # Non-LoRA children of `experts` keep their names.
        (
            "model.language_model.layers.9.experts.gate_up_proj",
            "model.layers.9.experts.gate_up_proj",
        ),
        (
            "model.language_model.layers.9.experts.down_proj_packed",
            "model.layers.9.experts.down_proj_packed",
        ),
    ],
)
def test_gemma4_expert_parent_mapper_non_lora_names(name, expected):
    """The expert-parent rewrite stays limited to the parent module itself and
    to stacked expert LoRA tensors."""
    assert Gemma4ForCausalLM.hf_to_vllm_mapper._map_name(name) == expected
