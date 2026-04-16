# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.baichuan import BaiChuanBaseForCausalLM
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
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


def _write_moe_lora_checkpoint(lora_dir: Path, module_prefix: str) -> None:
    (lora_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 1,
                "lora_alpha": 1,
                "target_modules": ["experts", "down_proj"],
                "bias": "none",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            f"{module_prefix}.lora_A.weight": torch.zeros(1, 1),
            f"{module_prefix}.lora_B.weight": torch.zeros(1, 1),
        },
        lora_dir / "adapter_model.safetensors",
    )


@pytest.mark.parametrize(
    (
        "module_prefix",
        "expected_lora_modules",
        "weights_mapper",
        "expected_module_name",
    ),
    [
        (
            "base_model.model.model.layers.0.mlp.experts",
            {"experts"},
            None,
            "model.layers.0.mlp.experts",
        ),
        (
            "base_model.model.model.layers.0.mlp.experts.0.down_proj",
            {"experts.0.down_proj"},
            None,
            "model.layers.0.mlp.experts.0.down_proj",
        ),
        (
            "base_model.model.model.layers.0.mlp.experts.0.down_proj",
            {"down_proj"},
            WeightsMapper(orig_to_new_prefix={"model.": "language_model.model."}),
            "language_model.model.layers.0.mlp.experts.0.down_proj",
        ),
    ],
    ids=["experts-module", "expert-qualified", "mapped-leaf"],
)
def test_moe_lora_module_validation(
    tmp_path: Path,
    module_prefix: str,
    expected_lora_modules: set[str],
    weights_mapper: WeightsMapper | None,
    expected_module_name: str,
) -> None:
    _write_moe_lora_checkpoint(tmp_path, module_prefix)

    peft_helper = PEFTHelper.from_local_dir(str(tmp_path), max_position_embeddings=4096)
    lora_model = LoRAModel.from_local_checkpoint(
        str(tmp_path),
        expected_lora_modules,
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        weights_mapper=weights_mapper,
    )

    assert list(lora_model.loras) == [expected_module_name]


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
