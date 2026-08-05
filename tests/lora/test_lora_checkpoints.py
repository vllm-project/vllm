# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
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


def _linear_peft_helper(*, use_dora: bool) -> PEFTHelper:
    return PEFTHelper(
        r=2,
        lora_alpha=4,
        target_modules=["linear"],
        use_dora=use_dora,
    )


@pytest.fixture
def disable_lora_pin_memory(monkeypatch):
    monkeypatch.setattr("vllm.lora.lora_model.PIN_MEMORY", False)


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
        "lora_a",
    )


def test_gemma4_moe_lora_weights_mapping():
    mapper = Gemma4ForCausalLM.hf_to_vllm_mapper
    name = (
        "base_model.model.model.language_model.layers.9.moe.experts."
        "gate_up_proj.lora_B.weight"
    )
    assert parse_fine_tuned_lora_name(name, mapper) == (
        "model.layers.9.moe.gate_up_proj",
        "lora_b",
    )


@pytest.mark.skip_global_cleanup
def test_load_dora_tensors(disable_lora_pin_memory):
    tensors = {
        "base_model.model.linear.lora_A.weight": torch.randn(2, 3),
        "base_model.model.linear.lora_B.weight": torch.randn(4, 2),
        "base_model.model.linear.lora_magnitude_vector": torch.randn(4),
    }

    lora_model = LoRAModel.from_lora_tensors(
        1,
        tensors,
        _linear_peft_helper(use_dora=True),
        device="cpu",
    )

    lora = lora_model.loras["linear"]
    assert lora.use_dora
    assert isinstance(lora.lora_a, torch.Tensor)
    assert isinstance(lora.lora_b, torch.Tensor)
    assert isinstance(lora.lora_magnitude_vector, torch.Tensor)
    assert torch.equal(lora.lora_a, tensors["base_model.model.linear.lora_A.weight"])
    assert torch.equal(lora.lora_b, tensors["base_model.model.linear.lora_B.weight"])
    assert torch.equal(
        lora.lora_magnitude_vector,
        tensors["base_model.model.linear.lora_magnitude_vector"],
    )


@pytest.mark.skip_global_cleanup
def test_pack_dora_lora_weights() -> None:
    dtype = torch.float32
    rank = 2
    loras = [
        LoRALayerWeights(
            f"proj_{i}",
            rank=rank,
            lora_alpha=rank,
            lora_a=torch.full((rank, 4), 0.1 * (i + 1), dtype=dtype),
            lora_b=torch.full((out_size, rank), 0.2 * (i + 1), dtype=dtype),
            lora_magnitude_vector=torch.full((out_size,), i + 1, dtype=dtype),
            use_dora=True,
        )
        for i, out_size in enumerate([4, 2, 2])
    ]

    packed_lora = PackedLoRALayerWeights.pack(loras)

    assert packed_lora.use_dora
    assert isinstance(packed_lora.lora_magnitude_vector, list)
    for i, lora in enumerate(loras):
        torch.testing.assert_close(packed_lora.lora_a[i], lora.lora_a)
        torch.testing.assert_close(packed_lora.lora_b[i], lora.lora_b)
        torch.testing.assert_close(
            packed_lora.lora_magnitude_vector[i], lora.lora_magnitude_vector
        )


@pytest.mark.skip_global_cleanup
def test_load_lora_tensors_rejects_unconfigured_dora_magnitude(
    disable_lora_pin_memory,
):
    tensors = {
        "base_model.model.linear.lora_A.weight": torch.randn(2, 3),
        "base_model.model.linear.lora_B.weight": torch.randn(4, 2),
        "base_model.model.linear.lora_magnitude_vector": torch.randn(4),
    }

    with pytest.raises(ValueError, match="use_dora=False"):
        LoRAModel.from_lora_tensors(
            1,
            tensors,
            _linear_peft_helper(use_dora=False),
            device="cpu",
        )


@pytest.mark.skip_global_cleanup
def test_load_dora_tensors_rejects_missing_magnitude(disable_lora_pin_memory):
    tensors = {
        "base_model.model.linear.lora_A.weight": torch.randn(2, 3),
        "base_model.model.linear.lora_B.weight": torch.randn(4, 2),
    }

    with pytest.raises(ValueError, match="missing lora_magnitude_vector"):
        LoRAModel.from_lora_tensors(
            1,
            tensors,
            _linear_peft_helper(use_dora=True),
            device="cpu",
        )
