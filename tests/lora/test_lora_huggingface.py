# SPDX-License-Identifier: Apache-2.0
import json
import os

import pytest
import safetensors.torch
import torch

from vllm.lora.models import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import (get_adapter_absolute_path,
                             parse_fine_tuned_lora_name)
from vllm.model_executor.models.llama import LlamaForCausalLM

# Provide absolute path and huggingface lora ids
lora_fixture_name = ["sql_lora_files", "sql_lora_huggingface_id"]
LLAMA_LORA_MODULES = [
    "qkv_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
    "embed_tokens",
    "lm_head",
]


@pytest.mark.parametrize("lora_fixture_name", lora_fixture_name)
def test_load_checkpoints_from_huggingface(lora_fixture_name, request):
    lora_name = request.getfixturevalue(lora_fixture_name)
    packed_modules_mapping = LlamaForCausalLM.packed_modules_mapping
    embedding_modules = LlamaForCausalLM.embedding_modules
    embed_padding_modules = LlamaForCausalLM.embedding_padding_modules
    expected_lora_modules: list[str] = []
    for module in LLAMA_LORA_MODULES:
        if module in packed_modules_mapping:
            expected_lora_modules.extend(packed_modules_mapping[module])
        else:
            expected_lora_modules.append(module)

    lora_path = get_adapter_absolute_path(lora_name)

    # lora loading should work for either absolute path and hugggingface id.
    peft_helper = PEFTHelper.from_local_dir(lora_path, 4096)
    lora_model = LoRAModel.from_local_checkpoint(
        lora_path,
        expected_lora_modules,
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        embedding_modules=embedding_modules,
        embedding_padding_modules=embed_padding_modules,
    )

    # Assertions to ensure the model is loaded correctly
    assert lora_model is not None, "LoRAModel is not loaded correctly"


def test_load_dora_checkpoint_from_huggingface(dora_files):
    """Test that a DoRA adapter can be loaded from HuggingFace correctly."""
    # DoRA adapters for LLaMA typically target these modules
    expected_modules = ["q_proj", "v_proj"]

    # For LLaMA model
    embedding_modules = LlamaForCausalLM.embedding_modules
    embed_padding_modules = LlamaForCausalLM.embedding_padding_modules

    # Get the adapter's path
    lora_path = get_adapter_absolute_path(dora_files)

    # Verify that the adapter config has DoRA settings
    with open(os.path.join(lora_path, "adapter_config.json")) as f:
        config = json.load(f)
        # Verify that the adapter is indeed a DoRA adapter
        assert config.get("use_dora", False) is True
        assert "target_modules" in config

    # Read the weights file to check for magnitude vectors
    weights_path = os.path.join(lora_path, "adapter_model.safetensors")
    tensors = safetensors.torch.load_file(weights_path)

    # Check that magnitude vectors exist in the tensors
    magnitude_vectors = [
        name for name in tensors.keys()
        if name.endswith("lora_magnitude_vector")
    ]
    assert len(
        magnitude_vectors) > 0, "No magnitude vectors found in DoRA adapter"

    # Load the adapter
    peft_helper = PEFTHelper.from_local_dir(lora_path,
                                            max_position_embeddings=4096)
    assert peft_helper.use_dora is True, "DoRA flag not set correctly in PEFTHelper"

    # Load the LoRA model
    lora_model = LoRAModel.from_local_checkpoint(
        lora_path,
        expected_modules,
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        embedding_modules=embedding_modules,
        embedding_padding_modules=embed_padding_modules,
    )

    # Verify that the model loaded correctly
    assert lora_model is not None, "DoRA model failed to load"
    assert lora_model.id == 1, "Wrong model ID"

    # Check that magnitudes are loaded correctly
    for module_name, lora_weights in lora_model.loras.items():
        # Make sure the magnitude parameter is loaded
        assert hasattr(
            lora_weights,
            "magnitude_param"), f"No magnitude parameter for {module_name}"
        assert (lora_weights.magnitude_param
                is not None), f"Magnitude parameter is None for {module_name}"
        # Verify it has the correct shape
        assert (lora_weights.magnitude_param.shape[0] == lora_weights.lora_b.
                shape[1]), f"Magnitude shape mismatch for {module_name}"


def test_dora_weight_structure_in_huggingface(dora_files):
    """Test the structure of DoRA weights in a HuggingFace adapter."""
    lora_path = get_adapter_absolute_path(dora_files)

    # Load the weights
    weights_path = os.path.join(lora_path, "adapter_model.safetensors")
    tensors = safetensors.torch.load_file(weights_path)

    # Get all the keys in the safetensors file
    all_keys = list(tensors.keys())

    # Identify modules with magnitude vectors
    magnitude_vector_keys = [
        key for key in all_keys if key.endswith("lora_magnitude_vector")
    ]
    assert len(magnitude_vector_keys) > 0, "No magnitude vectors found"

    # For each magnitude vector, test that the corresponding LoRA A and B matrices exist
    for magnitude_key in magnitude_vector_keys:
        # Extract module name from magnitude vector key
        module_name, _, _, is_magnitude = parse_fine_tuned_lora_name(
            magnitude_key)
        assert is_magnitude, f"Failed to identify {magnitude_key} as a magnitude vector"

        # Construct expected LoRA A and B keys
        lora_a_pattern = None
        lora_b_pattern = None

        # Find matching A and B weights by pattern
        for key in all_keys:
            if key.startswith(
                    magnitude_key.split(".lora_magnitude_vector")
                [0]) and key.endswith("lora_A.weight"):
                lora_a_pattern = key
            elif key.startswith(
                    magnitude_key.split(".lora_magnitude_vector")
                [0]) and key.endswith("lora_B.weight"):
                lora_b_pattern = key

        # Assert that both LoRA A and B weights exist
        assert (lora_a_pattern
                is not None), f"Missing LoRA A weight for module {module_name}"
        assert (lora_b_pattern
                is not None), f"Missing LoRA B weight for module {module_name}"

        # Check tensor shapes
        magnitude_tensor = tensors[magnitude_key]
        lora_a_tensor = tensors[lora_a_pattern]
        lora_b_tensor = tensors[lora_b_pattern]

        # Magnitude should be a 1D tensor matching the output dimension of LoRA B
        assert magnitude_tensor.dim(
        ) == 1, f"Magnitude should be 1D for {module_name}"

        # LoRA matrices should be 2D
        assert lora_a_tensor.dim(
        ) == 2, f"LoRA A should be 2D for {module_name}"
        assert lora_b_tensor.dim(
        ) == 2, f"LoRA B should be 2D for {module_name}"

        # Magnitude vector shape should match output dimension
        assert magnitude_tensor.shape[0] == lora_b_tensor.shape[0], (
            f"Magnitude vector shape mismatch for {module_name}: "
            f"{magnitude_tensor.shape[0]} vs {lora_b_tensor.shape[0]}")


def test_dora_vector_values_in_huggingface(dora_files):
    """Test that the magnitude vectors in a DoRA adapter have reasonable values."""
    lora_path = get_adapter_absolute_path(dora_files)

    # Load the weights
    weights_path = os.path.join(lora_path, "adapter_model.safetensors")
    tensors = safetensors.torch.load_file(weights_path)

    # Get all magnitude vectors
    magnitude_keys = [
        key for key in tensors.keys() if key.endswith("lora_magnitude_vector")
    ]
    assert len(magnitude_keys) > 0, "No magnitude vectors found"

    for magnitude_key in magnitude_keys:
        magnitude_tensor = tensors[magnitude_key]

        # Check that the magnitude vector has non-negative values (reasonable for DoRA)
        assert torch.all(
            magnitude_tensor >=
            0), f"Found negative magnitude values in {magnitude_key}"

        # Check that the magnitude vector has some variation (trained values should vary)
        assert (
            magnitude_tensor.min() != magnitude_tensor.max()
        ), f"Magnitude vector {magnitude_key} has constant values, which is unusual for a trained DoRA model"

        # Basic statistical checks for a reasonably trained model
        mean = magnitude_tensor.mean().item()
        std = magnitude_tensor.std().item()

        # A trained model should have reasonable statistics in magnitudes
        assert (
            mean > 0
        ), f"Magnitude vector {magnitude_key} has zero mean, which is unusual"
        assert (
            std > 0
        ), f"Magnitude vector {magnitude_key} has zero std, which is unusual"

        # Log statistics for reference (valuable for debugging but not strict assertions)
        min_val = magnitude_tensor.min().item()
        max_val = magnitude_tensor.max().item()
        # These values are expected to vary by model, but extremely large or small values might
        # indicate issues in training/loading
        assert min_val < max_val, f"Min ({min_val}) should be less than max ({max_val})"
