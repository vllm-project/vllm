# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import os

import pytest
import torch
from safetensors import safe_open

from vllm import LLM, ModelRegistry, SamplingParams


def get_and_check_output(output, expected_shape):
    assert output.kv_transfer_params is not None
    hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
    assert hidden_states_path is not None
    assert os.path.exists(hidden_states_path)

    # Load and verify the saved tensors
    with safe_open(hidden_states_path, "pt") as f:
        # Check that token_ids and hidden_states are present
        tensor_names = f.keys()
        assert "token_ids" in tensor_names
        assert "hidden_states" in tensor_names

        token_ids = f.get_tensor("token_ids")
        hidden_states = f.get_tensor("hidden_states")

        prompt_token_ids = output.prompt_token_ids
        assert torch.equal(token_ids, torch.tensor(prompt_token_ids))

        assert hidden_states.shape == expected_shape

        # Verify hidden_states are not all zeros (i.e., they were actually computed)
        assert not torch.allclose(hidden_states, torch.zeros_like(hidden_states))

    return token_ids, hidden_states


@pytest.fixture(scope="module")
def predictable_llama_config_path(tmp_path_factory):
    """Create a minimal LlamaConfig for PredictableLlamaForCausalLM."""
    from transformers import LlamaConfig, LlamaTokenizerFast

    config_dir = tmp_path_factory.mktemp("predictable_llama")

    # Create a minimal Llama config with small dimensions
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=24,  # Enough layers to test various layer_ids
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        architectures=["PredictableLlamaForCausalLM"],
    )

    # Save config
    config.save_pretrained(config_dir)

    # Create a simple tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cache_dir=os.path.expanduser("~/.cache/huggingface"),
    )
    tokenizer.save_pretrained(config_dir)

    return str(config_dir)


@pytest.fixture(scope="module", autouse=True)
def register_predictable_model():
    """Register the PredictableLlamaForCausalLM model."""
    from .predictable_llama import PredictableLlamaForCausalLM

    if "PredictableLlamaForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "PredictableLlamaForCausalLM", PredictableLlamaForCausalLM
        )
    yield


def test_extract_hidden_states_with_predictable_dummy_model(
    predictable_llama_config_path, tmp_path, monkeypatch
):
    """Comprehensive test using a predictable dummy model with synthetic weights.

    The PredictableLlamaForCausalLM outputs deterministic hidden states where
    each layer produces values equal to (layer_index). This test verifies:
    1. Hidden states are correctly extracted from requested layers
    2. Values match the expected predictable pattern
    3. Layer ordering is preserved correctly (non-sequential layer IDs)
    4. Multiple prompts of different lengths produce consistent layer values
    """
    # Force fork so the engine worker inherits the autouse fixture's
    # ModelRegistry.register_model("PredictableLlamaForCausalLM", ...).
    # Spawn (the CI default) starts a fresh Python process that wouldn't
    # see the registration.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")

    # Test with non-sequential layer ordering to verify correct association
    layer_ids = [5, 2, 10]
    num_layers = len(layer_ids)

    llm = LLM(
        model=predictable_llama_config_path,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": tmp_path},
        },
        max_model_len=128,
        enforce_eager=True,
        trust_remote_code=True,
        load_format="dummy",  # Don't try to load real weights
    )

    # Test with multiple prompts of different lengths
    prompts = [
        "Short",
        "Medium length",
        "Much longer prompt with many tokens",
        "Much longer prompt with many tokens",  # repeated prompt
    ]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    outputs = llm.generate(prompts, sampling_params)
    del llm
    gc.collect()

    assert len(outputs) == len(prompts)

    for output in outputs:
        # hidden_states shape is [prompt_len, num_hidden_layers, hidden_size]
        expected_shape = (
            len(output.prompt_token_ids),
            num_layers,
            hidden_size,
        )
        _token_ids, hidden_states = get_and_check_output(output, expected_shape)

        for idx, layer_id in enumerate(layer_ids):
            layer_hidden = hidden_states[:, idx, :]
            assert torch.allclose(
                layer_hidden,
                torch.full_like(layer_hidden, layer_id),
                atol=1e-5,
            ), (
                f"Layer {layer_id} at position {idx} should output {float(layer_id)}, "
                f"but got mean={layer_hidden.mean():.3f}, "
                f"min={layer_hidden.min():.3f}, max={layer_hidden.max():.3f}"
            )


def test_extract_hidden_states_qwen35_hybrid_smoke(tmp_path):
    """Smoke test for Qwen3.5 hybrid (mamba + full-attention) models.
    Uses load_format="dummy" to just check shape/plumbing.
    """
    layer_ids = [5, 11, 17]
    hidden_size = 1024  # Qwen/Qwen3.5-0.8B hidden_size

    llm = LLM(
        model="Qwen/Qwen3.5-0.8B",
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(tmp_path)},
        },
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        load_format="dummy",
    )

    prompts = ["Hello world", "Test prompt with several tokens"]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)
    del llm
    gc.collect()

    assert len(outputs) == len(prompts)
    for output in outputs:
        assert output.kv_transfer_params is not None
        hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
        assert hidden_states_path is not None
        assert os.path.exists(hidden_states_path)

        with safe_open(hidden_states_path, "pt") as f:
            token_ids = f.get_tensor("token_ids")
            hidden_states = f.get_tensor("hidden_states")

        assert torch.equal(token_ids, torch.tensor(output.prompt_token_ids))
        assert hidden_states.shape == (
            len(output.prompt_token_ids),
            len(layer_ids),
            hidden_size,
        )
