# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for MeanPoolHiddenStatesConnector.

1. Unit test with PredictableLlamaForCausalLM (deterministic values)
2. E2E accuracy tests with Llama-3.2-1B and Qwen3-0.6B
"""

import gc
import tempfile

import pytest
import torch

from vllm import LLM, ModelRegistry, SamplingParams

# =====================================================================
# Unit test: Predictable model with deterministic hidden states
# =====================================================================


@pytest.fixture(scope="module")
def predictable_llama_config_path(tmp_path_factory):
    """Create a minimal LlamaConfig for PredictableLlamaForCausalLM."""
    from transformers import AutoTokenizer, LlamaConfig

    config_dir = tmp_path_factory.mktemp("predictable_llama_mean_pool")

    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=24,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        architectures=["PredictableLlamaForCausalLM"],
    )
    config.save_pretrained(config_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        local_files_only=True,
    )
    tokenizer.save_pretrained(config_dir)

    return str(config_dir)


@pytest.fixture(scope="module", autouse=True)
def register_predictable_model():
    """Register the PredictableLlamaForCausalLM model."""
    from tests.v1.kv_connector.extract_hidden_states_integration.predictable_llama import (  # noqa: E501
        PredictableLlamaForCausalLM,
    )

    if "PredictableLlamaForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "PredictableLlamaForCausalLM", PredictableLlamaForCausalLM
        )
    yield


def test_mean_pool_with_predictable_model(predictable_llama_config_path, tmp_path):
    """Test mean pooling with deterministic hidden states.

    PredictableLlamaForCausalLM produces hidden states where layer i
    outputs values equal to i. If we extract layer 5, mean pooling over
    prompt tokens should produce a vector of all 5.0s.
    """
    layer_id = 5
    storage_path = str(tmp_path / "mean_pool")

    llm = LLM(
        model=predictable_llama_config_path,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": [layer_id]}
            },
        },
        kv_transfer_config={
            "kv_connector": "MeanPoolHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": storage_path,
            },
        },
        max_model_len=128,
        enforce_eager=True,
        trust_remote_code=True,
        load_format="dummy",
    )

    prompts = [
        "Short",
        "Medium length prompt",
        "Much longer prompt with many tokens for testing",
        "Much longer prompt with many tokens for testing",  # repeated
    ]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    outputs = llm.generate(prompts, sampling_params)
    del llm
    gc.collect()

    assert len(outputs) == len(prompts)

    for output in outputs:
        assert output.kv_transfer_params is not None, (
            "kv_transfer_params should not be None"
        )
        pooled_list = output.kv_transfer_params.get("mean_pooled_hidden_states")
        assert pooled_list is not None, (
            "mean_pooled_hidden_states not found in kv_transfer_params"
        )

        pooled = torch.tensor(pooled_list)
        # With num_heads=1 (single layer), pooled shape is [hidden_size]
        assert pooled.shape == (hidden_size,), (
            f"Expected shape ({hidden_size},), got {pooled.shape}"
        )

        # Mean pooling a constant value = that constant value
        expected = torch.full((hidden_size,), float(layer_id))
        assert torch.allclose(pooled, expected, atol=1e-4), (
            f"Expected all values to be {float(layer_id)}, "
            f"but got mean={pooled.mean():.4f}, "
            f"min={pooled.min():.4f}, max={pooled.max():.4f}"
        )


def test_mean_pool_multiple_layers(predictable_llama_config_path, tmp_path):
    """Test mean pooling with multiple hidden state layers.

    When extracting multiple layers, the pooled result should have shape
    [num_layers, hidden_size] with each row equal to the layer index.
    """
    layer_ids = [3, 7, 15]
    storage_path = str(tmp_path / "mean_pool_multi")

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
            "kv_connector": "MeanPoolHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": storage_path,
            },
        },
        max_model_len=128,
        enforce_eager=True,
        trust_remote_code=True,
        load_format="dummy",
    )

    prompts = ["Test prompt for multi-layer mean pooling"]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    outputs = llm.generate(prompts, sampling_params)
    del llm
    gc.collect()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.kv_transfer_params is not None
    pooled_list = output.kv_transfer_params.get("mean_pooled_hidden_states")
    assert pooled_list is not None

    pooled = torch.tensor(pooled_list)
    # With num_heads > 1, shape is [num_heads, hidden_size]
    assert pooled.shape == (len(layer_ids), hidden_size), (
        f"Expected shape ({len(layer_ids)}, {hidden_size}), got {pooled.shape}"
    )

    # Each row should equal the corresponding layer index
    for idx, layer_id in enumerate(layer_ids):
        expected = torch.full((hidden_size,), float(layer_id))
        assert torch.allclose(pooled[idx], expected, atol=1e-4), (
            f"Layer {layer_id} at index {idx}: expected {float(layer_id)}, "
            f"got mean={pooled[idx].mean():.4f}"
        )


# =====================================================================
# E2E accuracy tests with real models
# =====================================================================


def _get_hf_mean_pooled_hidden_states(
    model_name: str,
    prompts: list[str],
    layer_ids: list[int],
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Get mean-pooled hidden states from HuggingFace model as reference.

    Returns a list of tensors, one per prompt. Each tensor shape:
    - [hidden_size] if single layer
    - [num_layers, hidden_size] if multiple layers
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(
                device
            )
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )
            # outputs.hidden_states is a tuple:
            # (embedding, layer_0_output, layer_1_output, ..., layer_N_output)
            # layer_ids index into this tuple (0 = embedding, 1 = layer_0, etc.)
            selected = []
            for lid in layer_ids:
                # HF hidden_states[0] = embedding, [1] = after layer 0, etc.
                # vLLM's _maybe_add_hidden_state captures at idx=0
                # (pre-first-layer) and idx=i+1 (post-layer-i).
                # So layer_id in vLLM corresponds to hidden_states[layer_id]
                # in HF.
                hs = outputs.hidden_states[lid]  # [1, seq_len, hidden_size]
                hs = hs.squeeze(0)  # [seq_len, hidden_size]
                selected.append(hs.mean(dim=0, dtype=torch.float32))

            if len(selected) == 1:
                results.append(selected[0].cpu())
            else:
                results.append(torch.stack(selected, dim=0).cpu())

    del model
    gc.collect()
    torch.accelerator.empty_cache()
    return results


@pytest.mark.parametrize(
    "model_name,layer_ids",
    [
        ("meta-llama/Llama-3.2-1B", [8]),  # middle layer
        ("Qwen/Qwen3-0.6B", [14]),  # middle layer
    ],
)
@torch.inference_mode()
def test_mean_pool_accuracy_vs_hf(model_name: str, layer_ids: list[int]):
    """Compare vLLM mean-pooled hidden states against HuggingFace reference.

    Tests that the mean-pooled hidden states from vLLM's
    MeanPoolHiddenStatesConnector match HuggingFace's output within
    tolerance (accounting for FlashAttention numerical differences).
    """
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    # Get HF reference
    hf_results = _get_hf_mean_pooled_hidden_states(model_name, prompts, layer_ids)

    # Get vLLM results
    with tempfile.TemporaryDirectory() as storage_path:
        llm = LLM(
            model=model_name,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": layer_ids,
                    }
                },
            },
            kv_transfer_config={
                "kv_connector": "MeanPoolHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": storage_path,
                },
            },
            max_model_len=128,
            enforce_eager=True,
            dtype="bfloat16",
        )
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        outputs = llm.generate(prompts, sampling_params)
        del llm
        gc.collect()
        torch.accelerator.empty_cache()

    assert len(outputs) == len(prompts)

    for i, (output, hf_ref) in enumerate(zip(outputs, hf_results)):
        assert output.kv_transfer_params is not None, (
            f"Prompt {i}: kv_transfer_params should not be None"
        )
        pooled_list = output.kv_transfer_params.get("mean_pooled_hidden_states")
        assert pooled_list is not None, (
            f"Prompt {i}: mean_pooled_hidden_states not found"
        )

        vllm_pooled = torch.tensor(pooled_list)
        assert vllm_pooled.shape == hf_ref.shape, (
            f"Prompt {i}: shape mismatch: vLLM {vllm_pooled.shape} vs HF {hf_ref.shape}"
        )

        # Absolute accuracy check.
        # FlashAttention vs standard attention introduces numerical
        # differences in bf16. Max observed diff is ~0.08 on Qwen3-0.6B
        # (relative to values of magnitude ~40, this is <0.2%).
        max_diff = (vllm_pooled - hf_ref).abs().max().item()
        assert max_diff < 0.1, (
            f"Prompt {i} ({model_name}): max absolute diff {max_diff:.4f} "
            f"exceeds tolerance 0.1. "
            f"vLLM mean={vllm_pooled.mean():.4f}, "
            f"HF mean={hf_ref.mean():.4f}"
        )
