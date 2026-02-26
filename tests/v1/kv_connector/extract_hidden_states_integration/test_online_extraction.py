# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration test for OnlineHiddenStatesConnector.

Mirrors test_extraction.py but uses the async online connector instead
of ExampleHiddenStatesConnector.  Verifies that:
  1. Hidden states are written to disk asynchronously
  2. Values match the predictable dummy model pattern
  3. Compression produces valid output when enabled
  4. Percentile filtering config is accepted (functional test
     requires spec decode steps, so we just verify no crash)
"""

import gc
import os

import pytest
import torch
from safetensors import safe_open

from vllm import LLM, ModelRegistry, SamplingParams


def get_and_check_output(output, expected_shape, compressed=False):
    """Load and verify saved hidden states from a request output."""
    assert output.kv_transfer_params is not None
    hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
    assert hidden_states_path is not None

    # Online connector may add .zst extension for compressed files
    actual_path = hidden_states_path
    if compressed:
        actual_path = hidden_states_path + ".zst"
        if not os.path.exists(actual_path):
            # Fall back to uncompressed if zstd not available
            actual_path = hidden_states_path

    assert os.path.exists(actual_path), (
        f"Output file not found: {actual_path}"
    )

    if actual_path.endswith(".zst"):
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(actual_path, "rb") as f:
            raw = dctx.decompress(f.read())
        # Write to temp file for safe_open
        tmp_path = actual_path + ".tmp.safetensors"
        with open(tmp_path, "wb") as f:
            f.write(raw)
        actual_path = tmp_path

    with safe_open(actual_path, "pt") as f:
        tensor_names = f.keys()
        assert "token_ids" in tensor_names
        assert "hidden_states" in tensor_names

        token_ids = f.get_tensor("token_ids")
        hidden_states = f.get_tensor("hidden_states")

        prompt_token_ids = output.prompt_token_ids
        assert torch.equal(token_ids, torch.tensor(prompt_token_ids))
        assert hidden_states.shape == expected_shape
        assert not torch.allclose(
            hidden_states, torch.zeros_like(hidden_states)
        )

    return token_ids, hidden_states


@pytest.fixture(scope="module")
def predictable_llama_config_path(tmp_path_factory):
    """Create a minimal LlamaConfig for PredictableLlamaForCausalLM."""
    from transformers import LlamaConfig, LlamaTokenizerFast

    config_dir = tmp_path_factory.mktemp("predictable_llama")

    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=24,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        architectures=["PredictableLlamaForCausalLM"],
    )
    config.save_pretrained(config_dir)

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


def test_online_connector_with_predictable_model(
    predictable_llama_config_path, tmp_path
):
    """Test OnlineHiddenStatesConnector produces correct hidden states.

    Same test as test_extraction.py but using the async online connector
    with compression disabled for easy verification.
    """
    layer_ids = [5, 2, 10]
    num_layers = len(layer_ids)

    llm = LLM(
        model=predictable_llama_config_path,
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
            "kv_connector": "OnlineHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": str(tmp_path),
                "use_compression": False,
            },
        },
        max_model_len=128,
        enforce_eager=True,
        trust_remote_code=True,
        load_format="dummy",
    )

    prompts = [
        "Short",
        "Medium length",
        "Much longer prompt with many tokens",
        "Much longer prompt with many tokens",  # repeated
    ]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    hidden_size = llm.llm_engine.model_config.get_hidden_size()

    outputs = llm.generate(prompts, sampling_params)

    # Give async writer time to flush
    import time
    time.sleep(2)

    del llm
    gc.collect()

    assert len(outputs) == len(prompts)

    for output in outputs:
        expected_shape = (
            len(output.prompt_token_ids),
            num_layers,
            hidden_size,
        )
        _token_ids, hidden_states = get_and_check_output(
            output, expected_shape, compressed=False,
        )

        for idx, layer_id in enumerate(layer_ids):
            layer_hidden = hidden_states[:, idx, :]
            assert torch.allclose(
                layer_hidden,
                torch.full_like(layer_hidden, layer_id),
                atol=1e-5,
            ), (
                f"Layer {layer_id} at position {idx} should output "
                f"{float(layer_id)}, but got mean={layer_hidden.mean():.3f}"
            )


def test_online_connector_with_compression(
    predictable_llama_config_path, tmp_path
):
    """Test that zstd compression produces valid output."""
    try:
        import zstandard  # noqa: F401
    except ImportError:
        pytest.skip("zstandard not installed")

    layer_ids = [5, 2, 10]
    num_layers = len(layer_ids)

    llm = LLM(
        model=predictable_llama_config_path,
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
            "kv_connector": "OnlineHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": str(tmp_path),
                "use_compression": True,
                "compression_level": 1,
            },
        },
        max_model_len=128,
        enforce_eager=True,
        trust_remote_code=True,
        load_format="dummy",
    )

    prompts = ["Test compression"]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    hidden_size = llm.llm_engine.model_config.get_hidden_size()

    outputs = llm.generate(prompts, sampling_params)

    import time
    time.sleep(2)

    del llm
    gc.collect()

    assert len(outputs) == 1
    output = outputs[0]
    expected_shape = (
        len(output.prompt_token_ids),
        num_layers,
        hidden_size,
    )
    _token_ids, hidden_states = get_and_check_output(
        output, expected_shape, compressed=True,
    )

    # Verify values are correct even after compression round-trip
    for idx, layer_id in enumerate(layer_ids):
        layer_hidden = hidden_states[:, idx, :]
        assert torch.allclose(
            layer_hidden,
            torch.full_like(layer_hidden, layer_id),
            atol=1e-5,
        )


def test_online_connector_with_percentile_config(
    predictable_llama_config_path, tmp_path
):
    """Test that percentile filtering config is accepted without crash.

    With max_tokens=1 there are no decode steps, so the percentile
    tracker stays in warmup mode and captures everything.  This test
    verifies the config plumbing works end-to-end.
    """
    layer_ids = [5, 2, 10]

    llm = LLM(
        model=predictable_llama_config_path,
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
            "kv_connector": "OnlineHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": str(tmp_path),
                "use_compression": False,
                "capture_percentile": 36.0,
                "capture_window_size": 500,
                "capture_min_samples": 50,
            },
        },
        max_model_len=128,
        enforce_eager=True,
        trust_remote_code=True,
        load_format="dummy",
    )

    prompts = ["Test percentile config"]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

    outputs = llm.generate(prompts, sampling_params)

    import time
    time.sleep(2)

    del llm
    gc.collect()

    assert len(outputs) == 1
    # During warmup, everything is captured — file should exist
    assert outputs[0].kv_transfer_params is not None
