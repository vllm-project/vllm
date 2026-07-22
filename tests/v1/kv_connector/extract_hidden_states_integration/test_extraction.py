# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

import pytest
import torch

from tests.utils import create_new_process_for_each_test, multi_gpu_test
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)
from vllm.platforms import current_platform


def get_and_check_output(output, expected_shape):
    assert output.kv_transfer_params is not None
    hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
    assert hidden_states_path is not None

    obj = example_hidden_states_connector.load_hidden_states(hidden_states_path)
    token_ids = obj["token_ids"]
    hidden_states = obj["hidden_states"]

    prompt_token_ids = output.prompt_token_ids
    assert torch.equal(token_ids, torch.tensor(prompt_token_ids))

    assert hidden_states.shape == expected_shape

    # Verify hidden_states are not all zeros (i.e., they were actually computed)
    assert not torch.allclose(hidden_states, torch.zeros_like(hidden_states))

    return token_ids, hidden_states


@pytest.fixture
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
        max_position_embeddings=1024,
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
    """Test hidden-state extraction with a predictable dummy model.

    Tests 3 scenarios:

    1. **Basic extraction**: non-sequential layer ordering, multiple prompts
       of varying length — verifies correct layer association and
       deterministic values.
    2. **Chunked prefill**: max_num_batched_tokens=128 with ~500-token
       prompts so each is split across multiple scheduler iterations —
       verifies hidden states are reassembled correctly.
    3. **Per-request options**: custom hidden_states_path and
       include_output_tokens — verifies per-request kv_transfer_params
       plumbing.
    """
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")

    layer_ids = [5, 2, 10]
    num_layers = len(layer_ids)
    max_num_batched_tokens = 128

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
            "kv_connector_extra_config": {
                "shared_storage_path": tmp_path,
                "allow_custom_save_path": True,
            },
        },
        max_model_len=1024,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=True,
        trust_remote_code=True,
        load_format="dummy",
    )

    hidden_size = llm.llm_engine.model_config.get_hidden_size()

    # --- Scenario 1: basic extraction with non-sequential layers ----------
    prompts = [
        "Short",
        "Medium length",
        "Much longer prompt with many tokens",
        "Much longer prompt with many tokens",  # repeated prompt
    ]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)
    for output in outputs:
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
                f"Layer {layer_id} at position {idx} should output "
                f"{float(layer_id)}, but got mean="
                f"{layer_hidden.mean():.3f}, min="
                f"{layer_hidden.min():.3f}, max={layer_hidden.max():.3f}"
            )

    # --- Scenario 2: chunked prefill with long prompts --------------------
    long_prompt = " ".join(["word"] * 500)
    chunked_prompts = [
        long_prompt,
        long_prompt + " extra tokens here",
        "Short",
    ]
    outputs = llm.generate(chunked_prompts, sampling_params)

    assert len(outputs) == len(chunked_prompts)
    for output in outputs:
        prompt_len = len(output.prompt_token_ids)
        expected_shape = (prompt_len, num_layers, hidden_size)
        _token_ids, hidden_states = get_and_check_output(output, expected_shape)

        for idx, layer_id in enumerate(layer_ids):
            layer_hidden = hidden_states[:, idx, :]
            assert torch.allclose(
                layer_hidden,
                torch.full_like(layer_hidden, layer_id),
                atol=1e-5,
            ), (
                f"Layer {layer_id} at position {idx} should output "
                f"{float(layer_id)}, but got mean="
                f"{layer_hidden.mean():.3f}, min="
                f"{layer_hidden.min():.3f}, max="
                f"{layer_hidden.max():.3f}. "
                f"prompt_len={prompt_len}, "
                f"max_num_batched_tokens={max_num_batched_tokens}"
            )

    # --- Scenario 3: per-request options ----------------------------------
    max_tokens = 5
    custom_path = os.path.join(tmp_path, "subdir", "custom.safetensors")

    sampling_params_list = [
        SamplingParams(max_tokens=max_tokens, temperature=0.0),
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            extra_args={
                "kv_transfer_params": {
                    "hidden_states_path": custom_path,
                    "include_output_tokens": True,
                }
            },
        ),
    ]
    per_req_prompts = ["Short", "Medium length"]
    outputs = llm.generate(per_req_prompts, sampling_params_list)

    # First output: prompt-only hidden states, default path
    out0 = outputs[0]
    path0 = out0.kv_transfer_params["hidden_states_path"]
    assert path0 != custom_path
    obj0 = example_hidden_states_connector.load_hidden_states(path0)
    assert torch.equal(obj0["token_ids"], torch.tensor(out0.prompt_token_ids))
    assert obj0["hidden_states"].shape == (
        len(out0.prompt_token_ids),
        num_layers,
        hidden_size,
    )
    example_hidden_states_connector.cleanup_hidden_states(path0)

    # Second output: prompt + output tokens, custom path
    out1 = outputs[1]
    assert out1.kv_transfer_params["hidden_states_path"] == custom_path
    obj1 = example_hidden_states_connector.load_hidden_states(custom_path)
    token_ids = obj1["token_ids"]
    hidden_states = obj1["hidden_states"]
    # The final output token was never an input to the model, so its hidden
    # state is not in the cache — hence the -1.
    total_tokens = len(out1.prompt_token_ids) + len(out1.outputs[0].token_ids) - 1
    assert token_ids.shape[0] == total_tokens
    assert hidden_states.shape == (total_tokens, num_layers, hidden_size)

    # Verify predictable layer values hold for all tokens (prompt + output)
    for idx, layer_id in enumerate(layer_ids):
        layer_hidden = hidden_states[:, idx, :]
        assert torch.allclose(
            layer_hidden,
            torch.full_like(layer_hidden, layer_id),
            atol=1e-5,
        )
    example_hidden_states_connector.cleanup_hidden_states(custom_path)


@create_new_process_for_each_test()
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

    assert len(outputs) == len(prompts)
    for output in outputs:
        assert output.kv_transfer_params is not None
        hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
        assert hidden_states_path is not None

        obj = example_hidden_states_connector.load_hidden_states(hidden_states_path)
        token_ids = obj["token_ids"]
        hidden_states = obj["hidden_states"]

        assert torch.equal(token_ids, torch.tensor(output.prompt_token_ids))
        assert hidden_states.shape == (
            len(output.prompt_token_ids),
            len(layer_ids),
            hidden_size,
        )


@pytest.mark.timeout(240 if current_platform.is_rocm() else 120)
@multi_gpu_test(num_gpus=2)
@create_new_process_for_each_test()
def test_extract_hidden_states_tp2():
    """Test that hidden states extraction works with tensor_parallel_size=2."""
    tmp_dir = tempfile.mkdtemp()
    layer_ids = [5, 11, 17]
    hidden_size = 1024  # Qwen/Qwen3-0.6B hidden_size

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        tensor_parallel_size=2,
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
            "kv_connector_extra_config": {"shared_storage_path": tmp_dir},
        },
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        load_format="dummy",
    )

    prompts = ["Hello world", "Test prompt with several tokens"]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)
    for output in outputs:
        assert output.kv_transfer_params is not None
        hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
        assert hidden_states_path is not None

        obj = example_hidden_states_connector.load_hidden_states(hidden_states_path)
        token_ids = obj["token_ids"]
        hidden_states = obj["hidden_states"]

        assert torch.equal(token_ids, torch.tensor(output.prompt_token_ids))
        assert hidden_states.shape == (
            len(output.prompt_token_ids),
            len(layer_ids),
            hidden_size,
        )
