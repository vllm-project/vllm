# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest import mock

import numpy as np
import pytest
import torch
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, PretrainedConfig

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
)
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.load import LoadConfig
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_hf_text_config
from vllm.transformers_utils.configs.extract_hidden_states import (
    ExtractHiddenStatesConfig,
)
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

model_dir = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE_TYPE = current_platform.device_type


def _create_proposer(
    num_speculative_tokens: int = 1,
    layer_ids: list[int] | None = None,
) -> ExtractHiddenStatesProposer:
    """Create an ExtractHiddenStatesProposer for testing."""
    if layer_ids is None:
        layer_ids = [1, 2, 3, 4]

    model_config = ModelConfig(model=model_dir, runner="generate", max_model_len=100)

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        method="extract_hidden_states",
        num_speculative_tokens=num_speculative_tokens,
        draft_model_config={
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": layer_ids,
            }
        },
    )

    device = DEVICE_TYPE
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=device),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        attention_config=AttentionConfig(),
    )

    return ExtractHiddenStatesProposer(vllm_config=vllm_config, device=device)


def test_proposer_initialization():
    """Test that the proposer initializes correctly with the right parameters."""
    layer_ids = [1, 2, 3, 4]
    proposer = _create_proposer(num_speculative_tokens=1, layer_ids=layer_ids)

    assert proposer.num_hidden_states == len(layer_ids)
    assert proposer.vllm_config.speculative_config is not None
    assert proposer.vllm_config.speculative_config.num_speculative_tokens == 1

    # Verify the hidden states buffer is correctly shaped
    expected_shape = (
        proposer.max_num_tokens,
        len(layer_ids),
        proposer.hidden_size,
    )
    assert proposer.hidden_states.shape == expected_shape


def test_proposer_initialization_missing_layer_ids():
    """Test that initialization fails when layer_ids are not provided."""
    model_config = ModelConfig(model=model_dir, runner="generate", max_model_len=100)

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        method="extract_hidden_states",
        num_speculative_tokens=1,
        draft_model_config={
            "hf_config": {}  # Missing eagle_aux_hidden_state_layer_ids
        },
    )

    device = DEVICE_TYPE
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=device),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        attention_config=AttentionConfig(),
    )

    with pytest.raises(
        ValueError, match="eagle_aux_hidden_state_layer_ids must be set"
    ):
        ExtractHiddenStatesProposer(vllm_config=vllm_config, device=device)


def test_prepare_next_token_ids_padded():
    """
    Test for prepare_next_token_ids_padded with extract_hidden_states.

    Since num_speculative_tokens == 1, sampled_token_ids has shape (batch_size, 1).
    For each request we either use the sampled token (if valid and not discarded)
    or a backup token from the request state.
    """
    device = torch.device(DEVICE_TYPE)

    num_requests = 4
    req_ids = [f"req_{i + 1}" for i in range(num_requests)]
    mock_input_batch = mock.MagicMock(spec=InputBatch)
    mock_input_batch.req_ids = req_ids
    mock_input_batch.num_reqs = num_requests
    mock_input_batch.vocab_size = 100
    mock_input_batch.num_tokens_no_spec = np.array([5] * num_requests)

    mock_requests = {}
    for req_id in req_ids:
        mock_request = mock.MagicMock(spec=CachedRequestState)
        # Each request will have a backup next token id of 10, 20, 30, 40
        mock_request.get_token_id.return_value = int(req_id.split("_")[1]) * 10
        mock_requests[req_id] = mock_request

    # explicitly discard the last request
    discarded_req_mask = torch.tensor(
        [False, False, False, True], dtype=torch.bool, device=device
    )

    # With num_speculative_tokens=1, sampled_token_ids has shape [batch_size, 1]
    sampled_token_ids = torch.tensor(
        [
            [1],  # valid, use 1
            [4],  # valid, use 4
            [-1],  # invalid, use backup token "30"
            [2],  # explicitly discarded, use backup token "40"
        ],
        dtype=torch.int32,
        device=device,
    )

    expected_next_token_ids_cpu = [1, 4, 30, 40]
    expected_next_token_ids_tensor = torch.tensor(
        expected_next_token_ids_cpu, dtype=torch.int32, device=device
    )

    proposer = _create_proposer(num_speculative_tokens=1)

    # valid_sampled_tokens_count tracks if token is valid (not -1 and in vocab range)
    # It doesn't depend on whether the request is discarded
    expected_valid_sampled_tokens_count = torch.tensor(
        [1, 1, 0, 1], dtype=torch.int32, device=device
    )

    next_token_ids, valid_sampled_tokens_count = proposer.prepare_next_token_ids_padded(
        sampled_token_ids,
        mock_requests,
        mock_input_batch,
        discarded_req_mask,
    )

    assert torch.equal(next_token_ids, expected_next_token_ids_tensor)
    assert torch.equal(valid_sampled_tokens_count, expected_valid_sampled_tokens_count)


def test_propose():
    """
    Test the propose() method of ExtractHiddenStatesProposer.

    This should:
    1. Accept target hidden states and sampled token IDs
    2. Return the sampled tokens as "draft" tokens (shape [batch_size, 1])
    3. Cache the hidden states in the model's KV cache
    """
    device = torch.device(DEVICE_TYPE)

    # Setup test parameters
    batch_size = 2
    num_tokens = 5
    num_hidden_layers = 4

    proposer = _create_proposer(
        num_speculative_tokens=1, layer_ids=list(range(num_hidden_layers))
    )
    hidden_size = proposer.hidden_size

    # Create mock model
    model_mock = mock.MagicMock()
    proposer.model = model_mock

    # Mock attention layer names
    proposer.attn_layer_names = ["cache_only_layers.28"]

    # Mock attention metadata builder
    mock_attn_metadata = mock.MagicMock()
    mock_attn_metadata_builder = mock.MagicMock()
    mock_attn_metadata_builder.build_for_drafting.return_value = mock_attn_metadata
    proposer.attn_metadata_builder = mock_attn_metadata_builder

    # Create input tensors
    batch_spec = BatchSpec(
        seq_lens=[3, 2],
        query_lens=[3, 2],
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=16,
        device=device,
    )

    # Create target hidden states: list of tensors, one per layer
    # Each tensor has shape [num_tokens, hidden_size]
    target_hidden_states = [
        torch.randn(num_tokens, hidden_size, dtype=proposer.dtype, device=device)
        for _ in range(num_hidden_layers)
    ]

    # Sampled token IDs from target model
    sampled_token_ids = torch.tensor(
        [42, 60], dtype=torch.int32, device=device
    ).unsqueeze(-1)

    # Call propose
    draft_tokens = proposer.propose(
        sampled_token_ids=sampled_token_ids,
        target_hidden_states=target_hidden_states,
        common_attn_metadata=common_attn_metadata,
        slot_mappings=None,
    )

    # Verify draft tokens match sampled tokens
    # Shape should be [batch_size, 1] for num_speculative_tokens=1
    assert draft_tokens.shape == (batch_size, 1)
    assert torch.equal(draft_tokens, sampled_token_ids)

    # Verify the model was called
    model_mock.assert_called_once()

    # Verify hidden states were copied to the buffer The stacked hidden states
    # should have shape [num_tokens, num_hidden_layers, hidden_size]
    expected_stacked = torch.stack(target_hidden_states, dim=1)
    assert torch.allclose(
        proposer.hidden_states[:num_tokens], expected_stacked, atol=1e-6
    )


@pytest.mark.parametrize("num_hidden_layers", [1, 4, 8])
def test_propose_different_layer_counts(num_hidden_layers):
    """Test that propose works correctly with different numbers of hidden layers."""
    device = torch.device(DEVICE_TYPE)

    batch_size = 2
    num_tokens = 5

    proposer = _create_proposer(
        num_speculative_tokens=1, layer_ids=list(range(num_hidden_layers))
    )
    hidden_size = proposer.hidden_size

    # Setup mocks
    model_mock = mock.MagicMock()
    proposer.model = model_mock
    proposer.attn_layer_names = ["cache_only_layers.28"]

    mock_attn_metadata_builder = mock.MagicMock()
    mock_attn_metadata_builder.build_for_drafting.return_value = mock.MagicMock()
    proposer.attn_metadata_builder = mock_attn_metadata_builder

    batch_spec = BatchSpec(
        seq_lens=[3, 2],
        query_lens=[3, 2],
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=16,
        device=device,
    )

    # Create target hidden states
    target_hidden_states = [
        torch.randn(num_tokens, hidden_size, dtype=proposer.dtype, device=device)
        for _ in range(num_hidden_layers)
    ]

    sampled_token_ids = torch.tensor(
        [42, 60], dtype=torch.int32, device=device
    ).unsqueeze(-1)

    draft_tokens = proposer.propose(
        sampled_token_ids=sampled_token_ids,
        target_hidden_states=target_hidden_states,
        common_attn_metadata=common_attn_metadata,
        slot_mappings=None,
    )

    assert draft_tokens.shape == (batch_size, 1)
    assert torch.equal(draft_tokens, sampled_token_ids)


# ---------------------------------------------------------------------------
# VLM / composite config tests for ExtractHiddenStatesConfig
# ---------------------------------------------------------------------------


class _DummyVLMConfig(PretrainedConfig):
    """Minimal composite config that mimics VLMs like Kimi-K2.5 or LLaVA.

    The text model's parameters (hidden_size, num_attention_heads, …) live
    exclusively under ``text_config``; the top-level config has none of them.
    """

    model_type = "test_vlm"

    def __init__(self, text_config: PretrainedConfig, **kwargs):
        self.text_config = text_config
        super().__init__(architectures=["LlamaForCausalLM"], **kwargs)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        del decoder
        return self.text_config


def test_extract_hidden_states_text_only_config_regression():
    """Text-only models (no nested text_config) must keep working."""
    model_config = ModelConfig(model=model_dir, runner="generate", max_model_len=100)

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        method="extract_hidden_states",
        num_speculative_tokens=1,
        draft_model_config={
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4],
            }
        },
    )

    assert speculative_config.draft_model_config is not None
    # For text-only models, hf_text_config should be the config itself.
    assert speculative_config.draft_model_config.hf_text_config is (
        speculative_config.draft_model_config.hf_config
    )
    assert (
        speculative_config.draft_model_config.hf_text_config.num_attention_heads
        == model_config.hf_text_config.num_attention_heads
    )


def test_extract_hidden_states_config_preserves_vlm_text_config():
    """A real VLM config (LLaVA) with nested text_config must be preserved."""
    text_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
    )
    vlm_config = LlavaConfig(
        vision_config=CLIPVisionConfig(),
        text_config=text_config,
    )

    # Precondition: to_dict() flattens the nested config to a plain dict.
    assert isinstance(vlm_config.to_dict()["text_config"], dict)

    extract_config = ExtractHiddenStatesConfig(
        vlm_config,
        eagle_aux_hidden_state_layer_ids=[1, 2],
    )

    # The fix: text_config is still a PretrainedConfig, not a dict.
    assert isinstance(extract_config.text_config, LlamaConfig)

    extracted = get_hf_text_config(extract_config)
    assert extracted is extract_config.text_config
    assert extracted.num_attention_heads == text_config.num_attention_heads
    assert extracted.hidden_size == text_config.hidden_size

    # Serialization must still round-trip correctly.
    serialized = extract_config.to_dict()
    assert isinstance(serialized["text_config"], dict)
    assert serialized["text_config"]["num_attention_heads"] == (
        text_config.num_attention_heads
    )

    json_str = json.loads(extract_config.to_json_string())
    assert json_str["text_config"]["num_attention_heads"] == (
        text_config.num_attention_heads
    )


def test_extract_hidden_states_speculative_config_vlm():
    """SpeculativeConfig with a VLM target must build without errors."""
    nested_text_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
    )

    target_model_config = ModelConfig(
        model=model_dir,
        runner="generate",
        max_model_len=100,
    )
    # Replace the real text-only config with our composite VLM config.
    target_model_config.hf_config = _DummyVLMConfig(
        text_config=nested_text_config,
    )
    target_model_config.hf_text_config = nested_text_config

    speculative_config = SpeculativeConfig(
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
        method="extract_hidden_states",
        num_speculative_tokens=1,
        draft_model_config={
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [1, 2],
            }
        },
    )

    assert speculative_config.draft_model_config is not None
    assert isinstance(
        speculative_config.draft_model_config.hf_config.text_config,
        LlamaConfig,
    )
    assert speculative_config.draft_model_config.hf_text_config is (
        speculative_config.draft_model_config.hf_config.text_config
    )
    assert (
        speculative_config.draft_model_config.hf_text_config.num_attention_heads
        == nested_text_config.num_attention_heads
    )


def test_extract_hidden_states_config_invalid_text_config():
    """A nested text_config missing required attrs must still be rejected."""
    broken_text_config = PretrainedConfig(hidden_size=128)
    vlm_config = _DummyVLMConfig(text_config=broken_text_config)

    extract_config = ExtractHiddenStatesConfig(
        vlm_config,
        eagle_aux_hidden_state_layer_ids=[1],
    )

    # The object is preserved (not flattened), …
    assert extract_config.text_config is broken_text_config
    # … but validation still rejects the missing attribute.
    with pytest.raises(ValueError, match="num_attention_heads"):
        get_hf_text_config(extract_config)
