# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    try_get_attention_backend,
)
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.load import LoadConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.llm_base_proposer import compute_probs_and_sample_next_token

mimo_7b_dir = "XiaomiMiMo/MiMo-7B-Base"
DEVICE_TYPE = current_platform.device_type


def _create_sampling_metadata(
    all_greedy: bool,
    batch_size: int,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
) -> SamplingMetadata:
    temperature = None
    if not all_greedy:
        temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE)
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=top_p,
        top_k=top_k,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([], device=DEVICE_TYPE),
        presence_penalties=torch.tensor([], device=DEVICE_TYPE),
        repetition_penalties=torch.tensor([], device=DEVICE_TYPE),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=[],
    )


def _create_mtp_proposer(
    num_speculative_tokens: int,
    parallel_drafting: bool = False,
) -> EagleProposer:
    """Create an MTP proposer with unified model configuration."""
    model_config = ModelConfig(
        model=mimo_7b_dir, runner="generate", max_model_len=100, trust_remote_code=True
    )

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=mimo_7b_dir,
        method="mtp",
        num_speculative_tokens=num_speculative_tokens,
        parallel_drafting=parallel_drafting,
    )
    if parallel_drafting:
        speculative_config.draft_model_config.hf_config.ptd_token_id = 0

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=DEVICE_TYPE),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
    )

    return EagleProposer(vllm_config=vllm_config, device=DEVICE_TYPE)


@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_pp_group")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_layers_from_vllm_config")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_model")
def test_mtp_load_model_unified(mock_get_model, mock_get_layers, mock_get_pp_group):
    """Test MTP-specific model loading with unified model approach."""

    # Setup mocks
    mock_model = mock.MagicMock()
    mock_model.model.embed_tokens.weight.shape = (131072, 4096)
    mock_get_model.return_value = mock_model
    # MTP does not have its own embed_tokens or lm_head
    # so it should share them with the target model
    mock_model.has_own_embed_tokens = False
    mock_model.has_own_lm_head = False

    target_attn_layers = {"target_attn_1": mock.MagicMock()}
    all_attn_layers = {**target_attn_layers, "draft_attn_1": mock.MagicMock()}
    target_indexer_layers: dict = {}
    all_indexer_layers: dict = {}

    mock_get_layers.side_effect = [
        target_attn_layers,
        target_indexer_layers,
        all_attn_layers,
        all_indexer_layers,
    ]

    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = 1
    mock_get_pp_group.return_value = mock_pp_group

    # Create target model
    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.model.embed_tokens.weight.shape = (131072, 4096)
    target_model.lm_head = mock.MagicMock()

    # Create MTP proposer
    proposer = _create_mtp_proposer(num_speculative_tokens=4)
    proposer.load_model(target_model)

    # Verify MTP-specific behavior:
    # Model is loaded
    mock_get_model.assert_called_once()
    # MTP shares lm_head with target model
    assert proposer.model.lm_head == target_model.lm_head
    # MTP shares embed_tokens with target model
    assert proposer.model.model.embed_tokens == target_model.model.embed_tokens


@pytest.mark.parametrize("num_speculative_tokens", [1])
def test_mtp_propose(num_speculative_tokens, monkeypatch):
    """Test that MTP's forward method returns hidden states directly"""

    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 100

    proposer = _create_mtp_proposer(num_speculative_tokens)
    hidden_size = proposer.hidden_size

    # Mock the MTP model to verify it returns hidden states directly
    model_mock = mock.MagicMock()

    # MTP returns hidden states directly
    if num_speculative_tokens == 1:
        model_mock.return_value = torch.zeros(total_tokens, hidden_size, device=device)
    else:
        # Multiple forward passes for multi-token speculation
        forward_returns = []
        for i in range(num_speculative_tokens):
            if i == 0:
                h_states = torch.zeros(total_tokens, hidden_size, device=device)
            else:
                h_states = torch.zeros(batch_size, hidden_size, device=device)
            forward_returns.append(h_states)
        model_mock.side_effect = forward_returns

    # Mock compute_logits
    def create_deterministic_logits(batch_size, vocab_size, token_offset):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        logits[:, token_offset] = 100.0
        return logits

    if num_speculative_tokens == 1:
        model_mock.compute_logits.return_value = create_deterministic_logits(
            batch_size, vocab_size, 42
        )
    else:
        logits_returns = [
            create_deterministic_logits(batch_size, vocab_size, 42 + i)
            for i in range(num_speculative_tokens)
        ]
        model_mock.compute_logits.side_effect = logits_returns

    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    # Prepare inputs
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat(
        [
            torch.arange(seq_lens[0], device=device),
            torch.arange(seq_lens[1], device=device),
        ]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    # Setup attention metadata
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )

    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )

    proposer.runner = mock.MagicMock()
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    # Run propose
    result = proposer.propose(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    # Verify the model was called correctly
    assert model_mock.called
    # Verify output shape
    assert result.shape == (batch_size, num_speculative_tokens)


def test_mtp_propose_random_sampling_records_draft_probs():
    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    seq_lens = [3, 2]
    total_tokens = sum(seq_lens)
    vocab_size = 4

    proposer = _create_mtp_proposer(num_speculative_tokens=1)
    assert proposer._enable_probabilistic_draft_probs
    hidden_size = proposer.hidden_size

    model_mock = mock.MagicMock()
    model_mock.return_value = torch.zeros(total_tokens, hidden_size, device=device)
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], device=device)
    model_mock.compute_logits.return_value = logits.clone()
    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )
    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    result = proposer.propose(
        target_token_ids=torch.randint(0, vocab_size, (total_tokens,), device=device),
        target_positions=torch.arange(total_tokens, device=device),
        target_hidden_states=torch.randn(total_tokens, hidden_size, device=device),
        next_token_ids=torch.randint(
            0, vocab_size, (batch_size,), dtype=torch.int32, device=device
        ),
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=_create_sampling_metadata(
            all_greedy=False, batch_size=batch_size
        ),
    )

    assert result.shape == (batch_size, 1)
    assert proposer._last_draft_probs is not None
    assert proposer._last_draft_probs.shape == (batch_size, 1, vocab_size)
    expected_probs = torch.softmax(logits, dim=-1).view(batch_size, 1, vocab_size)
    assert torch.allclose(proposer._last_draft_probs, expected_probs)
    # ``take_last_draft_probs`` is the upstream-side accessor that
    # ``GPUModelRunner`` uses to plumb probs into the rejection sampler.
    assert torch.equal(
        proposer.take_last_draft_probs(), proposer._last_draft_probs
    )


def test_mtp_sequential_drafting_passes_spec_step_indices():
    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    seq_lens = [3, 2]
    total_tokens = sum(seq_lens)
    vocab_size = 4
    num_spec_tokens = 2

    proposer = _create_mtp_proposer(num_speculative_tokens=num_spec_tokens)
    proposer.block_size = 16
    hidden_size = proposer.hidden_size

    model_mock = mock.MagicMock()
    model_mock.side_effect = [
        torch.zeros(total_tokens, hidden_size, device=device),
        torch.zeros(batch_size, hidden_size, device=device),
    ]

    def logits_for_token(token_id: int):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        logits[:, token_id] = 100.0
        return logits

    model_mock.compute_logits.side_effect = [
        logits_for_token(1),
        logits_for_token(2),
    ]
    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )
    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    result = proposer.propose(
        target_token_ids=torch.randint(0, vocab_size, (total_tokens,), device=device),
        target_positions=torch.arange(total_tokens, device=device),
        target_hidden_states=torch.randn(total_tokens, hidden_size, device=device),
        next_token_ids=torch.randint(
            0, vocab_size, (batch_size,), dtype=torch.int32, device=device
        ),
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=_create_sampling_metadata(
            all_greedy=True, batch_size=batch_size
        ),
    )

    assert torch.equal(
        result,
        torch.tensor([[1, 2], [1, 2]], device=device),
    )
    assert [
        call.kwargs.get("spec_step_idx", 0)
        for call in model_mock.compute_logits.call_args_list
    ] == [0, 1]
    assert [
        call.kwargs.get("spec_step_idx", 0)
        for call in model_mock.call_args_list
    ] == [0, 1]


def test_mtp_draft_sampling_applies_top_k_to_draft_probs():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device=DEVICE_TYPE)
    top_k = torch.tensor([2], dtype=torch.int32, device=DEVICE_TYPE)

    _token_ids, draft_probs = compute_probs_and_sample_next_token(
        logits,
        _create_sampling_metadata(all_greedy=False, batch_size=1, top_k=top_k),
    )

    expected_logits = torch.tensor(
        [[-float("inf"), -float("inf"), 2.0, 3.0]], device=DEVICE_TYPE
    )
    expected_probs = torch.softmax(expected_logits, dim=-1, dtype=torch.float32)
    assert torch.allclose(draft_probs, expected_probs)


def test_mtp_parallel_drafting_random_sampling_records_draft_probs():
    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    num_spec_tokens = 2
    seq_lens = [2, 2]
    total_tokens = sum(seq_lens)
    vocab_size = 4

    proposer = _create_mtp_proposer(
        num_speculative_tokens=num_spec_tokens,
        parallel_drafting=True,
    )
    assert proposer._enable_probabilistic_draft_probs
    proposer.block_size = 16
    hidden_size = proposer.hidden_size

    model_mock = mock.MagicMock()
    model_mock.return_value = torch.zeros(
        total_tokens + batch_size,
        hidden_size,
        dtype=proposer.dtype,
        device=device,
    )
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
            [0.0, 0.5, 1.0, 1.5],
            [1.5, 1.0, 0.5, 0.0],
        ],
        device=device,
    )
    model_mock.compute_logits.return_value = logits.clone()
    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )
    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    result = proposer.propose(
        target_token_ids=torch.randint(0, vocab_size, (total_tokens,), device=device),
        target_positions=torch.arange(total_tokens, device=device),
        target_hidden_states=torch.randn(
            total_tokens,
            hidden_size,
            dtype=proposer.dtype,
            device=device,
        ),
        next_token_ids=torch.randint(
            0, vocab_size, (batch_size,), dtype=torch.int32, device=device
        ),
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=_create_sampling_metadata(
            all_greedy=False, batch_size=batch_size
        ),
    )

    assert result.shape == (batch_size, num_spec_tokens)
    assert proposer._last_draft_probs is not None
    assert proposer._last_draft_probs.shape == (
        batch_size, num_spec_tokens, vocab_size
    )
    assert torch.allclose(
        proposer._last_draft_probs,
        torch.softmax(logits, dim=-1).view(batch_size, num_spec_tokens, vocab_size),
    )
    assert torch.equal(
        proposer.take_last_draft_probs(), proposer._last_draft_probs
    )
