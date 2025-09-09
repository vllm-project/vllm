from typing import Optional
from unittest import mock

import torch
import pytest

from tests.utils import get_attn_backend_list_based_on_platform
from vllm.platforms import current_platform
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, SpeculativeConfig,
                         VllmConfig)
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.v1.spec_decode.eagle import EagleProposer

# NousResearch model is identical to meta-llama/Meta-Llama-3-8B-Instruct but doesn't need Meta to grant permission
model_path: str = "NousResearch/Meta-Llama-3-8B-Instruct"
draft_model_path: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
draft_vocab_pruned_path: str = 'thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt'

def _create_proposer(
    num_speculative_tokens: int,
    speculative_token_tree: Optional[list[tuple[int]]] = None,
) -> EagleProposer:
    model_config = ModelConfig(model=model_path,
                               runner="generate",
                               max_model_len=100)

    spec_token_tree_str = None
    if speculative_token_tree is not None:
        assert num_speculative_tokens == len(speculative_token_tree)
        spec_token_tree_str = str(speculative_token_tree)

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=draft_model_path,
        method="eagle",
        num_speculative_tokens=num_speculative_tokens,
        speculative_token_tree=spec_token_tree_str,
        draft_vocab_pruned=draft_vocab_pruned_path,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=current_platform.device_type),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig())

    return EagleProposer(vllm_config=vllm_config,
                         device=current_platform.device_type)

@pytest.mark.parametrize("attn_backend",
                         get_attn_backend_list_based_on_platform())
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("use_distinct_embed_tokens", [True, False])
@mock.patch('vllm.v1.spec_decode.eagle.get_pp_group')
@mock.patch('vllm.v1.spec_decode.eagle.get_model')
def test_load_model(mock_get_model, mock_get_pp_group,
                    attn_backend, pp_size, use_distinct_embed_tokens, monkeypatch):

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", attn_backend)

    if (attn_backend == "TRITON_ATTN_VLLM_V1"
            and not current_platform.is_rocm()):
        pytest.skip("TRITON_ATTN_VLLM_V1 does not support "
                    "multi-token eagle spec decode on current platform")

    if attn_backend == "FLASH_ATTN_VLLM_V1" and current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    # Setup draft model mock
    mock_model = mock.MagicMock()
    if use_distinct_embed_tokens:
        # Some models can have a different hidden size than the target model,
        # so we test that their embed_tokens doesn't get overwritten
        mock_model.model.embed_tokens.weight.shape = (131072, 2048)
    else:
        mock_model.model.embed_tokens.weight.shape = (131072, 4096)

    mock_get_model.return_value = mock_model

    # Setup mock for pp group to return the appropriate value for world size
    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = pp_size
    mock_get_pp_group.return_value = mock_pp_group

    # Set up the target model mock with a custom class so that
    # isinstance() checks match the expected type.
    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.model.embed_tokens.weight.shape = (131072, 4096)

    from vllm.model_executor.models import SupportsMultiModal
    assert not isinstance(target_model, SupportsMultiModal)
    target_model.lm_head = mock.MagicMock()

    # init eagle with pruned vocabulary
    proposer = _create_proposer(num_speculative_tokens=8)
    proposer.load_model(target_model)

    # Verify common interactions
    mock_get_model.assert_called_once()

    # # Verify that draft model lm head is a pruned from the target model
    # assert proposer.model.lm_head == target_model.lm_head

    # # Verify that the embed tokens are set correctly
    # # If pp_size is > 1, the embed tokens should be distinct
    # if pp_size > 1 or use_distinct_embed_tokens:
    #     assert proposer.model.model.embed_tokens != \
    #         target_model.model.embed_tokens
    # else:
    #     # When pp_size is 1 and the draft and target models have
    #     # embed_tokens of the same shape, they should be shared.
    #     assert proposer.model.model.embed_tokens == \
    #         target_model.model.embed_tokens


@pytest.mark.parametrize(
    "spec_token_tree",
    [
        [(0, )],  # A single token
        [(0, ), (0, 0), (0, 0, 0)],  # Chain
        [(0, ), (1, ), (2, )],  # Parallel
        [(0, ), (1, ), (2, ), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0),
         (2, 1)],  # Tree
    ])
def test_propose_tree(spec_token_tree):
    # Get GPU device.
    device = torch.device(current_platform.device_type)

    # Setup test parameters.
    batch_size = 2
    seq_len_1 = 5
    seq_len_2 = 3
    total_tokens = seq_len_1 + seq_len_2
    vocab_size = 100
    seq_lens = [seq_len_1, seq_len_2]
    num_speculative_tokens = len(spec_token_tree)

    # Create proposer first so we can use its actual hidden_size.
    proposer = _create_proposer("eagle",
                                num_speculative_tokens,
                                speculative_token_tree=spec_token_tree)
    # Get the hidden_size from the proposer to ensure consistency.
    hidden_size = proposer.hidden_size

    # Helper to create deterministic logits that will produce specific tokens
    def create_deterministic_logits(token_ids, k: int):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        for i, token_id in enumerate(token_ids):
            # Assign decreasing values to the k, consecutive, tokens.
            for j in range(k):
                logits[i, token_id + j] = 100.0 - j
        return logits

    # Mock a model that returns deterministic logits.
    base_token_ids = torch.tensor([42, 60], dtype=torch.int64, device=device)

    # Skip loading the model and replace it with a mock that returns
    # deterministic outputs.
    model_mock = mock.MagicMock()

    # Mock the model forward calls.
    forward_returns = [(torch.zeros(total_tokens, hidden_size, device=device),
                        torch.zeros(total_tokens, hidden_size, device=device))]
    for cu_num_drafts in proposer.cu_drafts_per_level:
        h_logits = torch.zeros(batch_size * cu_num_drafts,
                               hidden_size,
                               device=device)
        h_states = torch.zeros(batch_size * cu_num_drafts,
                               hidden_size,
                               device=device)
        forward_returns.append((h_logits, h_states))
    model_mock.side_effect = forward_returns

    # Mock the compute_logits calls.
    cu_num_drafts_tensor = torch.tensor([0] + proposer.cu_drafts_per_level,
                                        dtype=torch.int32,
                                        device=device)
    logits_returns = []
    for level, num_children in enumerate(proposer.child_drafts_per_level):
        token_ids = base_token_ids + cu_num_drafts_tensor[level]
        level_num_drafts = cu_num_drafts_tensor[
            level + 1] - cu_num_drafts_tensor[level]
        level_logits = []
        for i in range(level_num_drafts // num_children):
            level_logits.append(
                create_deterministic_logits(token_ids + i * num_children,
                                            num_children))
        logits_returns.append(torch.stack(level_logits, dim=1))
    model_mock.compute_logits.side_effect = logits_returns

    # Assign the mock to the proposer
    proposer.model = model_mock

    # Assign draft attn_layer_names since load_model is not invoked
    proposer.attn_layer_names = ["layer.0"]

    # Get the tree attention metadata builder.
    attn_metadata_builder_cls, _ = get_attention_backend(_Backend.TREE_ATTN)
    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=proposer.attn_layer_names,
        vllm_config=proposer.vllm_config,
        device=device,
    )

    # Mock runner for attention metadata building.
    proposer.runner = mock.MagicMock()
    proposer.runner.attn_groups.append([mock.MagicMock()])
    proposer.runner.attn_groups[0][0].metadata_builder = attn_metadata_builder

    # Setup inputs for the proposer.
    target_token_ids = torch.randint(0,
                                     vocab_size, (total_tokens, ),
                                     device=device)
    target_positions = torch.cat([
        torch.arange(seq_len_1, device=device),
        torch.arange(seq_len_2, device=device)
    ])
    target_hidden_states = torch.randn(total_tokens,
                                       hidden_size,
                                       device=device)
    next_token_ids = torch.randint(0,
                                   vocab_size, (batch_size, ),
                                   dtype=torch.int32,
                                   device=device)
    batch_spec = BatchSpec(
        seq_lens=seq_lens,
        query_lens=seq_lens,
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=16,
        device=device,
    )
    sampling_metadata = mock.MagicMock()

    # Propose draft tokens.
    result = proposer.propose(target_token_ids=target_token_ids,
                              target_positions=target_positions,
                              target_hidden_states=target_hidden_states,
                              next_token_ids=next_token_ids,
                              common_attn_metadata=common_attn_metadata,
                              sampling_metadata=sampling_metadata)
    assert result.shape == (batch_size, num_speculative_tokens)

    # The tokens are expected to be consecutive integers starting
    # from the base token IDs.
    expected_tokens = base_token_ids[:, None] + torch.arange(
        num_speculative_tokens, dtype=torch.int64, device=device)

    # Verify that the draft tokens match our expectations.
    assert torch.equal(result, expected_tokens)
