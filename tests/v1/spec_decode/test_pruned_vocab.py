from typing import Optional
from unittest import mock

import torch
import pytest

from tests.utils import get_attn_backend_list_based_on_platform
from tests.v1.spec_decode.test_eagle import _get_propose_args
from vllm.platforms import current_platform
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, SpeculativeConfig,
                         VllmConfig)
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.v1.spec_decode.eagle import EagleProposer

# NousResearch model is identical to meta-llama/Meta-Llama-3-8B-Instruct but doesn't need Meta to grant permission
model_path: str = "NousResearch/Meta-Llama-3.1-8B-Instruct" # "NousResearch/Meta-Llama-3-8B-Instruct"
draft_model_path: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
draft_vocab_pruned_path: str = 'thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt'

def _setup_proposer(
    device: torch.device,
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
        device_config=DeviceConfig(device=device),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig())

    return EagleProposer(vllm_config=vllm_config, device=device)

def _setup_attn_backend(attn_backend, monkeypatch):
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", attn_backend)

    if (attn_backend == "TRITON_ATTN_VLLM_V1"
            and not current_platform.is_rocm()):
        pytest.skip("TRITON_ATTN_VLLM_V1 does not support "
                    "multi-token eagle spec decode on current platform")

    if attn_backend == "FLASH_ATTN_VLLM_V1" and current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

def _setup_mock_get_model(mock_get_model, use_distinct_embed_tokens):
    # Setup draft model mock
    mock_model = mock.MagicMock()
    if use_distinct_embed_tokens:
        # Some models can have a different hidden size than the target model,
        # so we test that their embed_tokens doesn't get overwritten
        mock_model.model.embed_tokens.weight.shape = (131072, 2048)
    else:
        mock_model.model.embed_tokens.weight.shape = (131072, 4096)

    # create lm_head
    # mock_model.lm_head.weight.shape = (131072, 4096)

    mock_get_model.return_value = mock_model
    return mock_get_model


def _setup_mock_pp_group(mock_get_pp_group, pp_size):
    # Setup mock for pp group to return the appropriate value for world size
    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = pp_size
    mock_get_pp_group.return_value = mock_pp_group
    return mock_get_pp_group

def _setup_target_model(device: torch.device):

    # Set up the target model mock with a custom class so that
    # isinstance() checks match the expected type.
    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.model.embed_tokens.weight.shape = (131072, 4096)
    # target_model.model.lm_head.weight.shape = (5, 4096)
    from vllm.model_executor.models import SupportsMultiModal
    assert not isinstance(target_model, SupportsMultiModal)

    # target_model.lm_head needs to support deepcopy
    class DeepCopyableMock(mock.MagicMock):
        def __deepcopy__(self, memo):
            new_mock = DeepCopyableMock()
            new_mock.weight = mock.MagicMock()
            new_mock.weight.device = self.weight.device
            new_mock.weight.shape = self.weight.shape
            return new_mock
    target_model.lm_head = DeepCopyableMock()
    target_model.lm_head.weight.device = device

    return target_model


@pytest.mark.parametrize("attn_backend",
                         get_attn_backend_list_based_on_platform())
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("use_distinct_embed_tokens", [True, False])
@mock.patch('vllm.v1.spec_decode.eagle.get_pp_group')
@mock.patch('vllm.v1.spec_decode.eagle.get_model')
def test_load_model(mock_get_model, mock_get_pp_group,
                    attn_backend, pp_size, use_distinct_embed_tokens, monkeypatch):
    device = torch.device(current_platform.device_type)

    # create proposer where we prune the vocab and are left with 25% of the original vocab
    proposer = _setup_proposer(device, num_speculative_tokens=8)

    # load target model
    _setup_attn_backend(attn_backend, monkeypatch)
    mock_get_pp_group = _setup_mock_pp_group(mock_get_pp_group, pp_size)
    mock_get_model = _setup_mock_get_model(mock_get_model, use_distinct_embed_tokens)
    target_model = _setup_target_model(device)
    proposer.load_model(target_model)

    # Verify common interactions
    mock_get_model.assert_called_once()

    # # Verify that draft model lm head is a pruned from the target model
    # print(proposer.model.lm_head.shape, target_model.lm_head.shape)
    # prune_ratio = proposer.model.lm_head.shape / target_model.lm_head.shape
    # assert proposer.model.lm_head.shape < target_model.lm_head.shape

    # Verify that the embed tokens are set correctly
    # If pp_size is > 1, the embed tokens should be distinct
    if pp_size > 1 or use_distinct_embed_tokens:
        assert proposer.model.model.embed_tokens != \
            target_model.model.embed_tokens
    else:
        # When pp_size is 1 and the draft and target models have
        # embed_tokens of the same shape, they should be shared.
        assert proposer.model.model.embed_tokens == \
            target_model.model.embed_tokens


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
    proposer = _setup_proposer(device, num_speculative_tokens,
                                speculative_token_tree=spec_token_tree)
    # Get the hidden_size from the proposer to ensure consistency.
    hidden_size = proposer.hidden_size

    ret = _get_propose_args(batch_size, vocab_size, hidden_size, total_tokens, seq_len_1, seq_len_2, seq_lens, device, proposer)
    target_token_ids, target_positions,target_hidden_states, next_token_ids, common_attn_metadata, sampling_metadata, base_token_ids = ret

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
