from typing import Optional
from unittest import mock

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

    # Verify that draft model lm head is a pruned from the target model
    assert proposer.model.lm_head == target_model.lm_head

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
