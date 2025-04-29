# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
import torch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, SpeculativeConfig,
                         VllmConfig)
from vllm.v1.spec_decode.eagle import EagleProposer

model_dir = "meta-llama/Llama-3.1-8B-Instruct"
eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
eagle3_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


def _create_proposer(method: str, k: int) -> EagleProposer:
    model_config = ModelConfig(model=model_dir,
                               task="generate",
                               max_model_len=100,
                               tokenizer=model_dir,
                               tokenizer_mode="auto",
                               dtype="auto",
                               seed=None,
                               trust_remote_code=False)

    # Choose model directory based on method
    draft_model_dir = eagle_dir if method == "eagle" else eagle3_dir

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=draft_model_dir,
        method=method,
        num_speculative_tokens=k,
    )

    vllm_config = VllmConfig(model_config=model_config,
                             cache_config=CacheConfig(),
                             speculative_config=speculative_config,
                             device_config=DeviceConfig(device="cuda"),
                             parallel_config=ParallelConfig(),
                             load_config=LoadConfig(),
                             scheduler_config=SchedulerConfig())

    return EagleProposer(vllm_config=vllm_config, device='cuda')


def test_prepare_inputs():
    """
    cu_target_query_lens: [0, a, a + b, a + b + c]
    num_rejected_tokens: [n1, n2, n3]
    num_tokens_per_req: [a - n1, b - n2, c - n3]
    cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
    token_indices: [0, 1, ..., a - n1 - 1,
                    a, a + 1, ..., a + b - n2 - 1,
                    a + b, a + b + 1, ..., a + b + c - n3 - 1]
    """
    device = torch.device('cuda')

    # a = 4, b = 7, c = 5
    # n1 = 1, n2 = 3, n3 = 2

    # Cumulative lengths: [0, 4, 11, 16]
    cu_target_query_lens = torch.tensor([0, 4, 11, 16],
                                        dtype=torch.int32,
                                        device=device)

    # Rejected tokens per request: [1, 3, 2]
    num_rejected_tokens = torch.tensor([1, 3, 2],
                                       dtype=torch.int32,
                                       device=device)

    # Expected calculations:
    # query_len_per_req = [4, 7, 5]
    # num_tokens_per_req = [3, 4, 3]  (after subtracting rejected tokens)
    # Expected cumulative counts: [0, 3, 7, 10]
    expected_cu_num_tokens = torch.tensor([0, 3, 7, 10],
                                          dtype=torch.int32,
                                          device=device)

    # Expected token indices (mapped from original positions):
    # First request: indices 0, 1, 2      (keeping first 3 from positions 0-3)
    # Second request: indices 4, 5, 6, 7  (keeping first 4 from positions 4-10)
    # Third request: indices 11, 12, 13   (keeping first 3 from positions 11-15)
    expected_token_indices = torch.tensor(
        [
            0,
            1,
            2,  # First request: 3 tokens (4-1)
            4,
            5,
            6,
            7,  # Second request: 4 tokens (7-3)
            11,
            12,
            13
        ],  # Third request: 3 tokens (5-2)
        dtype=torch.int32,
        device=device)

    cu_num_tokens, token_indices = EagleProposer.prepare_inputs(
        cu_target_query_lens, num_rejected_tokens)

    assert torch.equal(cu_num_tokens, expected_cu_num_tokens)
    assert token_indices.shape[0] == expected_cu_num_tokens[-1].item()
    assert torch.equal(token_indices, expected_token_indices)


@pytest.mark.parametrize(
    "method,eagle_model_class,proposer_helper,draft_model_dir,target_attribute_path",
    [
        ("eagle", 'vllm.v1.spec_decode.eagle.EagleLlamaForCausalLM',
         lambda k: _create_proposer("eagle", k), eagle_dir, ('lm_head', )),
        ("eagle3", 'vllm.v1.spec_decode.eagle.Eagle3LlamaForCausalLM',
         lambda k: _create_proposer("eagle3", k), eagle3_dir,
         ('model', 'embed_tokens')),
    ])
@mock.patch('vllm.v1.spec_decode.eagle.set_default_torch_dtype')
@mock.patch('vllm.v1.spec_decode.eagle.set_current_vllm_config')
@mock.patch('vllm.v1.spec_decode.eagle.get_model_loader')
def test_load_model(mock_get_loader, mock_set_config, mock_set_dtype, method,
                    eagle_model_class, proposer_helper, draft_model_dir,
                    target_attribute_path):
    """Test loading an Eagle/Eagle3 model"""

    # Patch the appropriate Eagle model class
    with mock.patch(eagle_model_class) as mock_eagle_class:
        # Setup model loader mock
        mock_loader = mock.MagicMock()
        mock_get_loader.return_value = mock_loader

        # Setup model mock
        mock_model = mock.MagicMock()
        mock_eagle_class.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Configure mock to test the attribute sharing path
        if method == "eagle":
            # For eagle, test the lm_head path
            mock_model.load_weights.return_value = {
                "model.embed_tokens.weight": torch.zeros(1)
            }
        else:
            # For eagle3, test the embed_tokens path
            mock_model.load_weights.return_value = {}

        # Setup target model with the appropriate attributes
        target_model = mock.MagicMock()

        # Create the necessary attributes on the target model
        current_obj = target_model
        for i, attr in enumerate(target_attribute_path):
            if i == len(target_attribute_path) - 1:
                # Set the last attribute in the path to a MagicMock
                setattr(current_obj, attr, mock.MagicMock())
            else:
                # Create intermediate objects if needed
                setattr(current_obj, attr, mock.MagicMock())
                current_obj = getattr(current_obj, attr)

        # Create proposer using the helper function
        proposer = proposer_helper(k=8)

        # Call the method under test
        proposer.load_model(target_model)

        # Verify common interactions
        mock_get_loader.assert_called_once()
        mock_eagle_class.assert_called_once()
        mock_model.to.assert_called_once()
        mock_model.load_weights.assert_called_once()

        # Verify the loader was called with the right config
        mock_get_loader.assert_called_once_with(
            proposer.vllm_config.load_config)

        # Verify model configuration
        assert proposer.vllm_config.model_config.model == model_dir
        assert proposer.vllm_config.speculative_config.method == method
        assert proposer.vllm_config.speculative_config.num_speculative_tokens \
            == 8

        # Check correct draft model path
        assert proposer.vllm_config.speculative_config.model == draft_model_dir

        # Verify the specific attribute sharing based on the method
        if method == "eagle":
            assert proposer.model.lm_head == target_model.lm_head
        else:
            assert proposer.model.model.embed_tokens == \
                target_model.model.embed_tokens
