# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional
from unittest import mock

import pytest
import torch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, SpeculativeConfig,
                         VllmConfig)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.platforms import current_platform
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

model_dir = "meta-llama/Llama-3.1-8B-Instruct"
eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
eagle3_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

NUM_SPECULATIVE_TOKENS_NGRAM = 5
NUM_SPECULATIVE_TOKENS_EAGLE = 3
PROMPT_LOOKUP_MIN = 2
PROMPT_LOOKUP_MAX = 5
DEVICE = current_platform.device_type

def _create_proposer(num_speculative_tokens_ngram: int, num_speculative_tokens_eagle: int):
    model_config = ModelConfig(model=model_dir,
                               runner="generate",
                               max_model_len=100)

    # Choose model directory based on method
    draft_model_dir = eagle_dir

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=draft_model_dir,
        method="ngram-eagle",
        num_speculative_tokens_per_method={
            "ngram": num_speculative_tokens_ngram,
            "eagle": num_speculative_tokens_eagle
        },
        prompt_lookup_max=PROMPT_LOOKUP_MAX,
        prompt_lookup_min=PROMPT_LOOKUP_MIN,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=current_platform.device_type),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig())

    # ngram proposer
    ngram_proposer = NgramProposer(vllm_config=vllm_config,
                                    device=current_platform.device_type)
    assert ngram_proposer.num_speculative_tokens == num_speculative_tokens_ngram
    assert ngram_proposer.min_n == PROMPT_LOOKUP_MIN
    assert ngram_proposer.max_n == PROMPT_LOOKUP_MAX

    # eagle proposer
    eagle_proposer = EagleProposer(vllm_config=vllm_config,
                         device=current_platform.device_type)
    assert eagle_proposer.num_speculative_tokens == num_speculative_tokens_eagle

    return ngram_proposer, eagle_proposer, speculative_config, vllm_config


# @pytest.mark.parametrize("num_speculative_tokens_eagle", [3])
# @pytest.mark.parametrize("num_speculative_tokens_ngram", [5])
# @pytest.mark.parametrize("eagle_proposals", [[1, 2, 3]])
@pytest.mark.parametrize("sampled_token_ids", 
                         [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # ngram draft is empty
                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3]] # ngram draft is not empty
                          )
@mock.patch('vllm.v1.worker.gpu_model_runner.drafter_eagle.prepare_inputs', return_value=(None, 0))
@mock.patch('vllm.v1.worker.gpu_model_runner.drafter_eagle.propose')
@mock.patch('vllm.v1.worker.gpu_model_runner.drafter_eagle')
@mock.patch('vllm.v1.worker.gpu_model_runner.drafter_ngram')
@mock.patch('vllm.v1.worker.gpu_model_runner.speculative_config')
# doesnt matter what this is for this test
@mock.patch('vllm.v1.worker.gpu_model_runner.input_batch.req_ids', return_value=["0"])
# doesnt matter what this is for this test
@mock.patch('vllm.v1.worker.gpu_model_runner.input_ids.gpu', return_value=torch.tensor([0]))
# doesnt matter what this is for this test
@mock.patch('vllm.v1.worker.gpu_model_runner.positions.gpu', return_value=torch.tensor([0]))
@mock.patch('vllm.v1.worker.gpu_model_runner.supports_mm_inputs', return_value=False)
def test_propose_draft_token_ids(mock_drafter_ngram, 
                                 mock_drafter_eagle, 
                                 mock_speculative_config,
                                 mock_drafter_eagle_propose,
                                 sampled_token_ids,
                                #  num_speculative_tokens_eagle,
                                #  num_speculative_tokens_ngram,
                                #  eagle_proposals
                                 ):

    mock_drafter_ngram, 
    mock_drafter_eagle, 
    mock_speculative_config, 
    vllm_config = _create_proposer(NUM_SPECULATIVE_TOKENS_NGRAM, 
                                   NUM_SPECULATIVE_TOKENS_EAGLE)

    # with min matching ngram = 2, max matching ngram = 3
    # we will find the prefix [1, 2, 3] in the history
    # and speculate [4, 5, 6, 7, 8] for ngram
    expected_ngram_proposals = [4, 5, 6, 7, 8]

    # ngram will always speculate 5 tokens
    # we test eagle proposes lower than or greater than ngram
    expected_eagle_proposals = [i for i in range(NUM_SPECULATIVE_TOKENS_EAGLE)]
    mock_drafter_eagle_propose.return_value = torch.tensor([expected_eagle_proposals])
    
    # doesnt matter what this is for this test: START
    scheduler_output = mock.MagicMock()
    scheduler_output.total_num_scheduled_tokens = 1 + max(
        mock_speculative_config.num_speculative_tokens_per_method["ngram"],
        mock_speculative_config.num_speculative_tokens_per_method["eagle"]
    )
    hidden_states = torch.randn(len(sampled_token_ids[0]), 4096)
    sample_hidden_states = None
    aux_hidden_states = None
    spec_decode_metadata = mock.MagicMock()
    spec_decode_metadata.num_draft_tokens = max(
        NUM_SPECULATIVE_TOKENS_NGRAM, NUM_SPECULATIVE_TOKENS_EAGLE)
    common_attn_metadata = None
    sampling_metadata = None
    # doesnt matter what this is for this test: END

    runner = GPUModelRunner(vllm_config, DEVICE)
    final_draft = runner.propose_draft_token_ids(
        scheduler_output=scheduler_output,
        sampled_token_ids=sampled_token_ids,
        sampling_metadata=sampling_metadata,
        hidden_states=hidden_states,
        sample_hidden_states=sample_hidden_states,
        aux_hidden_states=aux_hidden_states,
        spec_decode_metadata=spec_decode_metadata,
        common_attn_metadata=common_attn_metadata,
    )
    
    if sampled_token_ids == [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]:
        # ngram draft is empty. Eagle draft is used
        assert final_draft == expected_eagle_proposals, "ngram-eagle should have selected eagle draft"
    elif sampled_token_ids == [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3]]:
        # ngram draft is not empty. Ngram draft is used
        assert final_draft == expected_ngram_proposals, "ngram-eagle should have selected ngram draft"
    else:
        raise ValueError("unexpected sampled_token_ids")