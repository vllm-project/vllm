# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.platforms import current_platform
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

model_dir = "meta-llama/Llama-3.1-8B-Instruct"
eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
eagle3_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

NUM_SPECULATIVE_TOKENS_NGRAM = 5
NUM_SPECULATIVE_TOKENS_EAGLE = 3
PROMPT_LOOKUP_MIN = 2
PROMPT_LOOKUP_MAX = 5
DEVICE = current_platform.device_type
MAX_MODEL_LEN = 100


def _create_vllm_config(
    num_speculative_tokens_ngram: int, num_speculative_tokens_eagle: int
):
    model_config = ModelConfig(
        model=model_dir, runner="generate", max_model_len=MAX_MODEL_LEN
    )

    # Choose model directory based on method
    draft_model_dir = eagle_dir

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=draft_model_dir,
        method="ngram-eagle",
        num_speculative_tokens_per_method={
            "ngram": num_speculative_tokens_ngram,
            "eagle": num_speculative_tokens_eagle,
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
        scheduler_config=SchedulerConfig(
            max_model_len=MAX_MODEL_LEN, is_encoder_decoder=False
        ),
    )

    return vllm_config


def test_proposer_config():
    vllm_config = _create_vllm_config(
        NUM_SPECULATIVE_TOKENS_NGRAM, NUM_SPECULATIVE_TOKENS_EAGLE
    )

    # ngram proposer
    ngram_proposer = NgramProposer(vllm_config=vllm_config)
    assert ngram_proposer.k == NUM_SPECULATIVE_TOKENS_NGRAM
    assert ngram_proposer.min_n == PROMPT_LOOKUP_MIN
    assert ngram_proposer.max_n == PROMPT_LOOKUP_MAX

    # eagle proposer
    eagle_proposer = EagleProposer(
        vllm_config=vllm_config, device=current_platform.device_type
    )
    assert eagle_proposer.num_speculative_tokens == NUM_SPECULATIVE_TOKENS_EAGLE


@pytest.mark.parametrize(
    "test_value",
    [
        {
            "sampled_token_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            # ngram draft is empty
            "propose_ngram_draft_token_ids": [[]],
        },
        {
            "sampled_token_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3]],
            # ngram draft is not empty
            "propose_ngram_draft_token_ids": [[4, 5, 6, 7, 8]],
        },
    ],
)
@pytest.mark.parametrize("pp_size", [1, 2])
@mock.patch("vllm.v1.worker.gpu_model_runner.get_pp_group")
@mock.patch(
    "vllm.v1.worker.gpu_model_runner.NgramProposer.propose",
)
@mock.patch(
    "vllm.v1.worker.gpu_model_runner.EagleProposer.propose",
    return_value=torch.tensor([[0, 1, 2]]),
)
@mock.patch(
    "vllm.v1.worker.gpu_model_runner.EagleProposer.prepare_inputs",
    return_value=(None, 0),
)
def test_propose_draft_token_ids(
    mock_eagle_proposer_prepare_input,
    mock_eagle_proposer_propose,
    mock_propose_ngram_draft_token_ids,
    mock_get_pp_group,
    test_value,
    pp_size,
):
    vllm_config = _create_vllm_config(
        NUM_SPECULATIVE_TOKENS_NGRAM, NUM_SPECULATIVE_TOKENS_EAGLE
    )

    runner = GPUModelRunner(vllm_config, DEVICE)

    # Setup mock for pp group to return the appropriate value for world size
    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = pp_size
    mock_get_pp_group.return_value = mock_pp_group

    sampled_token_ids = test_value["sampled_token_ids"]
    propose_ngram_draft_token_ids = test_value["propose_ngram_draft_token_ids"]

    # with min matching ngram = 2, max matching ngram = 3
    # we will find the prefix [1, 2, 3] in the history
    # and speculate [4, 5, 6, 7, 8] for ngram
    expected_ngram_proposals = [[4, 5, 6, 7, 8]]
    expected_eagle_proposals = [[i for i in range(NUM_SPECULATIVE_TOKENS_EAGLE)]]
    mock_propose_ngram_draft_token_ids.return_value = propose_ngram_draft_token_ids

    # doesnt matter what this is for this test: START
    scheduler_output = mock.MagicMock()
    scheduler_output.total_num_scheduled_tokens = 1 + max(
        vllm_config.speculative_config.num_speculative_tokens_per_method["ngram"],
        vllm_config.speculative_config.num_speculative_tokens_per_method["eagle"],
    )
    hidden_states = torch.randn(len(sampled_token_ids[0]), 4096)
    sample_hidden_states = None
    aux_hidden_states = None
    spec_decode_metadata = mock.MagicMock()
    spec_decode_metadata.num_draft_tokens = [
        max(NUM_SPECULATIVE_TOKENS_NGRAM, NUM_SPECULATIVE_TOKENS_EAGLE)
    ]
    common_attn_metadata = None
    sampling_metadata = None

    # set runner attributes that would normally be set during init
    runner.supports_mm_inputs = False

    mock_positions = mock.MagicMock()
    mock_positions_instance = mock_positions.return_value
    mock_positions_instance.gpu = torch.tensor([0])
    runner.positions = mock_positions_instance

    mock_input_ids = mock.MagicMock()
    mock_input_ids_instance = mock_input_ids.return_value
    mock_input_ids_instance.gpu = torch.tensor([0])
    runner.input_ids = mock_input_ids_instance

    mock_req_ids = mock.MagicMock()
    mock_req_ids.return_value = ["0"]
    # doesnt matter what this is for this test: END

    final_draft = runner.propose_draft_token_ids(
        scheduler_output=scheduler_output,
        sampled_token_ids=sampled_token_ids,
        sampling_metadata=sampling_metadata,
        hidden_states=hidden_states,
        sample_hidden_states=sample_hidden_states,
        use_padded_batch_for_eagle=False,
        aux_hidden_states=aux_hidden_states,
        spec_decode_metadata=spec_decode_metadata,
        common_attn_metadata=common_attn_metadata,
    )

    # case 1: ngram draft is empty. Eagle draft is used
    if sampled_token_ids == [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]:
        assert final_draft == expected_eagle_proposals, (
            "ngram-eagle should have selected eagle draft"
        )
    # case 2: ngram draft is not empty. Ngram draft is used
    elif sampled_token_ids == [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3]]:
        assert final_draft == expected_ngram_proposals, (
            "ngram-eagle should have selected ngram draft"
        )
    else:
        raise ValueError("unexpected sampled_token_ids")
