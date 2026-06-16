# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.mineru_diffusion import (
    get_num_transfer_tokens,
    sample_with_temperature_topk_topp,
    select_transfer_indices,
)
from vllm.transformers_utils.configs.mineru_diffusion import MinerUDiffusionConfig


def test_mineru_diffusion_config_defaults():
    config = MinerUDiffusionConfig()

    assert config.model_type == "mineru_diffusion"
    assert config.architectures == ["MinerUDiffusionForConditionalGeneration"]
    assert config.text_config.model_type == "sdar"
    assert config.vision_config.model_type == "qwen2_vl"
    assert config.mask_token_id == 151669


def test_num_transfer_tokens_matches_uniform_schedule():
    assert get_num_transfer_tokens(32, 8) == [4, 4, 4, 4, 4, 4, 4, 4]
    assert get_num_transfer_tokens(10, 4) == [3, 3, 2, 2]


def test_select_transfer_indices_prefers_threshold_then_topk():
    confidence = torch.tensor([[0.9, 0.1, 0.8, 0.7], [0.4, 0.3, 0.2, 0.1]])

    selected = select_transfer_indices(
        confidence,
        threshold=0.75,
        transfer_count=2,
    )

    assert selected.tolist() == [
        [True, False, True, False],
        [True, True, False, False],
    ]


def test_greedy_sampling_returns_argmax_confidence():
    logits = torch.tensor([[[0.0, 2.0, 1.0], [4.0, 0.0, 1.0]]])

    token_ids, confidence = sample_with_temperature_topk_topp(
        logits,
        temperature=0.0,
    )

    assert token_ids.tolist() == [[1, 0]]
    assert torch.all(confidence > 0.8)
