# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for v1 attention backends without GPUModelRunner dependency."""

import pytest
import torch

from tests.v1.attention.test_attention_backends import BATCH_SPECS
from tests.v1.attention.utils import (create_common_attn_metadata,
                                      create_vllm_config)


@pytest.mark.parametrize("batch_spec_name", [
    "small_decode", "small_prefill", "mixed_small", "medium_decode",
    "medium_prefill", "mixed_medium"
])
@pytest.mark.parametrize("model", ["meta-llama/Meta-Llama-3-8B"])
def test_attention_splitting_correctness(batch_spec_name: str, model: str):
    """
    """
    batch_spec = BATCH_SPECS[batch_spec_name]
    vllm_config = create_vllm_config(model_name=model)
    device = torch.device("cuda:0")

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device)

    # assert backend_output.shape == sdpa_output.shape, (
    #     f"[{backend_name}] shape {backend_output.shape} != "
    #     f"SDPA shape {sdpa_output.shape}")
