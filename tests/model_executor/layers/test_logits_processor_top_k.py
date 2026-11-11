# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)


@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("soft_cap", [None, 3.0])
def test_get_top_k_tokens_values_match_full_logits(scale, soft_cap):
    """Top-k values must equal full logits at the same ids, for every
    combination of scale and soft cap (issue #65401)."""
    with set_current_vllm_config(VllmConfig()):
        head = VocabParallelEmbedding(
            3, 1, params_dtype=torch.float32, padding_size=1, disable_tp=True
        )
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[4.0], [1.0], [-2.0]]))

        processor = LogitsProcessor(3, scale=scale, soft_cap=soft_cap)
        hidden_states = torch.ones(1, 1, dtype=torch.float32)

        full_logits = processor(head, hidden_states)
        ids, values = processor.get_top_k_tokens(head, hidden_states, k=2)
        expected = full_logits.gather(-1, ids)

        torch.testing.assert_close(values, expected)
