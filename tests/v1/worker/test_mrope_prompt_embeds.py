# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that M-RoPE position initialization handles prompt_embeds-only inputs.

Regression test for GHSA-33cg-gxv8-3p8g: sending /v1/completions with
prompt_embeds and no prompt_token_ids on M-RoPE models crashed the
EngineCore via an assertion failure.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class FakeMRoPEModel(SupportsMRoPE):
    """Minimal model that passes supports_mrope() check."""

    def get_mrope_input_positions(self, input_tokens, mm_features):
        seq_len = len(input_tokens)
        positions = torch.arange(seq_len).unsqueeze(0).expand(3, -1)
        return positions.clone(), 0


def _make_runner_and_req(prompt_token_ids, prompt_embeds):
    """Create a minimal GPUModelRunner instance and request state."""
    model = FakeMRoPEModel()
    instance = object.__new__(GPUModelRunner)
    instance.get_model = lambda: model

    req_state = Mock(spec=CachedRequestState)
    req_state.prompt_token_ids = prompt_token_ids
    req_state.prompt_embeds = prompt_embeds
    req_state.mm_features = []
    req_state.mrope_positions = None
    req_state.mrope_position_delta = None
    return instance, req_state


class TestMRopePromptEmbeds:
    """Verify _init_mrope_positions handles prompt_embeds-only inputs."""

    def test_prompt_embeds_only_does_not_crash(self):
        """Prompt-embeds-only request must not raise AssertionError."""
        instance, req_state = _make_runner_and_req(
            prompt_token_ids=None,
            prompt_embeds=torch.randn(15, 896),
        )

        instance._init_mrope_positions(req_state)

        assert req_state.mrope_positions is not None
        assert req_state.mrope_positions.shape == (3, 15)

    def test_prompt_token_ids_still_works(self):
        """Normal path with prompt_token_ids continues working."""
        instance, req_state = _make_runner_and_req(
            prompt_token_ids=[1, 2, 3, 4, 5],
            prompt_embeds=None,
        )

        instance._init_mrope_positions(req_state)

        assert req_state.mrope_positions is not None
        assert req_state.mrope_positions.shape == (3, 5)

    def test_neither_token_ids_nor_embeds_raises(self):
        """When both are None, a ValueError should be raised."""
        instance, req_state = _make_runner_and_req(
            prompt_token_ids=None,
            prompt_embeds=None,
        )

        with pytest.raises(ValueError, match="prompt_token_ids or prompt_embeds"):
            instance._init_mrope_positions(req_state)
