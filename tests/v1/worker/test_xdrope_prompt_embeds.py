# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that XD-RoPE position initialization handles prompt_embeds-only inputs.

Regression test for GHSA-5q78-j82c-2vr2: sending /v1/completions with
prompt_embeds and no prompt_token_ids on XD-RoPE models crashed the
EngineCore via an assertion failure (incomplete fix of CVE-2026-55514).
"""

from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.models.interfaces import SupportsXDRoPE
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

XDROPE_DIM = 4


class FakeXDRoPEModel(SupportsXDRoPE):
    """Minimal model that passes supports_xdrope() check."""

    def get_xdrope_input_positions(self, input_tokens, mm_features):
        seq_len = len(input_tokens)
        positions = torch.arange(seq_len).unsqueeze(0).expand(XDROPE_DIM, -1)
        return positions.clone()


def _make_runner_and_req(prompt_token_ids, prompt_embeds):
    """Create a minimal GPUModelRunner instance and request state."""
    model = FakeXDRoPEModel()
    instance = object.__new__(GPUModelRunner)
    instance.get_model = lambda: model

    req_state = Mock(spec=CachedRequestState)
    req_state.prompt_token_ids = prompt_token_ids
    req_state.prompt_embeds = prompt_embeds
    req_state.mm_features = []
    req_state.xdrope_positions = None
    return instance, req_state


class TestXDRopePromptEmbeds:
    """Verify _init_xdrope_positions handles prompt_embeds-only inputs."""

    def test_prompt_embeds_only_does_not_crash(self):
        """Prompt-embeds-only request must not raise AssertionError."""
        instance, req_state = _make_runner_and_req(
            prompt_token_ids=None,
            prompt_embeds=torch.randn(15, 128),
        )

        instance._init_xdrope_positions(req_state)

        assert req_state.xdrope_positions is not None
        assert req_state.xdrope_positions.shape == (XDROPE_DIM, 15)

    def test_prompt_token_ids_still_works(self):
        """Normal path with prompt_token_ids continues working."""
        instance, req_state = _make_runner_and_req(
            prompt_token_ids=[1, 2, 3, 4, 5],
            prompt_embeds=None,
        )

        instance._init_xdrope_positions(req_state)

        assert req_state.xdrope_positions is not None
        assert req_state.xdrope_positions.shape == (XDROPE_DIM, 5)

    def test_neither_token_ids_nor_embeds_raises(self):
        """When both are None, a ValueError should be raised."""
        instance, req_state = _make_runner_and_req(
            prompt_token_ids=None,
            prompt_embeds=None,
        )

        with pytest.raises(ValueError, match="prompt_token_ids or prompt_embeds"):
            instance._init_xdrope_positions(req_state)
