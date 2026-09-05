# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.sample.ops.fused_top_logprobs import (
    fused_top_logprobs,
    fused_top_logprobs_enabled,
)


def test_fused_top_logprobs_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_FUSED_TOP_LOGPROBS", raising=False)

    assert not fused_top_logprobs_enabled()


def test_fused_top_logprobs_env_gate(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_FUSED_TOP_LOGPROBS", "1")
    assert fused_top_logprobs_enabled()

    monkeypatch.setenv("VLLM_ENABLE_FUSED_TOP_LOGPROBS", "true")
    assert fused_top_logprobs_enabled()

    monkeypatch.setenv("VLLM_ENABLE_FUSED_TOP_LOGPROBS", "0")
    assert not fused_top_logprobs_enabled()


def test_fused_top_logprobs_returns_none_on_cpu():
    logits = torch.randn(2, 16)
    token_ids = torch.tensor([1, 2], dtype=torch.int64)

    assert fused_top_logprobs(logits, token_ids, 4) is None
