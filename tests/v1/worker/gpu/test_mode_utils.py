# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.sample.mode_utils import compute_prompt_scores_for_mode


def test_compute_prompt_scores_for_mode_logits_uses_raw_logits():
    logits = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    called = False

    def _compute_logprobs(x):
        nonlocal called
        called = True
        return x.log_softmax(dim=-1, dtype=torch.float32)

    out = compute_prompt_scores_for_mode(logits, "raw_logits", _compute_logprobs)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, logits.to(torch.float32))
    assert not called


def test_compute_prompt_scores_for_mode_logprobs_uses_log_softmax():
    logits = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    called = False

    def _compute_logprobs(x):
        nonlocal called
        called = True
        return x.log_softmax(dim=-1, dtype=torch.float32)

    out = compute_prompt_scores_for_mode(logits, "raw_logprobs", _compute_logprobs)
    expected = logits.log_softmax(dim=-1, dtype=torch.float32)
    torch.testing.assert_close(out, expected)
    assert called
