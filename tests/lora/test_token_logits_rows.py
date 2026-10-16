# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt logprob chunks map lm_head LoRA logits rows to batch tokens."""

from contextlib import contextmanager

import pytest
import torch

from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample import prompt_logprob

pytestmark = pytest.mark.skip_global_cleanup


def test_prompt_logprobs_chunks_use_token_rows(monkeypatch):
    num_tokens = prompt_logprob.CHUNK_SIZE + 3
    calls: list[tuple[int, int, int]] = []
    active: list[tuple[int, int]] = []

    @contextmanager
    def token_logits_rows(start: int, end: int):
        active.append((start, end))
        yield
        active.pop()

    def logits_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        # Every logits call must run inside token_logits_rows for its rows.
        ((start, end),) = active
        calls.append((start, end, hidden_states.shape[0]))
        return torch.zeros(hidden_states.shape[0], 8)

    def fake_topk(logits, num_logprobs, target_token_ids, **kwargs):
        n = logits.shape[0]
        return LogprobsTensors(
            logprob_token_ids=target_token_ids[:, None],
            logprobs=torch.zeros(n, 1),
            selected_token_ranks=torch.zeros(n, dtype=torch.int64),
        )

    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", fake_topk)
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.arange(num_tokens),
        torch.zeros(num_tokens, 2),
        logits_fn,
        0,
        token_logits_rows=token_logits_rows,
    )
    chunk = prompt_logprob.CHUNK_SIZE
    assert calls == [(0, chunk, chunk), (chunk, 2 * chunk, 3)]
