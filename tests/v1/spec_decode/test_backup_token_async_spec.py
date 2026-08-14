# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the backup token fix in prepare_next_token_ids_padded.

Fixes #38098: with async scheduling, seq_lens_cpu is inflated by unaccepted
draft token placeholders, causing get_token_id() to return -1.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class _FakeRequest:
    def __init__(self, prompt_tokens: list[int], output_tokens: list[int]):
        self.num_prompt_tokens = len(prompt_tokens)
        self._prompt = prompt_tokens
        self._output = output_tokens

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self._output)

    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            return self._prompt[idx]
        out_idx = idx - self.num_prompt_tokens
        if out_idx < len(self._output):
            return self._output[out_idx]
        return -1  # out of range


class _FakeInputBatch:
    def __init__(
        self,
        req_ids: list[str],
        num_tokens_no_spec: list[int],
        vocab_size: int = 32000,
    ):
        self.req_ids = req_ids
        self.num_reqs = len(req_ids)
        self.vocab_size = vocab_size
        self.num_tokens_no_spec = np.array(num_tokens_no_spec, dtype=np.int64)


def _make_requests(
    req_ids: list[str],
    prompt_lens: list[int],
    output_lens: list[int],
) -> dict[str, _FakeRequest]:
    requests = {}
    for rid, plen, olen in zip(req_ids, prompt_lens, output_lens):
        requests[rid] = _FakeRequest(list(range(plen)), list(range(1000, 1000 + olen)))
    return requests


def _backup_buggy(
    seq_lens_cpu: torch.Tensor,
    requests: dict[str, _FakeRequest],
    batch: _FakeInputBatch,
) -> list[int]:
    """Old logic: uses seq_lens_cpu directly (may be inflated)."""
    n = batch.num_reqs
    return [
        requests[batch.req_ids[i]].get_token_id(int(seq_lens_cpu[i])) for i in range(n)
    ]


def _backup_fixed(
    requests: dict[str, _FakeRequest],
    batch: _FakeInputBatch,
) -> list[int]:
    """New logic: uses num_tokens_no_spec - 1 (last committed token)."""
    n = batch.num_reqs
    idx = (batch.num_tokens_no_spec[:n] - 1).tolist()
    return [requests[batch.req_ids[i]].get_token_id(int(idx[i])) for i in range(n)]


class TestBackupTokenAsyncSpec:
    def test_no_inflation_fixed_returns_last_token(self):
        req_ids = ["r0", "r1"]
        requests = _make_requests(req_ids, [3, 3], [2, 2])
        batch = _FakeInputBatch(req_ids, [5, 5])
        # idx = 5-1 = 4 → output[1] = 1001
        assert _backup_fixed(requests, batch) == [1001, 1001]

    def test_inflation_buggy_returns_placeholder(self):
        req_ids = ["r0", "r1"]
        requests = _make_requests(req_ids, [3, 3], [2, 2])
        batch = _FakeInputBatch(req_ids, [5, 5])
        # inflated by 3 spec tokens → idx 8 is out of range
        seq_lens = torch.tensor([8, 8], dtype=torch.int64)
        assert _backup_buggy(seq_lens, requests, batch) == [-1, -1]

    def test_inflation_fixed_returns_correct_token(self):
        req_ids = ["r0", "r1"]
        requests = _make_requests(req_ids, [3, 3], [2, 2])
        batch = _FakeInputBatch(req_ids, [5, 5])
        assert _backup_fixed(requests, batch) == [1001, 1001]

    def test_mixed_inflation_per_request(self):
        req_ids = ["r0", "r1", "r2"]
        requests = {
            "r0": _FakeRequest([0, 1], [1000, 1001, 1002]),
            "r1": _FakeRequest([0, 1, 2, 3], [2000]),
            "r2": _FakeRequest([0], [3000, 3001, 3002, 3003]),
        }
        batch = _FakeInputBatch(req_ids, [5, 5, 5])
        seq_lens = torch.tensor([7, 9, 5], dtype=torch.int64)

        assert _backup_buggy(seq_lens, requests, batch) == [-1, -1, -1]
        assert _backup_fixed(requests, batch) == [1002, 2000, 3003]

    def test_prefill_only_request(self):
        """No output tokens yet — backup should be the last prompt token."""
        req_ids = ["r0"]
        requests = {"r0": _FakeRequest([10, 20, 30], [])}
        batch = _FakeInputBatch(req_ids, [3])
        # idx = 3-1 = 2 → prompt[2] = 30
        assert _backup_fixed(requests, batch) == [30]

    @pytest.mark.parametrize("num_spec_tokens", [1, 2, 3, 4, 5])
    def test_various_spec_token_counts(self, num_spec_tokens: int):
        req_ids = ["r0"]
        requests = {"r0": _FakeRequest([0, 1, 2], list(range(1000, 1005)))}
        batch = _FakeInputBatch(req_ids, [8])
        # idx = 8-1 = 7 → output[4] = 1004
        assert _backup_fixed(requests, batch) == [1004]

    def test_buggy_code_was_always_off_by_one(self):
        """The original code used seq_len as index, which is always one past
        the end of output_token_ids even without async inflation."""
        req_ids = ["r0"]
        requests = {"r0": _FakeRequest([0, 1, 2], [1000, 1001])}
        batch = _FakeInputBatch(req_ids, [5])

        # no inflation: seq_len == num_tokens == 5 → idx 5 is out of range
        seq_lens = torch.tensor([5], dtype=torch.int64)
        assert _backup_buggy(seq_lens, requests, batch) == [-1]
        assert _backup_fixed(requests, batch) == [1001]

        # with inflation: still -1, fixed still correct
        seq_lens_inf = torch.tensor([8], dtype=torch.int64)
        assert _backup_buggy(seq_lens_inf, requests, batch) == [-1]
        assert _backup_fixed(requests, batch) == [1001]
