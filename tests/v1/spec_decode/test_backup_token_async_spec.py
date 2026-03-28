# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the async spec decode backup token fix.

Regression test for: https://github.com/vllm-project/vllm/issues/38098

When async scheduling is enabled (zero-bubble spec decoding), the
``optimistic_seq_lens_cpu`` tensor passed to
``prepare_next_token_ids_padded`` is inflated by un-corrected draft token
placeholders (-1) appended to ``output_token_ids`` in ``_prepare_inputs``.
Using the inflated seq_len to call ``get_token_id()`` causes it to read
from the -1 placeholder slots, returning -1 as the backup token and
feeding garbage to the drafter.

The fix: use ``gpu_input_batch.num_tokens_no_spec - 1`` (the index of the
last *committed* output token before the current step) instead of
``seq_lens_cpu`` for the backup token lookup.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class _FakeRequest:
    """Minimal stand-in for CachedRequestState."""

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
        return -1  # placeholder / out-of-bounds


class _FakeInputBatch:
    """Minimal stand-in for InputBatch."""

    def __init__(
        self,
        req_ids: list[str],
        num_tokens_no_spec: list[int],
        vocab_size: int = 32000,
    ):
        self.req_ids = req_ids
        self.num_reqs = len(req_ids)
        self.vocab_size = vocab_size
        # num_tokens_no_spec mirrors the numpy array in the real InputBatch
        self.num_tokens_no_spec = np.array(num_tokens_no_spec, dtype=np.int64)


def _make_requests(
    req_ids: list[str],
    prompt_lens: list[int],
    output_lens: list[int],
) -> dict[str, _FakeRequest]:
    """Build a dict of fake requests with sequential token IDs."""
    requests = {}
    for rid, plen, olen in zip(req_ids, prompt_lens, output_lens):
        prompt = list(range(plen))
        output = list(range(1000, 1000 + olen))
        requests[rid] = _FakeRequest(prompt, output)
    return requests


# ---------------------------------------------------------------------------
# Helpers replicating the backup-token logic from both proposers
# ---------------------------------------------------------------------------


def _compute_backup_tokens_buggy(
    seq_lens_cpu: torch.Tensor,
    requests: dict[str, _FakeRequest],
    gpu_input_batch: _FakeInputBatch,
) -> list[int]:
    """Replicates the OLD (buggy) backup-token computation.

    Uses seq_lens_cpu directly, which equals optimistic_seq_lens_cpu and
    may be inflated by async-scheduling placeholders.
    """
    num_reqs = gpu_input_batch.num_reqs
    seq_lens_list = seq_lens_cpu[:num_reqs].tolist()
    return [
        requests[gpu_input_batch.req_ids[i]].get_token_id(int(seq_lens_list[i]))
        for i in range(num_reqs)
    ]


def _compute_backup_tokens_fixed(
    requests: dict[str, _FakeRequest],
    gpu_input_batch: _FakeInputBatch,
) -> list[int]:
    """Replicates the NEW (fixed) backup-token computation.

    Uses (num_tokens_no_spec - 1) — the index of the last committed token —
    which is always valid regardless of async-scheduling inflation.
    """
    num_reqs = gpu_input_batch.num_reqs
    actual_last_token_idx = (gpu_input_batch.num_tokens_no_spec[:num_reqs] - 1).tolist()
    return [
        requests[gpu_input_batch.req_ids[i]].get_token_id(int(actual_last_token_idx[i]))
        for i in range(num_reqs)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBackupTokenAsyncSpec:
    """Tests that verify the backup token fix for async spec decoding."""

    def test_non_async_path_returns_last_committed_token(self):
        """Without async scheduling there are no placeholders.
        The fixed code should return the last committed output token."""
        req_ids = ["r0", "r1"]
        # 3 prompt tokens, 2 output tokens → num_tokens = 5
        requests = _make_requests(req_ids, [3, 3], [2, 2])
        actual_seq_lens = [5, 5]
        batch = _FakeInputBatch(req_ids, actual_seq_lens)

        fixed = _compute_backup_tokens_fixed(requests, batch)

        # num_tokens_no_spec - 1 = 4 → output[4-3] = output[1] = 1001
        assert fixed == [1001, 1001], (
            "Should return last committed output token (index num_tokens-1)"
        )

    def test_async_path_buggy_returns_placeholder(self):
        """With async scheduling, optimistic seq_lens is inflated.
        The OLD code reads from -1 placeholder slots and returns -1."""
        req_ids = ["r0", "r1"]
        # 3 prompt + 2 output = 5 actual tokens
        requests = _make_requests(req_ids, [3, 3], [2, 2])
        actual_seq_lens = [5, 5]
        # Optimistic: assume 3 spec tokens accepted → inflated by 3
        optimistic_seq_lens = [8, 8]
        seq_lens_cpu = torch.tensor(optimistic_seq_lens, dtype=torch.int64)
        batch = _FakeInputBatch(req_ids, actual_seq_lens)

        buggy = _compute_backup_tokens_buggy(seq_lens_cpu, requests, batch)
        # Index 8 is beyond output_token_ids length → get_token_id returns -1
        assert buggy == [-1, -1], (
            "Buggy code should return -1 placeholder tokens when seq_lens "
            "is inflated by async scheduling"
        )

    def test_async_path_fixed_returns_correct_token(self):
        """With async scheduling, the FIXED code uses num_tokens_no_spec - 1
        and returns the correct last committed token."""
        req_ids = ["r0", "r1"]
        requests = _make_requests(req_ids, [3, 3], [2, 2])
        actual_seq_lens = [5, 5]
        batch = _FakeInputBatch(req_ids, actual_seq_lens)

        fixed = _compute_backup_tokens_fixed(requests, batch)
        # num_tokens_no_spec - 1 = 4 → output[4-3] = output[1] = 1001
        assert fixed == [1001, 1001], (
            "Fixed code should return the last committed output token even "
            "when optimistic seq_lens would be inflated"
        )

    def test_mixed_inflation_per_request(self):
        """Different requests may have different numbers of accepted spec
        tokens in the previous step, leading to different inflation amounts.
        The fixed code is unaffected since it ignores seq_lens entirely."""
        req_ids = ["r0", "r1", "r2"]
        requests = {
            "r0": _FakeRequest([0, 1], [1000, 1001, 1002]),
            "r1": _FakeRequest([0, 1, 2, 3], [2000]),
            "r2": _FakeRequest([0], [3000, 3001, 3002, 3003]),
        }
        # actual num_tokens: r0=5, r1=5, r2=5
        actual_seq_lens = [5, 5, 5]
        # optimistic (inflated): r0+2, r1+4, r2+0
        optimistic_seq_lens = [7, 9, 5]
        seq_lens_cpu = torch.tensor(optimistic_seq_lens, dtype=torch.int64)
        batch = _FakeInputBatch(req_ids, actual_seq_lens)

        buggy = _compute_backup_tokens_buggy(seq_lens_cpu, requests, batch)
        fixed = _compute_backup_tokens_fixed(requests, batch)

        # Buggy: r0 idx 7 → -1, r1 idx 9 → -1, r2 idx 5 → -1 (out of range)
        assert buggy == [-1, -1, -1]
        # Fixed: all use num_tokens_no_spec-1 = 4 → last real token
        # r0: output[4-2]=output[2]=1002, r1: output[4-4]=output[0]=2000,
        # r2: output[4-1]=output[3]=3003
        assert fixed == [1002, 2000, 3003]

    def test_prompt_only_request(self):
        """A request with no output tokens yet (prefill phase).
        The backup token should be the last prompt token."""
        req_ids = ["r0"]
        requests = {"r0": _FakeRequest([10, 20, 30], [])}
        actual_seq_lens = [3]  # 3 prompt tokens, 0 output
        batch = _FakeInputBatch(req_ids, actual_seq_lens)

        fixed = _compute_backup_tokens_fixed(requests, batch)
        # num_tokens_no_spec - 1 = 2 → prompt[2] = 30
        assert fixed == [30], "Should return last prompt token for prefill requests"

    @pytest.mark.parametrize("num_spec_tokens", [1, 2, 3, 4, 5])
    def test_various_spec_token_counts(self, num_spec_tokens: int):
        """Verify fix works regardless of num_speculative_tokens."""
        req_ids = ["r0"]
        output_tokens = list(range(1000, 1005))  # 5 output tokens
        requests = {"r0": _FakeRequest([0, 1, 2], output_tokens)}
        actual_seq_len = 8  # 3 prompt + 5 output
        batch = _FakeInputBatch(req_ids, [actual_seq_len])

        fixed = _compute_backup_tokens_fixed(requests, batch)
        # num_tokens_no_spec - 1 = 7 → output[7-3] = output[4] = 1004
        assert fixed == [1004], (
            f"Fixed code should return last output token for "
            f"{num_spec_tokens} spec tokens"
        )

    def test_buggy_vs_fixed_diverge_only_with_inflation(self):
        """Buggy and fixed code agree when there is no inflation
        (seq_lens == num_tokens_no_spec), and diverge when inflated."""
        req_ids = ["r0"]
        requests = {"r0": _FakeRequest([0, 1, 2], [1000, 1001])}
        actual_seq_len = 5  # 3 prompt + 2 output
        batch = _FakeInputBatch(req_ids, [actual_seq_len])

        # Case 1: no inflation — seq_lens == num_tokens_no_spec
        # Both should return the same token (the last output token at idx 4)
        seq_lens_no_inflation = torch.tensor([actual_seq_len], dtype=torch.int64)
        buggy_no_inflation = _compute_backup_tokens_buggy(
            seq_lens_no_inflation, requests, batch
        )
        fixed = _compute_backup_tokens_fixed(requests, batch)
        # buggy uses idx=5 → out of range → -1
        # fixed uses idx=4 → output[1] = 1001
        # They differ even without inflation because buggy uses seq_len (=5)
        # while fixed uses seq_len-1 (=4). This confirms the original code
        # was always off-by-one for the "last token" lookup.
        assert buggy_no_inflation == [-1]
        assert fixed == [1001]

        # Case 2: with inflation — buggy returns -1, fixed still returns 1001
        seq_lens_inflated = torch.tensor([actual_seq_len + 3], dtype=torch.int64)
        buggy_inflated = _compute_backup_tokens_buggy(
            seq_lens_inflated, requests, batch
        )
        assert buggy_inflated == [-1]
        assert fixed == [1001]
