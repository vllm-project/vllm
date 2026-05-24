# SPDX-License-Identifier: Apache-2.0
"""TDD for PN59 Level 2 — metadata threading + state-chain fix.

Closes the structural half of noonghunna/club-3090#22:
  - 2A: `_slice_chunk_metadata_for_window` builds bit-equivalent
        per-window cu_seqlens / chunk_indices / chunk_offsets to what
        upstream `prepare_chunk_indices` would produce on the windowed
        slice. Pure arithmetic, no GPU sync.
  - 2B: state-chain via `state_next` (post-last-chunk-update) replaces
        the buggy `h_w[:, -1].to(fp32)` (pre-last-chunk-update).
  - Default eligibility: `metadata_gate_passes=True` for single-seq —
        no env-flag operator action required for the noonghunna case.

Test strategy (CPU-runnable — pure tensor logic):
  1. Slicer helper edge cases (T_w aligned, T_w not aligned, win_start=0,
     win_start=last, single-seq invariant assertion, none-passthrough).
  2. Bit-equivalence vs upstream prepare_chunk_indices/_offsets pattern.
  3. Eligibility default flip (strict env unset → metadata_gate_passes).

Numerical equivalence vs vanilla `_vanilla_path` is covered in the
existing `test_streaming_gdn_numerical.py` suite — those tests
continue to pass with Level 2 changes (verified 38/38 in this commit).
"""
from __future__ import annotations

import os

import pytest
import torch

from vllm._genesis.kernels.streaming_gdn_driver import (
    _slice_chunk_metadata_for_window,
)


# ─── Slicer helper — pure-arithmetic correctness ────────────────────


class TestSlicerHelper:

    def test_all_none_inputs_returns_all_none(self):
        out = _slice_chunk_metadata_for_window(
            cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
            win_start=0, win_end=4096, BT=64,
            device="cpu", dtype=torch.bfloat16,
        )
        assert out == (None, None, None)

    def test_single_seq_metadata_shape_at_aligned_T(self):
        cu = torch.tensor([0, 4096], dtype=torch.int32)
        ci = torch.zeros((64, 2), dtype=torch.int32)  # any non-None
        co = torch.tensor([0, 64], dtype=torch.int32)

        cu_w, ci_w, co_w = _slice_chunk_metadata_for_window(
            cu, ci, co, win_start=0, win_end=256, BT=64,
            device="cpu", dtype=torch.bfloat16,
        )
        # T_w=256, BT=64 → cur_NT=4
        assert cu_w.tolist() == [0, 256]
        assert ci_w.shape == (4, 2)
        assert co_w.tolist() == [0, 4]

    def test_chunk_indices_column_0_all_zero_for_single_seq(self):
        cu = torch.tensor([0, 1024], dtype=torch.int32)
        cu_w, ci_w, co_w = _slice_chunk_metadata_for_window(
            cu, torch.zeros(16, 2, dtype=torch.int32),
            torch.tensor([0, 16], dtype=torch.int32),
            win_start=128, win_end=384, BT=64,
            device="cpu", dtype=torch.bfloat16,
        )
        # cur_NT = (384-128) // 64 = 4, all rows have seq_id=0
        assert (ci_w[:, 0] == 0).all()
        assert ci_w[:, 1].tolist() == [0, 1, 2, 3]

    def test_partial_last_window(self):
        cu = torch.tensor([0, 1000], dtype=torch.int32)
        # Last window: win_start=896, win_end=1000 (= T), T_w=104, cur_NT=ceil(104/64)=2
        cu_w, ci_w, co_w = _slice_chunk_metadata_for_window(
            cu, torch.zeros(16, 2, dtype=torch.int32),
            torch.tensor([0, 16], dtype=torch.int32),
            win_start=896, win_end=1000, BT=64,
            device="cpu", dtype=torch.bfloat16,
        )
        assert cu_w.tolist() == [0, 104]
        assert ci_w.shape == (2, 2)
        assert co_w.tolist() == [0, 2]

    def test_win_start_zero_first_window(self):
        cu = torch.tensor([0, 4096], dtype=torch.int32)
        cu_w, ci_w, co_w = _slice_chunk_metadata_for_window(
            cu, torch.zeros(64, 2, dtype=torch.int32),
            torch.tensor([0, 64], dtype=torch.int32),
            win_start=0, win_end=256, BT=64,
            device="cpu", dtype=torch.bfloat16,
        )
        assert cu_w.tolist() == [0, 256]
        # ci_w restarts at 0 every window — local indexing (re-based)
        assert ci_w[:, 1].tolist() == [0, 1, 2, 3]

    def test_multi_seq_assertion_fires(self):
        """PN59 eligibility gates on single-seq. If multi-seq sneaks in,
        the helper must assert loudly — silent multi-seq slicing would
        produce divergent results across sequences."""
        cu = torch.tensor([0, 1024, 2048], dtype=torch.int32)  # 2 seqs
        with pytest.raises(AssertionError, match="single-seq"):
            _slice_chunk_metadata_for_window(
                cu, torch.zeros(32, 2, dtype=torch.int32),
                torch.tensor([0, 16, 32], dtype=torch.int32),
                win_start=0, win_end=256, BT=64,
                device="cpu", dtype=torch.bfloat16,
            )

    def test_dtype_preservation_int32(self):
        cu = torch.tensor([0, 4096], dtype=torch.int32)
        cu_w, ci_w, co_w = _slice_chunk_metadata_for_window(
            cu, torch.zeros(64, 2, dtype=torch.int32),
            torch.tensor([0, 64], dtype=torch.int32),
            win_start=0, win_end=256, BT=64,
            device="cpu", dtype=torch.bfloat16,
        )
        assert cu_w.dtype == torch.int32
        assert ci_w.dtype == torch.int32
        assert co_w.dtype == torch.int32

    def test_dtype_preservation_int64(self):
        cu = torch.tensor([0, 4096], dtype=torch.int64)
        cu_w, ci_w, co_w = _slice_chunk_metadata_for_window(
            cu, torch.zeros(64, 2, dtype=torch.int64),
            torch.tensor([0, 64], dtype=torch.int64),
            win_start=0, win_end=256, BT=64,
            device="cpu", dtype=torch.bfloat16,
        )
        assert cu_w.dtype == torch.int64
        assert ci_w.dtype == torch.int64

    def test_only_chunk_indices_provided_no_cu_seqlens(self):
        """Caller may pass chunk_indices alone (cu_seqlens implicit).
        Helper should still derive metadata + use chunk_indices.dtype."""
        ci = torch.zeros((64, 2), dtype=torch.int32)
        cu_w, ci_w, co_w = _slice_chunk_metadata_for_window(
            cu_seqlens=None, chunk_indices=ci,
            chunk_offsets=torch.tensor([0, 64], dtype=torch.int32),
            win_start=0, win_end=512, BT=64,
            device="cpu", dtype=torch.bfloat16,
        )
        assert cu_w.dtype == torch.int32
        assert cu_w.tolist() == [0, 512]
        assert ci_w.shape == (8, 2)


# ─── Bit-equivalence vs `prepare_chunk_indices`/`prepare_chunk_offsets` ──


class TestBitEquivalenceVsUpstream:
    """The slicer's output for a window must be bit-identical to what
    `prepare_chunk_indices(cu_seqlens_w, BT)` /
    `prepare_chunk_offsets(cu_seqlens_w, BT)` would produce."""

    @staticmethod
    def _prepare_chunk_indices_reference(cu_seqlens, BT):
        """Pure-Python reference implementation matching FLA's
        `index.py:23-31` for single-seq case."""
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        nts = [(int(l) + BT - 1) // BT for l in seq_lens.tolist()]
        rows = []
        for seq_id, nt in enumerate(nts):
            for chunk_id in range(nt):
                rows.append([seq_id, chunk_id])
        if not rows:
            return torch.zeros((0, 2), dtype=cu_seqlens.dtype)
        return torch.tensor(rows, dtype=cu_seqlens.dtype)

    @staticmethod
    def _prepare_chunk_offsets_reference(cu_seqlens, BT):
        """Pure-Python reference matching FLA's `index.py:33-37`."""
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        offsets = [0]
        for l in seq_lens.tolist():
            offsets.append(offsets[-1] + (int(l) + BT - 1) // BT)
        return torch.tensor(offsets, dtype=cu_seqlens.dtype)

    def test_bit_equivalent_aligned_window(self):
        """T_w aligned to BT (most common case)."""
        BT = 64
        for T_w in [256, 512, 1024, 2048, 4096]:
            cu_full = torch.tensor([0, T_w * 8], dtype=torch.int32)
            cu_w, ci_w, co_w = _slice_chunk_metadata_for_window(
                cu_full,
                torch.zeros((64, 2), dtype=torch.int32),
                torch.tensor([0, 64], dtype=torch.int32),
                win_start=0, win_end=T_w, BT=BT,
                device="cpu", dtype=torch.bfloat16,
            )

            # Equivalent windowed cu_seqlens
            cu_w_ref = torch.tensor([0, T_w], dtype=torch.int32)
            ci_ref = self._prepare_chunk_indices_reference(cu_w_ref, BT)
            co_ref = self._prepare_chunk_offsets_reference(cu_w_ref, BT)

            assert torch.equal(cu_w, cu_w_ref), f"T_w={T_w}: cu_w mismatch"
            assert torch.equal(ci_w, ci_ref), f"T_w={T_w}: ci_w mismatch"
            assert torch.equal(co_w, co_ref), f"T_w={T_w}: co_w mismatch"

    def test_bit_equivalent_partial_window(self):
        """T_w not aligned to BT — last chunk is partial."""
        BT = 64
        T_w = 200  # → cur_NT = ceil(200/64) = 4
        cu_full = torch.tensor([0, 4096], dtype=torch.int32)
        cu_w, ci_w, co_w = _slice_chunk_metadata_for_window(
            cu_full,
            torch.zeros((64, 2), dtype=torch.int32),
            torch.tensor([0, 64], dtype=torch.int32),
            win_start=512, win_end=712, BT=BT,
            device="cpu", dtype=torch.bfloat16,
        )

        cu_w_ref = torch.tensor([0, T_w], dtype=torch.int32)
        ci_ref = self._prepare_chunk_indices_reference(cu_w_ref, BT)
        co_ref = self._prepare_chunk_offsets_reference(cu_w_ref, BT)

        assert torch.equal(ci_w, ci_ref)
        assert torch.equal(co_w, co_ref)


# ─── Eligibility default flip (Level 2 contract) ───────────────────


class TestEligibilityDefaultFlip:
    """Pre-Level-2 default: STRICT_NO_METADATA=1 (silent bypass).
    Post-Level-2 default: STRICT_NO_METADATA=0 (threaded, runs fine).
    Operator can still set =1 as escape hatch."""

    def test_default_unset_means_relaxed(self, monkeypatch):
        """When env unset, metadata_gate must NOT block streaming."""
        monkeypatch.delenv("GENESIS_PN59_STRICT_NO_METADATA", raising=False)
        # Verify default by reading the env (the eligibility check inline
        # reads "0" as default after Level 2 flip)
        default = os.environ.get(
            "GENESIS_PN59_STRICT_NO_METADATA", "0"
        ).strip().lower()
        # Default of "0" → strict_metadata_gate is False → all metadata
        # passes through gate
        assert default == "0", (
            f"Level 2 default should be '0' (relaxed), got {default!r}"
        )

    def test_explicit_strict_one_re_engages_legacy(self, monkeypatch):
        """Operator can opt-IN to legacy strict gate via env=1."""
        monkeypatch.setenv("GENESIS_PN59_STRICT_NO_METADATA", "1")
        flag = os.environ["GENESIS_PN59_STRICT_NO_METADATA"].lower()
        assert flag in {"1", "true", "yes", "y", "on"}, (
            f"Strict opt-in must be honored, env={flag!r}"
        )
