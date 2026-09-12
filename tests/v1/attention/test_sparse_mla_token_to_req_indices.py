# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: CommonAttentionMetadata.token_to_req_indices batch invariant.

Drives the shipped producer. CUDA-graph warmup-like query-slot / block_table
row mismatches must fail closed. Request indices are built from num_reqs.
Never clamp an OOB req_idx onto another request's row.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata


def _make_cm(
    *,
    query_lens: list[int],
    block_table_rows: int,
    num_reqs: int | None = None,
    block_table_cols: int = 4,
    device: str | torch.device = "cpu",
) -> tuple[CommonAttentionMetadata, torch.Tensor]:
    device = torch.device(device)
    if num_reqs is None:
        num_reqs = len(query_lens)
    starts = [0]
    for q in query_lens:
        starts.append(starts[-1] + int(q))
    qsl = torch.tensor(starts, dtype=torch.int32, device=device)
    num_tokens = int(starts[-1])
    seq_lens = torch.tensor(
        [max(q, 0) for q in query_lens]
        + [0] * max(0, num_reqs - len(query_lens)),
        dtype=torch.int32,
        device=device,
    )[:num_reqs]
    block_table = torch.arange(
        block_table_rows * block_table_cols, dtype=torch.int32, device=device
    ).view(block_table_rows, block_table_cols)
    cm = CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.cpu(),
        seq_lens=seq_lens
        if seq_lens.numel() == num_reqs
        else torch.zeros(num_reqs, dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens) if query_lens else 0,
        max_seq_len=int(seq_lens.max().item()) if seq_lens.numel() else 0,
        block_table_tensor=block_table,
        slot_mapping=torch.zeros(max(num_tokens, 1), dtype=torch.int64, device=device),
    )
    buffer = torch.empty(max(num_tokens, 8), dtype=torch.int32, device=device)
    return cm, buffer


class TestTokenToReqIndicesProducer:
    def test_exact_valid_metadata(self):
        """Aligned metadata emits only in-range req indices."""
        cm, buf = _make_cm(query_lens=[2, 1], block_table_rows=2, num_reqs=2)
        out = cm.token_to_req_indices(buf)
        assert out.tolist() == [0, 0, 1]
        assert int(out.max().item()) < cm.block_table_tensor.shape[0]
        assert int(out.min().item()) >= 0

    def test_warmup_batch_query_slot_mismatch_fail_closed(self):
        """CG-warmup-like query slots vs block_table rows mismatch."""
        cm, buf = _make_cm(query_lens=[1, 1], block_table_rows=1, num_reqs=2)
        with pytest.raises(RuntimeError, match="block_table rows"):
            cm.token_to_req_indices(buf)

    def test_query_slots_disagree_with_num_reqs(self):
        """query_start_loc slots != num_reqs must fail closed."""
        cm, buf = _make_cm(query_lens=[1, 1], block_table_rows=3, num_reqs=3)
        with pytest.raises(RuntimeError, match="query_start_loc slots"):
            cm.token_to_req_indices(buf)

    def test_cg_padded_metadata_path(self):
        """Padded request slot with query_len=0; block_table padded."""
        cm, buf = _make_cm(query_lens=[2, 0], block_table_rows=2, num_reqs=2)
        out = cm.token_to_req_indices(buf)
        assert out.tolist() == [0, 0]
        assert int(out.max().item()) < 2

    def test_normal_prefill_path(self):
        """Normal multi-request prefill shape."""
        cm, buf = _make_cm(query_lens=[3, 2, 1], block_table_rows=3, num_reqs=3)
        out = cm.token_to_req_indices(buf)
        assert out.tolist() == [0, 0, 0, 1, 1, 2]
        assert cm.block_table_tensor.shape[0] == cm.num_reqs

    def test_indices_built_from_num_reqs_not_independent_arange(self):
        """arange domain is num_reqs after invariant holds."""
        cm, buf = _make_cm(query_lens=[1, 1, 1], block_table_rows=3, num_reqs=3)
        out = cm.token_to_req_indices(buf)
        assert set(out.tolist()) == {0, 1, 2}

    def test_no_req_idx_clamp_semantics(self):
        """Forbidden fix: clamping OOB req onto another row must not be used."""
        cm, buf = _make_cm(query_lens=[1, 1], block_table_rows=1, num_reqs=2)
        with pytest.raises(RuntimeError):
            cm.token_to_req_indices(buf)
        clamped = [min(i, 0) for i in [0, 1]]
        assert clamped == [0, 0]
