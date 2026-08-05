# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression gate: the DSv4 sparse builders must agree on the decode boundary.

The indexer, the sparse-SWA builder and the C128A builder all slice the SAME
``topk_indices_buffer`` at ``num_decode_tokens``. Before this gate the indexer
passed ``treat_short_extends_as_decodes=not has_prefilling_rows`` while the other
two took the default ``True``, so in a MIXED prefill+decode batch they tiered a
short extend differently and produced different boundaries -- one wrote at one
offset while another read at a different one, and tokens received each other's
top-k indices.

The indices stay individually valid, so per-slot validity checks and byte-level
sentinels cannot see it; it only shows up as garbled output under concurrency.
"""

import torch

from vllm.v1.attention.backends.utils import (
    sparse_short_extend_tiering,
    split_decodes_and_prefills,
)


class _CM:
    """Minimal CommonAttentionMetadata stand-in for the split helpers."""

    def __init__(self, query_start_loc_cpu, seq_lens_cpu, is_prefilling):
        self.query_start_loc_cpu = query_start_loc_cpu
        self.seq_lens_cpu = seq_lens_cpu
        self.is_prefilling = is_prefilling
        self.num_reqs = len(seq_lens_cpu)
        diffs = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        self.max_query_len = int(diffs.max().item())
        self.num_actual_tokens = int(query_start_loc_cpu[-1].item())


def _mixed_batch():
    # rows: 1 pure decode (q=1), 1 short extend (q=2), 1 real prefill (q=64)
    q = torch.tensor([0, 1, 3, 67], dtype=torch.int32)
    seq = torch.tensor([128, 130, 512], dtype=torch.int32)
    is_prefilling = torch.tensor([False, False, True])
    return _CM(q, seq, is_prefilling)


def test_all_sparse_builders_agree_on_decode_boundary():
    cm = _mixed_batch()
    tiering = sparse_short_extend_tiering(cm)

    # Every consumer of topk_indices_buffer must derive the SAME boundary.
    boundaries = {
        name: split_decodes_and_prefills(
            cm, decode_threshold=threshold,
            treat_short_extends_as_decodes=tiering,
        )[2]  # num_decode_tokens
        for name, threshold in (
            ("indexer", 1),
            ("sparse_swa", 1),
            ("c128a", 1),
        )
    }
    assert len(set(boundaries.values())) == 1, boundaries


def test_tiering_is_false_when_batch_has_prefilling_rows():
    # This is the case the three builders used to disagree on.
    assert sparse_short_extend_tiering(_mixed_batch()) is False


def test_tiering_is_true_for_a_pure_decode_batch():
    q = torch.tensor([0, 1, 2], dtype=torch.int32)
    seq = torch.tensor([128, 130], dtype=torch.int32)
    cm = _CM(q, seq, torch.tensor([False, False]))
    assert sparse_short_extend_tiering(cm) is True
