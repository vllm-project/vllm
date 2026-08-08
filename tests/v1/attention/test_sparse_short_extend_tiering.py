# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression gate: the DSv4 sparse builders must agree on the decode boundary.

The indexer, the sparse-SWA builder and the C128A builder all slice the SAME
``topk_indices_buffer`` at ``num_decode_tokens``. If they disagree about where
that boundary falls, one writes at one offset while another reads at a
different one and tokens receive each other's top-k indices. Every index stays
individually valid -- a real slot in the owning request's block table -- so
per-slot validity checks and byte-level sentinels cannot see it; it surfaces
only as garbled output, and only under concurrency, since a pure-decode or
pure-prefill batch cannot expose the disagreement.

Two independent things move that boundary, and each gets a test:

  1. ``treat_short_extends_as_decodes`` -- fixed here via
     ``sparse_short_extend_tiering()``. Asserted by inspecting the three call
     sites, because asserting on a shared helper's return value cannot fail
     when the builders do not call it.

  2. ``decode_threshold`` -- NOT fixed here, and asserted as a documented
     divergence so the next reader does not take axis 1 for the whole story.
"""

import ast
import inspect

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
    # 1 pure decode (q=1), 1 short extend (q=2), 1 real prefill (q=64)
    q = torch.tensor([0, 1, 3, 67], dtype=torch.int32)
    seq = torch.tensor([128, 130, 512], dtype=torch.int32)
    return _CM(q, seq, torch.tensor([False, False, True]))


def _tiering_call_sites() -> dict[str, str]:
    """The ``treat_short_extends_as_decodes=`` expression at each builder."""
    from vllm.models.deepseek_v4 import sparse_mla
    from vllm.v1.attention.backends.mla import indexer, sparse_swa

    found: dict[str, str] = {}
    for name, module in (
        ("indexer", indexer),
        ("sparse_swa", sparse_swa),
        ("c128a", sparse_mla),
    ):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            fname = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if fname != "split_decodes_and_prefills":
                continue
            for kw in node.keywords:
                if kw.arg == "treat_short_extends_as_decodes":
                    found[name] = ast.unparse(kw.value)
    return found


def test_every_builder_derives_the_flag_from_the_shared_helper():
    """The coupling, asserted where it can actually break.

    A test that calls one helper three times with the same arguments agrees
    with itself whatever the builders do -- it passes unchanged against a tree
    where none of them were touched, so it guards nothing. What has to hold is
    that each call site *uses* the helper, so that is what is inspected.
    """
    sites = _tiering_call_sites()
    assert set(sites) == {"indexer", "sparse_swa", "c128a"}, sites
    for name, expr in sites.items():
        assert "sparse_short_extend_tiering" in expr, (
            f"{name} derives treat_short_extends_as_decodes independently "
            f"({expr!r}); it will drift from the others again"
        )


def test_tiering_is_false_when_batch_has_prefilling_rows():
    assert sparse_short_extend_tiering(_mixed_batch()) is False


def test_tiering_is_true_for_a_pure_decode_batch():
    q = torch.tensor([0, 1, 2], dtype=torch.int32)
    seq = torch.tensor([128, 130], dtype=torch.int32)
    cm = _CM(q, seq, torch.tensor([False, False]))
    assert sparse_short_extend_tiering(cm) is True


def test_decode_threshold_is_a_second_unfixed_divergence():
    """Aligning the tiering flag is necessary but not sufficient.

    The indexer's threshold is ``num_speculative_tokens + 1`` (indexer.py, its
    paged-MQA next_n) while sparse-SWA and C128A both use
    ``1 + (2 if parallel_drafting else 1) * num_speculative_tokens``
    (sparse_swa.py, and backend.py's ``_init_reorder_batch_threshold``). Under
    DSpark parallel drafting with k=5 that is 6 against 11, so a row whose
    query length falls in (6, 11] is tiered prefill by the producer of
    ``topk_indices_buffer`` and decode by both of its consumers -- the same
    boundary misattribution this file guards for short extends, reached by a
    different route and still open.

    This pins the arithmetic rather than a fix, so the gap is not mistaken for
    covered ground.
    """
    k = 5
    indexer_threshold = k + 1
    swa_threshold = 1 + 2 * k  # parallel_drafting
    assert indexer_threshold != swa_threshold

    qlen = 8  # between the two thresholds
    assert indexer_threshold < qlen <= swa_threshold
    q = torch.tensor([0, 1, 1 + qlen], dtype=torch.int32)
    seq = torch.tensor([128, 256], dtype=torch.int32)
    cm = _CM(q, seq, torch.tensor([False, False]))

    tiering = sparse_short_extend_tiering(cm)
    indexer_decode_tokens = split_decodes_and_prefills(
        cm, decode_threshold=indexer_threshold, treat_short_extends_as_decodes=tiering
    )[2]
    swa_decode_tokens = split_decodes_and_prefills(
        cm, decode_threshold=swa_threshold, treat_short_extends_as_decodes=tiering
    )[2]

    assert indexer_decode_tokens != swa_decode_tokens, (
        "the threshold axis no longer diverges -- if that was fixed "
        "deliberately, delete this test and say so in the commit"
    )
