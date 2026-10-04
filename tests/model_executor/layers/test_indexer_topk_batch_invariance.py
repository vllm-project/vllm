# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant mode pins the sparse indexer to a row-local top-k."""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.indexer_topk import (
    BATCH_INVARIANT_BACKEND,
    SparseIndexerTopk,
)


@patch("vllm.envs.VLLM_BATCH_INVARIANT", True)
def test_auto_resolves_to_the_row_local_backend():
    """'auto' switches kernels at a row-count threshold, so it cannot stand.

    Under the flag it must resolve to the backend whose selection depends on
    one row only, not on how many rows share the batch.
    """
    topk = SparseIndexerTopk("auto")
    logits = torch.zeros(4, 128, dtype=torch.float32)

    assert topk.resolve_backend(logits, 512, 4) == BATCH_INVARIANT_BACKEND


@patch("vllm.envs.VLLM_BATCH_INVARIANT", True)
@pytest.mark.parametrize("backend", ["cooperative", "persistent", "per_row"])
def test_batch_dependent_backends_are_refused(backend):
    with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT"):
        SparseIndexerTopk(backend)


@patch("vllm.envs.VLLM_BATCH_INVARIANT", True)
def test_the_row_local_backend_is_accepted_explicitly():
    assert SparseIndexerTopk(BATCH_INVARIANT_BACKEND) is not None


@patch("vllm.envs.VLLM_BATCH_INVARIANT", False)
@pytest.mark.parametrize("backend", ["auto", "per_row", "persistent"])
def test_backend_choice_is_untouched_without_the_flag(backend):
    assert SparseIndexerTopk(backend)._backend == backend


def _run(topk, logits, row_lens, topk_tokens):
    out = torch.full((logits.shape[0], topk_tokens), 99, dtype=torch.int32)
    seq_lens = torch.tensor([[n] for n in row_lens], dtype=torch.int32)
    topk.forward(logits, seq_lens, 1, out, topk_tokens, max(row_lens))
    return out


@patch("vllm.envs.VLLM_BATCH_INVARIANT", False)
def test_row_local_selection_is_independent_of_batch_composition():
    """The reason this backend is the one pinned under the flag.

    The same row must select the same columns whether it is alone or sharing
    the batch with longer, shorter and differently-scored rows.
    """
    torch.manual_seed(0)
    topk = SparseIndexerTopk(BATCH_INVARIANT_BACKEND)
    row = torch.randn(1, 256)
    alone = _run(topk, row.clone(), [200], 32)

    for companions in (1, 7, 63):
        batch = torch.cat([row.clone(), torch.randn(companions, 256)])
        lens = [200] + [256] * companions
        assert torch.equal(_run(topk, batch, lens, 32)[0], alone[0])


@patch("vllm.envs.VLLM_BATCH_INVARIANT", False)
def test_rows_narrower_than_topk_are_padded_not_rejected():
    """A row shorter than topk_tokens selects what it has, then -1 fill."""
    topk = SparseIndexerTopk(BATCH_INVARIANT_BACKEND)
    out = _run(topk, torch.randn(1, 128), [128], 2048)

    selected = [i for i in out[0].tolist() if i >= 0]
    assert sorted(selected) == list(range(128))
    assert out[0].tolist()[128:] == [-1] * (2048 - 128)


@patch("vllm.envs.VLLM_BATCH_INVARIANT", False)
def test_fully_filtered_rows_never_return_padding_columns():
    """Candidate filtering writes -inf, so a whole row can tie at -inf.

    Selecting by output rank instead of by column validity lets columns past
    the row's end escape as real token indices.
    """
    topk = SparseIndexerTopk(BATCH_INVARIANT_BACKEND)
    out = _run(topk, torch.full((1, 8), float("-inf")), [2], 4)

    assert sorted(i for i in out[0].tolist() if i >= 0) == [0, 1]
    assert out[0].tolist().count(-1) == 2


@patch("vllm.envs.VLLM_BATCH_INVARIANT", False)
def test_a_filtered_column_never_outranks_a_real_score():
    """Candidate filtering writes -inf; the finite floor is a real score.

    Lifting filtered columns unconditionally would make them tie with a score
    that happens to sit at the floor, and the tie breaks by column index.
    """
    topk = SparseIndexerTopk(BATCH_INVARIANT_BACKEND)
    floor = torch.finfo(torch.float32).min
    logits = torch.tensor([[float("-inf"), floor]])

    out = _run(topk, logits, [2], 1)

    assert out[0].tolist() == [1]


@patch("vllm.envs.VLLM_BATCH_INVARIANT", False)
def test_ranking_follows_the_scores():
    topk = SparseIndexerTopk(BATCH_INVARIANT_BACKEND)
    logits = torch.tensor([[-5.0, 9.0, -1.0, 3.0]])

    out = _run(topk, logits, [4], 3)

    assert out[0].tolist() == [1, 3, 2]


@patch("vllm.envs.VLLM_BATCH_INVARIANT", True)
def test_the_pinned_backend_actually_runs():
    """Auto must resolve *and* execute under the flag, not just resolve."""
    topk = SparseIndexerTopk("auto")
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])

    out = _run(topk, logits, [4], 2)

    assert out[0].tolist() == [1, 3]


@patch("vllm.envs.VLLM_BATCH_INVARIANT", False)
def test_padding_columns_never_fill_the_tail_of_a_short_row():
    """One live score, the rest of the row filtered, padding past the row.

    The tail ties at -inf with the padding columns. The kernels emit
    `0..seq_len-1` whenever the row fits in the budget, so the row's own
    filtered columns come back and only the padding is dropped.
    """
    topk = SparseIndexerTopk(BATCH_INVARIANT_BACKEND)
    logits = torch.full((1, 8), float("-inf"))
    logits[0, 0] = 5.0

    out = _run(topk, logits, [4], 6)

    assert out[0].tolist() == [0, 1, 2, 3, -1, -1]


@patch("vllm.envs.VLLM_BATCH_INVARIANT", False)
def test_a_filtered_column_beats_padding_in_a_short_row():
    """A filtered column is still one of the row's tokens.

    Letting it tie with the padding and lose drops a token the kernels return,
    which changes the attended set rather than just its order.
    """
    topk = SparseIndexerTopk(BATCH_INVARIANT_BACKEND)
    logits = torch.full((1, 4096), float("-inf"))
    logits[0, 0] = 5.0

    out = _run(topk, logits, [2], 512)

    assert sorted(i for i in out[0].tolist() if i >= 0) == [0, 1]


@patch("vllm.envs.VLLM_BATCH_INVARIANT", False)
def test_tied_scores_resolve_the_same_way_in_every_batch():
    """Topk leaves the order of equal scores undefined; a filtered row is all
    ties, so the selection has to be pinned to something. It is the column
    index, and that cannot depend on who else is in the batch."""
    topk = SparseIndexerTopk(BATCH_INVARIANT_BACKEND)
    torch.manual_seed(0)
    logits = torch.randint(0, 3, (4, 64)).float()
    row_lens = [64, 9, 64, 40]

    batched = _run(topk, logits, row_lens, 8)

    for i in range(len(row_lens)):
        alone = _run(topk, logits[i : i + 1], row_lens[i : i + 1], 8)
        assert alone[0].tolist() == batched[i].tolist()
