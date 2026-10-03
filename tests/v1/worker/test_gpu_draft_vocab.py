# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.v1.worker.gpu.spec_decode.draft_vocab import (
    DraftVocab,
    load_draft_token_ids,
)

VOCAB = 200
HIDDEN = 32
DRAFT_IDS = [0, 3, 7, 50, 51, 128, 130, 199]


class _Drafter(nn.Module):
    """MTP-like drafter holding the target head at the top level and in a
    per-layer shared_head, plus a shared input embedding."""

    def __init__(self, lm_head: nn.Module, embed: nn.Module):
        super().__init__()
        self.lm_head = lm_head
        self.embed_tokens = embed
        self.shared_head = nn.Module()
        self.shared_head.head = lm_head


@pytest.fixture
def lm_head(default_vllm_config) -> ParallelLMHead:
    head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen))
    head.quant_method.process_weights_after_loading(head)
    return head


@pytest.fixture
def logits_processor(default_vllm_config) -> LogitsProcessor:
    return LogitsProcessor(VOCAB)


def _hidden(n: int = 64) -> torch.Tensor:
    return torch.randn(n, HIDDEN, generator=torch.Generator().manual_seed(1))


@pytest.mark.cpu_test
@pytest.mark.parametrize("fmt", ["pt_list", "pt_tensor", "json"])
def test_load_draft_token_ids(tmp_path, fmt):
    ids = [9, 2, 9, 4]
    if fmt == "pt_list":
        path = tmp_path / "map.pt"
        torch.save(ids, path)
    elif fmt == "pt_tensor":
        path = tmp_path / "map.pt"
        torch.save(torch.tensor(ids), path)
    else:
        path = tmp_path / "map.json"
        path.write_text(json.dumps(ids))
    token_ids = load_draft_token_ids(str(path), extra_ids=[7, 2])
    assert token_ids.tolist() == [2, 4, 7, 9]
    assert token_ids.dtype == torch.int64


@pytest.mark.cpu_test
@pytest.mark.parametrize("bad", [-1, VOCAB])
def test_draft_vocab_rejects_out_of_range_ids(lm_head, bad):
    with pytest.raises(ValueError, match="outside"):
        DraftVocab(torch.tensor([0, bad]), lm_head)


@pytest.mark.cpu_test
def test_draft_vocab_rejects_quantized_head(lm_head):
    # Packed quantized heads (e.g. GPTQ qweight) have no `weight` to slice.
    del lm_head.weight
    lm_head.quant_method = object()
    with pytest.raises(ValueError, match="unquantized"):
        DraftVocab(torch.tensor(DRAFT_IDS), lm_head)


@pytest.mark.cpu_test
def test_install_replaces_only_shared_heads(lm_head):
    embed = VocabParallelEmbedding(VOCAB, HIDDEN, disable_tp=True)
    drafter = _Drafter(lm_head, embed)
    dv = DraftVocab(torch.tensor(DRAFT_IDS), lm_head)
    dv.install(drafter, lm_head)
    assert drafter.lm_head is dv.head
    assert drafter.shared_head.head is dv.head
    assert drafter.embed_tokens is embed
    assert dv.head.weight.shape == (len(DRAFT_IDS), HIDDEN)


@pytest.mark.cpu_test
def test_install_requires_shared_head(lm_head):
    own_head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    drafter = _Drafter(own_head, nn.Identity())
    dv = DraftVocab(torch.tensor(DRAFT_IDS), lm_head)
    with pytest.raises(ValueError, match="shares the target"):
        dv.install(drafter, lm_head)


@pytest.mark.cpu_test
def test_greedy_draft_matches_restricted_full_argmax(lm_head, logits_processor):
    ids = torch.tensor(DRAFT_IDS)
    hidden = _hidden()
    full = logits_processor(lm_head, hidden)
    expected = ids[full[:, ids].argmax(dim=-1)]

    drafter = _Drafter(lm_head, nn.Identity())
    dv = DraftVocab(ids, lm_head)
    dv.install(drafter, lm_head)

    logits = dv.restrict(logits_processor(drafter.lm_head, hidden))
    assert logits.shape == (hidden.shape[0], len(DRAFT_IDS))
    assert torch.equal(dv.target_ids[logits.argmax(dim=-1)], expected)

    # Local-argmax path (use_local_argmax_reduction) maps columns the same way.
    top = logits_processor.get_top_tokens(drafter.lm_head, hidden)
    assert torch.equal(dv.col_to_target[top], expected)


@pytest.mark.cpu_test
def test_probabilistic_draft_probs_are_exact_over_full_vocab(lm_head, logits_processor):
    ids = torch.tensor(DRAFT_IDS)
    hidden = _hidden(8)
    drafter = _Drafter(lm_head, nn.Identity())
    dv = DraftVocab(ids, lm_head)
    dv.install(drafter, lm_head)

    reduced = dv.restrict(logits_processor(drafter.lm_head, hidden))
    probs = dv.scatter(reduced, VOCAB).softmax(dim=-1)

    outside = torch.ones(VOCAB, dtype=torch.bool)
    outside[ids] = False
    assert torch.all(probs[:, outside] == 0)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(hidden.shape[0]))
    full = logits_processor(lm_head, hidden)
    torch.testing.assert_close(probs[:, ids], full[:, ids].softmax(dim=-1))


@pytest.mark.cpu_test
def test_dynamic_rows_at_full_rank_and_width_match_full_argmax(
    lm_head, logits_processor
):
    """With every off-list row picked, drafting is exact full-vocab drafting."""
    ids = torch.tensor(DRAFT_IDS)
    hidden = _hidden()
    drafter = _Drafter(lm_head, nn.Identity())
    dv = DraftVocab(ids, lm_head, VOCAB - len(DRAFT_IDS), HIDDEN)
    dv.install(drafter, lm_head)

    logits = dv.restrict(logits_processor(drafter.lm_head, hidden))
    full = logits_processor(lm_head, hidden)
    assert torch.equal(dv.to_target(logits.argmax(dim=-1)), full.argmax(dim=-1))
    torch.testing.assert_close(dv.scatter(logits, VOCAB), full)


@pytest.mark.cpu_test
@pytest.mark.parametrize("rank", [4, HIDDEN])
def test_dynamic_rows_draft_over_list_plus_picked_rows(lm_head, logits_processor, rank):
    """The proposal is the exact softmax over the list plus each token's picks."""
    ids = torch.tensor(DRAFT_IDS)
    hidden = _hidden(8)
    drafter = _Drafter(lm_head, nn.Identity())
    dv = DraftVocab(ids, lm_head, 16, rank)
    dv.install(drafter, lm_head)

    logits = dv.restrict(logits_processor(drafter.lm_head, hidden))
    picked = dv.dynamic.ids
    assert logits.shape == (hidden.shape[0], len(DRAFT_IDS) + 16)
    assert not torch.isin(picked, ids).any()

    full = logits_processor(lm_head, hidden)
    allowed = torch.zeros_like(full, dtype=torch.bool)
    allowed[:, ids] = True
    allowed.scatter_(1, picked, True)
    expected = full.masked_fill(~allowed, float("-inf"))
    torch.testing.assert_close(dv.scatter(logits, VOCAB), expected)
    assert torch.equal(dv.to_target(logits.argmax(dim=-1)), expected.argmax(dim=-1))


class _FakeTPHead(nn.Module):
    """One TP rank of a vocab-parallel lm_head, without a process group."""

    _get_indices = VocabParallelEmbedding._get_indices

    def __init__(self, full_weight: torch.Tensor, tp_rank: int, tp_size: int):
        super().__init__()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.org_vocab_size = self.num_embeddings = full_weight.shape[0]
        self.org_vocab_size_padded = self.num_embeddings_padded = 256
        self.embedding_dim = full_weight.shape[1]
        self.shard_indices = self._get_indices(256, 256, VOCAB, VOCAB, tp_rank, tp_size)
        rows = 256 // tp_size
        padded = torch.cat([full_weight, full_weight.new_zeros(256 - VOCAB, HIDDEN)])
        self.weight = nn.Parameter(padded[tp_rank * rows : (tp_rank + 1) * rows])
        self.quant_method = UnquantizedEmbeddingMethod()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("draft_ids", "rank1_pad"),
    # Rank 0 owns ids [0, 128), rank 1 owns [128, 200).
    [(DRAFT_IDS, 2), ([0, 3, 7, 50, 51], 5)],
    ids=["uneven", "empty_rank"],
)
def test_tensor_parallel_layout_pads_ranks(default_vllm_config, draft_ids, rank1_pad):
    full_w = torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(2))
    ids = torch.tensor(draft_ids)
    ranks = [DraftVocab(ids, _FakeTPHead(full_w, r, 2)) for r in (0, 1)]
    dv = ranks[0]
    assert ranks[1].head.shard_indices.num_org_vocab_padding == rank1_pad
    assert dv.col_to_target[dv.valid_cols].tolist() == draft_ids

    hidden = _hidden()
    # Emulate the TP all-gather of each rank's reduced logits.
    gathered = torch.cat([hidden @ r.head.weight.t() for r in ranks], dim=-1)
    expected = ids[(hidden @ full_w[ids].t()).argmax(-1)]
    assert torch.equal(dv.target_ids[dv.restrict(gathered).argmax(-1)], expected)
