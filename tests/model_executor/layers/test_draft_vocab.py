# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.draft_vocab import DraftVocab, load_draft_token_ids
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import LocalArgmaxMixin

VOCAB = 200
HIDDEN = 32
DRAFT_IDS = [0, 3, 7, 50, 51, 128, 130, 199]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _head(vocab: int = VOCAB, device: str = "cpu") -> ParallelLMHead:
    head = ParallelLMHead(vocab, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen))
    head.quant_method.process_weights_after_loading(head)
    return head.to(device)


def _hidden(n: int = 64, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator().manual_seed(1)
    return torch.randn(n, HIDDEN, generator=gen).to(device)


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
def test_pruned_head_maps_draft_ids_to_target_ids(default_vllm_config):
    """EAGLE3/DFlash pruned heads: logits land on their d2t target ids."""
    head = _head(len(DRAFT_IDS))
    draft_vocab = DraftVocab(LogitsProcessor(len(DRAFT_IDS)), VOCAB)
    d2t = draft_vocab.draft_id_to_target_id
    assert d2t is not None and d2t.shape == (len(DRAFT_IDS),)
    d2t.data.copy_(torch.tensor(DRAFT_IDS) - torch.arange(len(DRAFT_IDS)))

    hidden = _hidden()
    draft_logits = draft_vocab.logits_processor(head, hidden)
    logits = draft_vocab.compute_logits(head, hidden)
    assert logits.shape == (hidden.shape[0], VOCAB)
    assert torch.equal(logits[:, DRAFT_IDS], draft_logits)
    outside = torch.ones(VOCAB, dtype=torch.bool)
    outside[DRAFT_IDS] = False
    assert torch.all(logits[:, outside] == float("-inf"))
    top = draft_vocab.get_top_tokens(head, hidden)
    assert torch.equal(top, logits.argmax(dim=-1))

    # A full-vocabulary head is a no-op.
    assert DraftVocab(LogitsProcessor(VOCAB), VOCAB).draft_id_to_target_id is None


@pytest.mark.cpu_test
def test_token_map_drafts_over_listed_rows(default_vllm_config, dist_init):
    head = _head()
    draft_vocab = DraftVocab(LogitsProcessor(VOCAB), VOCAB)
    full = draft_vocab.compute_logits(head, _hidden())
    draft_vocab.load_token_map(head, torch.tensor(DRAFT_IDS))

    logits = draft_vocab.compute_logits(head, _hidden())
    expected = torch.full_like(full, float("-inf"))
    expected[:, DRAFT_IDS] = full[:, DRAFT_IDS]
    torch.testing.assert_close(logits, expected)
    top = draft_vocab.get_top_tokens(head, _hidden())
    assert torch.equal(top, logits.argmax(dim=-1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_rows", [VOCAB - len(DRAFT_IDS), 16])
@pytest.mark.parametrize("rank", [4, HIDDEN])
def test_dynamic_rows_draft_over_list_plus_picked_rows(
    default_vllm_config, dist_init, device, num_rows, rank
):
    """Each token drafts over the list plus `num_rows` picked rows, with the
    exact logits of the full head on all of them."""
    head = _head(device=device)
    hidden = _hidden(device=device)
    draft_vocab = DraftVocab(LogitsProcessor(VOCAB), VOCAB)
    full = draft_vocab.compute_logits(head, hidden).float()
    draft_vocab.load_token_map(head, torch.tensor(DRAFT_IDS), num_rows, rank)

    logits = draft_vocab.compute_logits(head, hidden).float()
    kept = logits.isfinite()
    assert torch.all(kept[:, DRAFT_IDS])
    assert torch.all(kept.sum(dim=-1) == len(DRAFT_IDS) + num_rows)
    torch.testing.assert_close(logits[kept], full[kept], atol=1e-4, rtol=1e-4)
    top = draft_vocab.get_top_tokens(head, hidden)
    assert torch.equal(top, logits.argmax(dim=-1))


@pytest.mark.cpu_test
def test_local_argmax_mixin_without_draft_vocab(default_vllm_config):
    """Target architectures (e.g. Llama, Qwen3) mix it in without a DraftVocab."""

    class _Model(LocalArgmaxMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = _head()
            self.logits_processor = LogitsProcessor(VOCAB)

    model = _Model()
    logits = model.logits_processor(model.lm_head, _hidden())
    assert torch.equal(model.get_top_tokens(_hidden()), logits.argmax(dim=-1))


class _FakeTPHead(VocabParallelEmbedding):
    """One TP rank of a vocab-parallel lm_head, without a process group."""

    def __init__(self, full_weight: torch.Tensor, tp_rank: int, tp_size: int):
        nn.Module.__init__(self)
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.org_vocab_size = self.num_embeddings = full_weight.shape[0]
        self.org_vocab_size_padded = self.num_embeddings_padded = 256
        self.shard_indices = self._get_indices(256, 256, VOCAB, VOCAB, tp_rank, tp_size)
        rows = 256 // tp_size
        padded = torch.cat([full_weight, full_weight.new_zeros(256 - VOCAB, HIDDEN)])
        self.weight = nn.Parameter(padded[tp_rank * rows : (tp_rank + 1) * rows])
        self.quant_method = UnquantizedEmbeddingMethod()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "draft_ids",
    # Rank 0 owns ids [0, 128), rank 1 owns [128, 200).
    [DRAFT_IDS, [0, 3, 7, 50, 51]],
    ids=["uneven", "empty_rank"],
)
def test_tensor_parallel_ranks_keep_their_listed_rows(
    default_vllm_config, dist_init, draft_ids
):
    full_w = torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(2))
    hidden = _hidden()
    rank_logits = []
    for rank in (0, 1):
        draft_vocab = DraftVocab(LogitsProcessor(VOCAB), VOCAB)
        draft_vocab.load_token_map(
            _FakeTPHead(full_w, rank, 2), torch.tensor(draft_ids)
        )
        assert draft_vocab.rows is not None
        rank_logits.append(draft_vocab.rows(hidden))
    # The all-gather of each rank's logits, then the padding columns dropped.
    gathered = torch.cat(rank_logits, dim=-1)
    valid_cols = draft_vocab.valid_cols
    assert valid_cols is not None
    torch.testing.assert_close(gathered[:, valid_cols], hidden @ full_w[draft_ids].t())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="quantized rows need CUDA")
@pytest.mark.parametrize(
    ("quantization", "max_rel_err", "min_top1"),
    [("fp8", 0.05, 0.9), ("nvfp4", 0.15, 0.75)],
)
def test_quantized_rows_track_bf16_rows(
    default_vllm_config, dist_init, quantization, max_rel_err, min_top1
):
    """Listed and dynamic rows in fp8/nvfp4 stay close to the bf16 head."""
    vocab, hidden_size, num_rows = 4000, 512, 256
    head = ParallelLMHead(vocab, hidden_size, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen) * 0.02)
    head = head.to("cuda", torch.bfloat16)
    token_ids = torch.arange(0, vocab, 4)
    hidden = torch.randn(512, hidden_size, device="cuda", dtype=torch.bfloat16)

    def draft(quantization):
        draft_vocab = DraftVocab(LogitsProcessor(vocab), vocab)
        draft_vocab.load_token_map(head, token_ids, num_rows, 64, quantization)
        return draft_vocab.compute_logits(head, hidden).float()

    full = LogitsProcessor(vocab)(head, hidden).float()
    out = draft(quantization)
    kept = out.isfinite()
    assert torch.all(kept.sum(dim=-1) == token_ids.numel() + num_rows)
    assert (out[kept] - full[kept]).norm() / full[kept].norm() < max_rel_err
    top1 = (out.argmax(-1) == draft(None).argmax(-1)).float().mean()
    assert top1 > min_top1
