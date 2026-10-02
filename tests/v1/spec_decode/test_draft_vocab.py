# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the reduced draft vocabulary (speculative_config.draft_token_map).

The drafter's shared lm_head is restricted to a token subset. Greedy drafting
must equal the argmax of the full logits restricted to that subset, and
probabilistic drafting must give a distribution over the full vocabulary with
zero mass outside the subset, so rejection sampling stays exact.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedConfig

from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.v1.spec_decode.draft_vocab import (
    DraftVocab,
    build_draft_token_ids,
    load_draft_token_map,
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
@pytest.mark.parametrize("fmt", ["pt_list", "pt_tensor", "json", "txt"])
def test_load_draft_token_map_formats(tmp_path, fmt):
    ids = [5, 1, 9, 1]
    if fmt == "pt_list":
        path = tmp_path / "map.pt"
        torch.save(ids, path)
    elif fmt == "pt_tensor":
        path = tmp_path / "map.pt"
        torch.save(torch.tensor(ids), path)
    elif fmt == "json":
        path = tmp_path / "map.json"
        path.write_text(json.dumps(ids))
    else:
        path = tmp_path / "map.txt"
        path.write_text("5\n1, 9\n 1\n")
    assert load_draft_token_map(str(path)) == ids


@pytest.mark.cpu_test
def test_load_draft_token_map_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_draft_token_map("missing.txt")


@pytest.mark.cpu_test
def test_build_draft_token_ids_dedupes_sorts_and_adds_specials():
    ids = build_draft_token_ids([9, 2, 9, 4], vocab_size=10, always_include=[7, 2])
    assert ids.tolist() == [2, 4, 7, 9]
    assert ids.dtype == torch.int64


@pytest.mark.cpu_test
@pytest.mark.parametrize("bad", [[-1], [10], []])
def test_build_draft_token_ids_rejects_invalid(bad):
    with pytest.raises(ValueError):
        build_draft_token_ids(bad, vocab_size=10, always_include=[])


@pytest.mark.cpu_test
def test_install_replaces_only_shared_heads(lm_head):
    embed = VocabParallelEmbedding(VOCAB, HIDDEN, disable_tp=True)
    drafter = _Drafter(lm_head, embed)
    dv = DraftVocab(torch.tensor(DRAFT_IDS), lm_head, torch.float32)
    dv.install(drafter, lm_head)
    assert drafter.lm_head is dv.head
    assert drafter.shared_head.head is dv.head
    assert drafter.embed_tokens is embed
    assert dv.head.weight.shape == (len(DRAFT_IDS), HIDDEN)


@pytest.mark.cpu_test
def test_install_requires_shared_head(lm_head):
    own_head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    drafter = _Drafter(own_head, nn.Identity())
    dv = DraftVocab(torch.tensor(DRAFT_IDS), lm_head, torch.float32)
    with pytest.raises(ValueError, match="shares the target"):
        dv.install(drafter, lm_head)


@pytest.mark.cpu_test
def test_greedy_draft_matches_restricted_full_argmax(lm_head, logits_processor):
    ids = torch.tensor(DRAFT_IDS)
    hidden = _hidden()
    full = logits_processor(lm_head, hidden)
    expected = ids[full[:, ids].argmax(dim=-1)]

    drafter = _Drafter(lm_head, nn.Identity())
    dv = DraftVocab(ids, lm_head, torch.float32)
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
    dv = DraftVocab(ids, lm_head, torch.float32)
    dv.install(drafter, lm_head)

    buf = torch.full((16, VOCAB), float("-inf"))
    reduced = dv.restrict(logits_processor(drafter.lm_head, hidden))
    for _ in range(2):  # The buffer is reused across steps.
        logits = dv.scatter(reduced, buf)
    probs = logits.softmax(dim=-1)

    outside = torch.ones(VOCAB, dtype=torch.bool)
    outside[ids] = False
    assert torch.all(probs[:, outside] == 0)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(hidden.shape[0]))
    full = logits_processor(lm_head, hidden)
    torch.testing.assert_close(probs[:, ids], full[:, ids].softmax(dim=-1))


class _Int8RowwiseMethod:
    """Weight-only int8 lm_head with per-row scales."""

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight_q.to(x.dtype) * layer.weight_s, bias)


@pytest.mark.cpu_test
def test_quantized_head_is_materialized(lm_head, logits_processor):
    w = lm_head.weight.data
    scale = w.abs().amax(dim=1, keepdim=True) / 127
    lm_head.weight_q = torch.round(w / scale).to(torch.int8)
    lm_head.weight_s = scale
    lm_head.quant_method = _Int8RowwiseMethod()

    ids = torch.tensor(DRAFT_IDS)
    dv = DraftVocab(ids, lm_head, torch.float32)
    expected_w = lm_head.weight_q[ids].float() * scale[ids]
    torch.testing.assert_close(dv.head.weight, expected_w)

    hidden = _hidden()
    full = logits_processor(lm_head, hidden)
    logits = logits_processor(dv.head, hidden)
    assert torch.equal(dv.target_ids[logits.argmax(-1)], ids[full[:, ids].argmax(-1)])


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
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        self.quant_method = UnquantizedEmbeddingMethod()


@pytest.mark.cpu_test
def test_tensor_parallel_layout_pads_ranks(default_vllm_config):
    # Rank 0 owns ids [0, 128), rank 1 owns [128, 200): 5 vs 3 draft rows.
    full_w = torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(2))
    ids = torch.tensor(DRAFT_IDS)
    ranks = [DraftVocab(ids, _FakeTPHead(full_w, r, 2), torch.float32) for r in (0, 1)]
    dv = ranks[0]
    assert ranks[1].head.shard_indices.num_org_vocab_padding == 2
    assert dv.col_to_target[dv.valid_cols].tolist() == DRAFT_IDS

    hidden = _hidden()
    # Emulate the TP all-gather of each rank's reduced logits.
    gathered = torch.cat([hidden @ r.head.weight.t() for r in ranks], dim=-1)
    expected = ids[(hidden @ full_w[ids].t()).argmax(-1)]
    assert torch.equal(dv.target_ids[dv.restrict(gathered).argmax(-1)], expected)


def _speculative_config(model_type: str, method: str) -> SpeculativeConfig:
    hf_config = SpeculativeConfig.hf_config_override(
        PreTrainedConfig(
            architectures=["Draft"], model_type=model_type, num_hidden_layers=2
        )
    )
    draft = MagicMock(
        hf_config=hf_config, architectures=hf_config.architectures, max_model_len=128
    )
    draft.registry.inspect_model_cls.return_value = (None, "Draft")
    target = MagicMock(max_model_len=128, quantization=None, hf_overrides={})
    target.hf_config.model_type = model_type
    with patch("vllm.config.speculative.ModelConfig", return_value=draft):
        return SpeculativeConfig(
            model="draft",
            method=method,
            num_speculative_tokens=1,
            draft_token_map="map.txt",
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
def test_config_accepts_mtp():
    assert _speculative_config("qwen3_5_mtp", "mtp").draft_token_map == "map.txt"


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("model_type", "method"),
    [("gemma4_mtp", "mtp"), ("llama", "draft_model")],
)
def test_config_rejects_drafters_with_own_head(model_type, method):
    with pytest.raises(ValueError, match="draft_token_map"):
        _speculative_config(model_type, method)
