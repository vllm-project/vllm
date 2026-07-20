# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Domino pruned-vocab logit pipeline.

Verifies that base logits + Domino correction happen in draft space
before scattering to target space, so pruned-vocab Domino checkpoints
work correctly (draft_vocab_size != target_vocab_size).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

DRAFT_VOCAB = 32
TARGET_VOCAB = 64
HIDDEN_SIZE = 16
GRU_HIDDEN = 8
EMB_DIM = 12


class _ScatterModel(nn.Module):
    """Minimal nn.Module with scatter_logits_to_target from DFlashQwen3ForCausalLM."""

    def __init__(self, draft_vocab, target_vocab):
        super().__init__()
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

        self.config = SimpleNamespace(
            draft_vocab_size=draft_vocab,
            vocab_size=target_vocab,
        )
        # d2t stores offsets: target_id = draft_id + d2t[draft_id].
        # Identity mapping (offset=0) places draft tokens at positions 0..N-1.
        d2t = torch.zeros(draft_vocab, dtype=torch.long)
        if draft_vocab == target_vocab:
            self.draft_id_to_target_id = None
        else:
            self.draft_id_to_target_id = nn.Parameter(d2t, requires_grad=False)

        self.scatter_logits_to_target = (
            DFlashQwen3ForCausalLM.scatter_logits_to_target.__get__(
                self, type(self)
            )
        )


def _make_scatter_model(draft_vocab=DRAFT_VOCAB, target_vocab=TARGET_VOCAB):
    return _ScatterModel(draft_vocab, target_vocab)


class TestScatterLogitsToTarget:
    def test_shape_with_d2t_mapping(self):
        model = _make_scatter_model()
        logits = torch.randn(4, DRAFT_VOCAB)
        result = model.scatter_logits_to_target(logits)
        assert result.shape == (4, TARGET_VOCAB)

    def test_values_scattered_correctly(self):
        model = _make_scatter_model()
        logits = torch.ones(1, DRAFT_VOCAB) * 42.0
        result = model.scatter_logits_to_target(logits)
        # Identity offset (d2t=0): draft[i] → target[i], so first DRAFT_VOCAB
        # positions get 42.0, the rest get -inf.
        assert result[0, :DRAFT_VOCAB].eq(42.0).all()
        assert result[0, DRAFT_VOCAB:].eq(float("-inf")).all()

    def test_noop_when_no_mapping(self):
        model = _make_scatter_model()
        model.draft_id_to_target_id = None
        logits = torch.randn(4, DRAFT_VOCAB)
        result = model.scatter_logits_to_target(logits)
        assert torch.equal(result, logits)

    def test_noncontiguous_mapping(self):
        """d2t offsets that scatter draft tokens to non-contiguous positions."""
        model = _make_scatter_model(draft_vocab=3, target_vocab=8)
        model.draft_id_to_target_id = nn.Parameter(
            torch.tensor([2, 0, 1], dtype=torch.long), requires_grad=False
        )

        logits = torch.tensor([[10.0, 20.0, 30.0]])
        result = model.scatter_logits_to_target(logits)
        assert result.shape == (1, 8)
        assert result[0, 2] == 10.0  # draft[0] → target[0+2]
        assert result[0, 1] == 20.0  # draft[1] → target[1+0]
        assert result[0, 3] == 30.0  # draft[2] → target[2+1]


class TestDominoHeadDraftSpace:
    @pytest.fixture
    def domino_head(self):
        """Standalone DominoHead using plain nn.Linear (no vLLM parallelism)."""
        head = nn.Module()
        head.gru_hidden_dim = GRU_HIDDEN
        head.emb_dim = EMB_DIM
        head.prefix_gru = nn.GRU(
            input_size=HIDDEN_SIZE,
            hidden_size=GRU_HIDDEN,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        head.embed_proj = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + GRU_HIDDEN, EMB_DIM, bias=False),
            nn.SiLU(),
            nn.Linear(EMB_DIM, DRAFT_VOCAB, bias=False),
        )

        from vllm.model_executor.models.qwen3_dflash import DominoHead

        head.compute_logits = DominoHead.compute_logits.__get__(head, type(head))
        return head

    def test_correction_same_space_as_base(self, domino_head):
        batch = 2
        hidden = torch.randn(batch, HIDDEN_SIZE)
        gru_hidden = torch.randn(1, batch, GRU_HIDDEN)
        base_logits = torch.randn(batch, DRAFT_VOCAB)

        result = domino_head.compute_logits(hidden, gru_hidden, base_logits)
        assert result.shape == (batch, DRAFT_VOCAB)

    def test_correction_adds_to_base(self, domino_head):
        batch = 1
        hidden = torch.randn(batch, HIDDEN_SIZE)
        gru_hidden = torch.zeros(1, batch, GRU_HIDDEN)
        base_logits = torch.zeros(batch, DRAFT_VOCAB)

        result = domino_head.compute_logits(hidden, gru_hidden, base_logits)
        assert not result.eq(0.0).all(), "correction should modify base logits"


class TestDominoPrunedVocabPipeline:
    def test_full_pipeline(self):
        """Draft logits → Domino correction → scatter: end-to-end shape check."""
        model = _make_scatter_model()

        draft_logits = torch.randn(4, DRAFT_VOCAB)
        correction = torch.randn(4, DRAFT_VOCAB)
        corrected = draft_logits + correction
        final = model.scatter_logits_to_target(corrected)

        assert final.shape == (4, TARGET_VOCAB)

    def test_argmax_returns_target_space_ids(self):
        """After scatter, argmax should return valid target-space token IDs."""
        model = _make_scatter_model()

        draft_logits = torch.randn(4, DRAFT_VOCAB)
        final = model.scatter_logits_to_target(draft_logits)
        token_ids = final.argmax(dim=-1)

        assert token_ids.shape == (4,)
        assert (token_ids < TARGET_VOCAB).all()
        assert (token_ids >= 0).all()

    @pytest.mark.parametrize("draft_eq_target", [True, False])
    def test_same_tokens_with_and_without_scatter(self, draft_eq_target):
        """When draft==target vocab, scatter is a noop; tokens should match."""
        if draft_eq_target:
            model = _make_scatter_model(draft_vocab=32, target_vocab=32)
        else:
            model = _make_scatter_model(draft_vocab=32, target_vocab=64)
            model.draft_id_to_target_id = nn.Parameter(
                torch.zeros(32, dtype=torch.long), requires_grad=False
            )

        logits = torch.randn(4, 32)
        scattered = model.scatter_logits_to_target(logits)
        tokens = scattered.argmax(dim=-1)
        direct_tokens = logits.argmax(dim=-1)

        assert torch.equal(tokens, direct_tokens)
