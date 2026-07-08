# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SpecDecodeBaseProposer._maybe_share_embeddings.

Regression tests for https://github.com/vllm-project/vllm/issues/47794:
MTP drafts must share the target model's embeddings even when their own
embedding width differs, while EAGLE drafts must not (#43957).
"""

from unittest import mock

import torch
from torch import nn

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

VOCAB_SIZE = 128
TARGET_DIM = 32
DRAFT_DIM = 8


class _Embed(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(VOCAB_SIZE, dim))


class _Wrapper(nn.Module):
    """Mimics the `.model.embed_tokens` layout of target and draft models."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = _Embed(dim)


def _share_embeddings(draft: nn.Module, target: nn.Module) -> None:
    proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    proposer.model = draft
    pp_group = mock.MagicMock()
    pp_group.world_size = 1
    with mock.patch(
        "vllm.v1.spec_decode.llm_base_proposer.get_pp_group",
        return_value=pp_group,
    ):
        proposer._maybe_share_embeddings(target)


def test_mtp_shares_embeddings_when_dims_differ():
    """Gemma4 MTP: the draft's own embed_tokens only populates the tied
    draft-dim lm_head; pre_projection consumes target-width embeddings,
    so sharing is required even though the widths differ."""
    target = _Wrapper(TARGET_DIM)
    draft = _Wrapper(DRAFT_DIM)  # No has_own_embed_tokens -> MTP path.
    _share_embeddings(draft, target)
    assert draft.model.embed_tokens is target.model.embed_tokens


def test_mtp_shares_embeddings_when_dims_match():
    target = _Wrapper(TARGET_DIM)
    draft = _Wrapper(TARGET_DIM)
    _share_embeddings(draft, target)
    assert draft.model.embed_tokens is target.model.embed_tokens


def test_eagle_keeps_own_embeddings_when_dims_differ():
    """EAGLE drafts consume embeddings at their own hidden size, so
    sharing must stay disabled on width mismatch (#43957)."""
    target = _Wrapper(TARGET_DIM)
    draft = _Wrapper(DRAFT_DIM)
    draft.has_own_embed_tokens = False  # EAGLE path, no trained embeddings.
    original_embed = draft.model.embed_tokens
    _share_embeddings(draft, target)
    assert draft.model.embed_tokens is original_embed


def test_eagle_shares_embeddings_when_dims_match():
    target = _Wrapper(TARGET_DIM)
    draft = _Wrapper(TARGET_DIM)
    draft.has_own_embed_tokens = False
    _share_embeddings(draft, target)
    assert draft.model.embed_tokens is target.model.embed_tokens
