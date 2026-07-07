# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``SpecDecodeBaseProposer._maybe_share_embeddings``.

These exercise the embedding-sharing decision in isolation (no real model
load) by driving the method with light-weight fake modules.

Regression coverage for the interaction between two behaviors:

* PR #43957 added an embedding-*width* guard so that EAGLE draft models that
  ship their own differently-sized ``embed_tokens`` keep it instead of
  (incorrectly) sharing the target's.
* Issue #47794: that guard must NOT apply to MTP draft models, whose
  projection is built for the *target* embedding width and therefore must
  always share the target embedding regardless of the draft checkpoint's own
  ``embed_tokens`` width (e.g. Gemma4 MTP: draft 1024 vs target 2816).
"""

import types

import torch
import torch.nn as nn

from vllm.v1.spec_decode import llm_base_proposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

VOCAB = 8


class _Embed(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        # weight shape is (vocab, hidden); the method compares shape[-1].
        self.weight = nn.Parameter(torch.zeros(VOCAB, width))


class _Inner(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.embed_tokens = _Embed(width)


class _Model(nn.Module):
    """A stand-in language model exposing ``.model.embed_tokens``."""

    def __init__(self, width: int, has_own_embed_tokens: bool | None = None):
        super().__init__()
        self.model = _Inner(width)
        # Only EAGLE drafts define this attribute; MTP drafts must not.
        if has_own_embed_tokens is not None:
            self.has_own_embed_tokens = has_own_embed_tokens


def _call(monkeypatch, draft: _Model, target: _Model) -> None:
    monkeypatch.setattr(
        llm_base_proposer,
        "get_pp_group",
        lambda: types.SimpleNamespace(world_size=1),
    )
    fake_self = types.SimpleNamespace(model=draft)
    SpecDecodeBaseProposer._maybe_share_embeddings(fake_self, target)


def test_mtp_shares_even_with_different_embed_width(monkeypatch):
    # Regression test for #47794: an MTP draft (no ``has_own_embed_tokens``)
    # with a narrower embedding than the target must still share the target's
    # embedding, otherwise the concat width is wrong and init crashes.
    target = _Model(width=2816)
    draft = _Model(width=1024)  # MTP: no has_own_embed_tokens attribute

    _call(monkeypatch, draft, target)

    assert draft.model.embed_tokens is target.model.embed_tokens


def test_eagle_keeps_separate_embed_when_width_differs(monkeypatch):
    # Preserves PR #43957: an EAGLE draft without its own checkpoint embedding
    # but a different embedding width must NOT share the target's embedding.
    target = _Model(width=2816)
    draft = _Model(width=1024, has_own_embed_tokens=False)
    own_embed = draft.model.embed_tokens

    _call(monkeypatch, draft, target)

    assert draft.model.embed_tokens is own_embed
    assert draft.model.embed_tokens is not target.model.embed_tokens


def test_eagle_shares_when_width_matches(monkeypatch):
    # Sanity: EAGLE draft with matching width shares the target embedding.
    target = _Model(width=2816)
    draft = _Model(width=2816, has_own_embed_tokens=False)

    _call(monkeypatch, draft, target)

    assert draft.model.embed_tokens is target.model.embed_tokens
