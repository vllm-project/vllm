# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input-embedding provisioning for drafters that run on the last PP rank.

Most EAGLE3 checkpoints (e.g. yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) and every
DSpark one ship no ``embed_tokens`` and alias the target's, which under PP lives
on the first stage. Nothing raises when that alias is missing -- the drafter
proposes against uninitialized weights and only acceptance suffers -- so these
rules are asserted directly. CPU only, no distributed init.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.utils import (
    PPMissingLayer,
    spec_decode_needs_target_embed,
)
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

VOCAB, HIDDEN = 32, 8


def _fake_pp(world_size: int, is_last_rank: bool = True):
    return lambda: SimpleNamespace(
        world_size=world_size,
        is_last_rank=is_last_rank,
        is_first_rank=world_size == 1,
    )


def _inner(embed: nn.Module | None) -> nn.Module:
    inner = nn.Module()
    if embed is not None:
        inner.embed_tokens = embed
    return inner


def _embed(fill: float | None = None) -> nn.Embedding:
    embed = nn.Embedding(VOCAB, HIDDEN)
    if fill is not None:
        with torch.no_grad():
            embed.weight.fill_(fill)
    return embed


@pytest.mark.parametrize("pp_size", [1, 2, 4])
def test_drafter_without_own_embedding_gets_the_targets(monkeypatch, pp_size):
    """The case the acceptance-rate regression came from."""
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(pp_size))
    target_embed = _embed()
    draft_inner = _inner(_embed())
    draft = SimpleNamespace(has_own_embed_tokens=False)

    eagle_utils.maybe_share_target_embed(draft, draft_inner, _inner(target_embed))

    assert draft_inner.embed_tokens is target_embed


def test_missing_target_embedding_raises_instead_of_running_on_garbage(monkeypatch):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    draft_inner = _inner(_embed())
    draft = SimpleNamespace(has_own_embed_tokens=False)

    with pytest.raises(RuntimeError, match="nothing to embed"):
        eagle_utils.maybe_share_target_embed(
            draft, draft_inner, _inner(PPMissingLayer())
        )


def test_drafter_with_distinct_weights_keeps_them(monkeypatch):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    draft_embed = _embed(fill=1.0)
    draft_inner = _inner(draft_embed)
    draft = SimpleNamespace(has_own_embed_tokens=True)

    eagle_utils.maybe_share_target_embed(draft, draft_inner, _inner(_embed(fill=2.0)))

    assert draft_inner.embed_tokens is draft_embed


def test_duplicate_of_the_target_is_deduplicated(monkeypatch):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    target_embed = _embed(fill=1.0)
    draft_inner = _inner(_embed(fill=1.0))
    draft = SimpleNamespace(has_own_embed_tokens=True)

    eagle_utils.maybe_share_target_embed(draft, draft_inner, _inner(target_embed))

    assert draft_inner.embed_tokens is target_embed


def test_mtp_style_drafter_is_left_alone_under_pp(monkeypatch):
    """MTP drafts load an embedding from the target checkpoint, and
    has_own_embed_tokens -- the only signal that separates a loaded one from a
    missing one -- is EAGLE-only, so PP must not touch them."""
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    draft_embed = _embed()
    draft_inner = _inner(draft_embed)

    eagle_utils.maybe_share_target_embed(nn.Module(), draft_inner, _inner(_embed()))

    assert draft_inner.embed_tokens is draft_embed


def test_drafter_without_any_embedding_needs_nothing(monkeypatch):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    draft_inner = _inner(None)

    eagle_utils.maybe_share_target_embed(
        SimpleNamespace(has_own_embed_tokens=False),
        draft_inner,
        _inner(PPMissingLayer()),
    )

    assert not hasattr(draft_inner, "embed_tokens")


def test_drafter_awaiting_the_alias_gets_it(monkeypatch):
    """DSpark drafts declare embed_tokens as None until the alias lands."""
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    draft_inner = _inner(None)
    draft_inner.embed_tokens = None
    target_embed = _embed()

    eagle_utils.maybe_share_target_embed(
        SimpleNamespace(has_own_embed_tokens=False), draft_inner, _inner(target_embed)
    )

    assert draft_inner.embed_tokens is target_embed


def test_drafter_awaiting_an_alias_that_never_comes_raises(monkeypatch):
    """The declared-but-unset case must not be mistaken for needing nothing."""
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    draft_inner = _inner(None)
    draft_inner.embed_tokens = None

    with pytest.raises(RuntimeError, match="nothing to embed"):
        eagle_utils.maybe_share_target_embed(
            SimpleNamespace(has_own_embed_tokens=False),
            draft_inner,
            _inner(PPMissingLayer()),
        )


@pytest.mark.parametrize("method", ["eagle", "eagle3", "dflash", "dspark"])
def test_last_stage_provisions_the_target_embedding(monkeypatch, method):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group", _fake_pp(2), raising=True
    )
    config = SimpleNamespace(speculative_config=SimpleNamespace(method=method))
    assert spec_decode_needs_target_embed(config)


@pytest.mark.parametrize(
    "method,pp_size,is_last_rank",
    [
        ("eagle3", 1, True),  # PP=1 keeps the first rank's embedding
        ("eagle3", 2, False),  # only the drafter's stage needs it
        ("mtp", 2, True),  # MTP provisioning is not this feature's business
        ("ngram", 2, True),  # no draft model at all
    ],
)
def test_no_provisioning_elsewhere(monkeypatch, method, pp_size, is_last_rank):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        _fake_pp(pp_size, is_last_rank),
        raising=True,
    )
    config = SimpleNamespace(speculative_config=SimpleNamespace(method=method))
    assert not spec_decode_needs_target_embed(config)


def test_no_provisioning_without_speculative_decoding(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group", _fake_pp(2), raising=True
    )
    assert not spec_decode_needs_target_embed(SimpleNamespace(speculative_config=None))
