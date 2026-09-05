# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


@pytest.mark.parametrize("draft_embed", ["loaded", "unset"])
def test_drafter_without_own_embedding_gets_the_targets(monkeypatch, draft_embed):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    target_embed = _embed()
    draft_inner = _inner(_embed() if draft_embed == "loaded" else None)
    if draft_embed == "unset":
        draft_inner.embed_tokens = None
    draft = SimpleNamespace(has_own_embed_tokens=False)

    eagle_utils.maybe_share_target_embed(draft, draft_inner, _inner(target_embed))

    assert draft_inner.embed_tokens is target_embed


def test_missing_target_embedding_raises_instead_of_running_on_garbage(monkeypatch):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    draft_inner = _inner(_embed())
    draft = SimpleNamespace(has_own_embed_tokens=False)

    with pytest.raises(RuntimeError, match="needs the target input embedding"):
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


def test_mtp_style_drafter_is_left_alone_under_pp(monkeypatch):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    draft_embed = _embed()
    draft_inner = _inner(draft_embed)

    eagle_utils.maybe_share_target_embed(nn.Module(), draft_inner, _inner(_embed()))

    assert draft_inner.embed_tokens is draft_embed


@pytest.mark.parametrize(
    "method,pp_size,is_last_rank,expected",
    [
        ("eagle", 2, True, True),
        ("eagle3", 2, True, True),
        ("dflash", 2, True, True),
        ("dspark", 2, True, True),
        ("eagle3", 1, True, False),
        ("eagle3", 2, False, False),
        ("mtp", 2, True, False),
        (None, 2, True, False),
    ],
)
def test_target_embedding_provisioning(
    monkeypatch, method, pp_size, is_last_rank, expected
):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        _fake_pp(pp_size, is_last_rank),
        raising=True,
    )
    speculative_config = None if method is None else SimpleNamespace(method=method)
    config = SimpleNamespace(speculative_config=speculative_config)
    assert spec_decode_needs_target_embed(config) is expected
