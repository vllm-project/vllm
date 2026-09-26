# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.utils import (
    PPMissingLayer,
    spec_decode_needs_target_embed,
)
from vllm.platforms import current_platform
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


@pytest.mark.parametrize(
    "pp_size,is_last_rank,expected",
    [(2, True, True), (2, False, False), (1, True, False)],
)
def test_mtp_target_embedding_requires_model_opt_in(
    monkeypatch, pp_size, is_last_rank, expected
):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        _fake_pp(pp_size, is_last_rank),
    )
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="mtp"))
    assert spec_decode_needs_target_embed(config, include_mtp=True) is expected
    assert not spec_decode_needs_target_embed(config)


@pytest.fixture
def m3_modules():
    if not current_platform.is_cuda_alike():
        pytest.skip("M3 model imports require CUDA or ROCm dependencies")
    backend = "amd" if current_platform.is_rocm() else "nvidia"
    root = f"vllm.models.minimax_m3.{backend}"
    return importlib.import_module(f"{root}.mtp"), importlib.import_module(
        f"{root}.model"
    )


@pytest.fixture
def m3_config():
    hf_config = SimpleNamespace(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_local_experts=0,
        num_hidden_layers=0,
        rms_norm_eps=1e-5,
    )
    return SimpleNamespace(
        use_v2_model_runner=True,
        load_config=SimpleNamespace(load_format="auto"),
        model_config=SimpleNamespace(hf_text_config=hf_config),
        speculative_config=SimpleNamespace(
            method="mtp", draft_model_config=SimpleNamespace(hf_config=hf_config)
        ),
        quant_config=None,
        cache_config=None,
    )


@pytest.fixture
def make_m3_draft(monkeypatch, m3_modules, m3_config):
    mtp, _ = m3_modules
    inner = _inner(_embed())
    inner.num_mtp_layers = 0
    monkeypatch.setattr(mtp, "MiniMaxM3MultiTokenPredictor", lambda **_: inner)
    monkeypatch.setattr(mtp, "ParallelLMHead", lambda *_, **kw: _embed())
    monkeypatch.setattr(mtp, "LogitsProcessor", lambda *_: None)
    monkeypatch.setattr(
        mtp, "fused_moe_make_expert_params_mapping", lambda *_, **kw: []
    )
    monkeypatch.setattr(mtp, "get_pp_group", _fake_pp(2))
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    return lambda: mtp.MiniMaxM3MTP(vllm_config=m3_config)


@pytest.mark.parametrize(
    "key",
    [None, "model.embed_tokens.weight", "language_model.model.embed_tokens.weight"],
)
def test_m3_checkpoint_embedding_ownership(make_m3_draft, key):
    """Actual M3 loading determines sharing and survives an unrelated reload."""
    draft = make_m3_draft()
    assert not draft.has_own_embed_tokens
    weights = [] if key is None else [(key, torch.ones(VOCAB, HIDDEN))]
    loaded = draft.load_weights(weights)
    assert ("model.embed_tokens.weight" in loaded) is (key is not None)
    draft.load_weights([("lm_head.weight", torch.zeros(VOCAB, HIDDEN))])
    assert draft.has_own_embed_tokens is (key is not None)
    target = _inner(_embed(fill=2.0))
    original = draft.model.embed_tokens
    eagle_utils.maybe_share_target_embed(draft, draft.model, target)
    if key is None:
        assert draft.model.embed_tokens is target.embed_tokens
    else:
        assert draft.model.embed_tokens is original
        torch.testing.assert_close(original.weight, torch.ones(VOCAB, HIDDEN))


@pytest.mark.parametrize(
    "load_format", ["dummy", "sharded_state", "runai_streamer_sharded"]
)
def test_m3_direct_loader_embedding_ownership(make_m3_draft, m3_config, load_format):
    """Complete sharded state keeps distinct weights without load_weights."""
    m3_config.load_config.load_format = load_format
    draft = make_m3_draft()
    draft.load_state_dict(
        {key: torch.ones_like(value) for key, value in draft.state_dict().items()}
    )
    target = _inner(_embed(fill=2.0))
    original = draft.model.embed_tokens
    eagle_utils.maybe_share_target_embed(draft, draft.model, target)
    assert draft.model.embed_tokens is (
        target.embed_tokens if load_format == "dummy" else original
    )


@pytest.mark.parametrize("use_v2,pp_size", [(True, 1), (False, 1), (False, 2)])
def test_m3_preserves_legacy_embedding_sharing(
    monkeypatch, m3_modules, m3_config, make_m3_draft, use_v2, pp_size
):
    mtp, _ = m3_modules
    m3_config.use_v2_model_runner = use_v2
    monkeypatch.setattr(mtp, "get_pp_group", _fake_pp(pp_size))
    draft = make_m3_draft()
    draft.load_weights([("model.embed_tokens.weight", torch.ones(VOCAB, HIDDEN))])
    assert not hasattr(draft, "has_own_embed_tokens")


@pytest.mark.parametrize(
    "rank,use_v2,method,expected",
    [
        (0, True, "mtp", True),
        (1, True, "mtp", False),
        (2, True, "mtp", True),
        (2, False, "mtp", False),
        (2, True, None, False),
    ],
)
def test_m3_target_allocates_embedding_on_required_stage(
    monkeypatch, m3_modules, m3_config, rank, use_v2, method, expected
):
    _, model = m3_modules
    pp = lambda: SimpleNamespace(
        world_size=3, is_first_rank=rank == 0, is_last_rank=rank == 2
    )
    monkeypatch.setattr(model, "get_pp_group", pp)
    monkeypatch.setattr("vllm.distributed.parallel_state.get_pp_group", pp)
    monkeypatch.setattr(model, "VocabParallelEmbedding", lambda *_, **kw: _embed())
    monkeypatch.setattr(model, "make_layers", lambda *_, **kw: (0, 0, nn.ModuleList()))
    monkeypatch.setattr(model, "MiniMAXGemmaRMSNorm", lambda *_, **kw: nn.Identity())
    m3_config.use_v2_model_runner = use_v2
    if method is None:
        m3_config.speculative_config = None
    target = model.MiniMaxM3Model(vllm_config=m3_config)
    assert isinstance(target.embed_tokens, nn.Embedding) is expected
