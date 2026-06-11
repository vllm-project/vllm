# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for adapter `.pt` classification and splitting.

Covers §4.3 of docs/design/hybrid_lora_tokenformer.md.
"""

import pytest

from vllm.tokenformer.adapter_format import (
    AdapterClassification,
    LoadedAdapter,
    classify_adapter,
    has_lora_keys,
    has_tokenformer_keys,
    load_adapter_from_pt,
    load_adapter_state_dict,
    normalize_lora_key,
    normalize_lora_state_dict,
    split_adapter_state_dict,
)


# --- classification -----------------------------------------------------


def _fake_sd(*keys):
    """Build a state-dict-like dict with sentinel values."""
    return {k: i for i, k in enumerate(keys)}


def test_pure_tokenformer_is_classified_tokenformer():
    sd = _fake_sd(
        "model.layers.0.mlp.tokenformer_k",
        "model.layers.0.mlp.tokenformer_v",
        "model.layers.0.mlp.tokenformer_p",
    )
    c = classify_adapter(sd)
    assert c == AdapterClassification(has_tokenformer=True, has_lora=False)
    assert c.kind == "tokenformer"


def test_pure_lora_is_classified_lora():
    sd = _fake_sd(
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
    )
    c = classify_adapter(sd)
    assert c == AdapterClassification(has_tokenformer=False, has_lora=True)
    assert c.kind == "lora"


def test_hybrid_is_classified_hybrid():
    sd = _fake_sd(
        "model.layers.0.mlp.tokenformer_k",
        "model.layers.0.self_attn.q_proj.lora_A.weight",
    )
    c = classify_adapter(sd)
    assert c.has_tokenformer and c.has_lora
    assert c.kind == "hybrid"


def test_tokenformer_with_base_overrides_is_still_tokenformer():
    # Base-weight overrides (norms, embeddings, lm_head) count as
    # Tokenformer for classification purposes.
    sd = _fake_sd(
        "model.layers.0.mlp.tokenformer_k",
        "model.embed_tokens.weight",
        "lm_head.weight",
    )
    assert classify_adapter(sd).kind == "tokenformer"


def test_empty_state_dict_is_rejected():
    c = classify_adapter({})
    assert c.has_tokenformer is False and c.has_lora is False
    with pytest.raises(ValueError, match="neither tokenformer"):
        _ = c.kind


def test_unrelated_keys_only_is_rejected():
    # Just base weights with no tokenformer or lora markers — malformed.
    sd = _fake_sd("model.embed_tokens.weight", "lm_head.weight")
    c = classify_adapter(sd)
    assert c.has_tokenformer is False and c.has_lora is False
    with pytest.raises(ValueError):
        _ = c.kind


# --- key predicates -----------------------------------------------------


def test_has_tokenformer_keys_matches_leaf_only():
    # The match is on the leaf segment, not substring, so a key that
    # contains "tokenformer_k" mid-path should NOT match.
    assert has_tokenformer_keys(["mlp.tokenformer_k"]) is True
    assert (
        has_tokenformer_keys(["mlp.tokenformer_k.extra"])
        is False
    )


def test_has_lora_keys_requires_delimited_segment():
    # Match requires `.lora_A.` / `.lora_B.` as a path segment —
    # a variable named e.g. `lora_A_config` must not match.
    assert has_lora_keys(["q_proj.lora_A.weight"]) is True
    assert has_lora_keys(["q_proj.lora_B.weight"]) is True
    assert has_lora_keys(["some.lora_A_config"]) is False
    assert has_lora_keys(["prefix_lora_A.weight"]) is False


# --- splitting ----------------------------------------------------------


def test_split_routes_tokenformer_and_base_to_tokenformer_sd():
    sd = {
        "model.layers.0.mlp.tokenformer_k": "tk",
        "model.embed_tokens.weight": "emb",
        "lm_head.weight": "head",
    }
    tk_sd, lora_sd = split_adapter_state_dict(sd)
    assert set(tk_sd) == {
        "model.layers.0.mlp.tokenformer_k",
        "model.embed_tokens.weight",
        "lm_head.weight",
    }
    assert lora_sd == {}


def test_split_routes_lora_to_lora_sd():
    # Input: HF-shaped trainer keys. Output: vLLM module paths after
    # normalize_lora_key runs inside split_adapter_state_dict.
    sd = {
        "model.language_model.layers.0.self_attn.q_proj.lora_A.weight": "A",
        "model.language_model.layers.0.self_attn.q_proj.lora_B.weight": "B",
    }
    tk_sd, lora_sd = split_adapter_state_dict(sd)
    assert tk_sd == {}
    assert set(lora_sd) == {
        "language_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "language_model.model.layers.0.self_attn.q_proj.lora_B.weight",
    }


def test_normalize_swaps_gemma4_language_model_prefix():
    """vLLM's Gemma4 module tree has `language_model.model.layers.*`
    (extra `.model.` nesting). HF has `model.language_model.layers.*`.
    We swap them so LoRA keys land on the right modules.
    """
    trainer_key = (
        "model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight"
    )
    assert normalize_lora_key(trainer_key) == (
        "language_model.model.layers.0.self_attn.q_proj.lora_A.weight"
    )


def test_normalize_strips_top_level_model_prefix_for_vision():
    """Vision tower is at the top level in vLLM — no `model.` prefix.
    Also verifies the `.linear.` wrapper stripping still runs."""
    trainer_key = (
        "model.vision_tower.encoder.layers.0.self_attn.q_proj"
        ".linear.lora_A.default.weight"
    )
    assert normalize_lora_key(trainer_key) == (
        "vision_tower.encoder.layers.0.self_attn.q_proj.lora_A.weight"
    )


def test_normalize_strips_top_level_model_prefix_for_embed_vision():
    trainer_key = (
        "model.embed_vision.embedding_projection.lora_A.default.weight"
    )
    assert normalize_lora_key(trainer_key) == (
        "embed_vision.embedding_projection.lora_A.weight"
    )


def test_normalize_gemma4_mlp_key_from_real_trainer_output():
    """Exact key from the user's training log (2026-04-23)."""
    trainer_key = "model.language_model.layers.0.mlp.gate_proj.lora_A.default.weight"
    assert normalize_lora_key(trainer_key) == (
        "language_model.model.layers.0.mlp.gate_proj.lora_A.weight"
    )


def test_normalize_keeps_model_prefix_for_causal_lm_decoder():
    """Text-only causal LMs (Gemma3ForCausalLM, Qwen2ForCausalLM, ...)
    keep the top-level `model.` prefix in vLLM's module tree
    (`model.layers.<N>...`). Stripping it produces `layers.<N>...`,
    which matches no module, so the adapter loads but silently no-ops
    at activation. Regression for that exact failure."""
    for trainer_key, expected in [
        (
            "model.layers.0.self_attn.q_proj.lora_A.default.weight",
            "model.layers.0.self_attn.q_proj.lora_A.weight",
        ),
        (
            "model.layers.5.mlp.down_proj.lora_B.default.weight",
            "model.layers.5.mlp.down_proj.lora_B.weight",
        ),
    ]:
        assert normalize_lora_key(trainer_key) == expected


def test_split_normalizes_peft_default_segment():
    """`.lora_A.default.weight` and `.lora_B.default.weight` collapse
    to vLLM's expected `.lora_A.weight` / `.lora_B.weight`, combined
    with the model.language_model → language_model.model swap."""
    sd = {
        "model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight": "A",
        "model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight": "B",
    }
    _, lora_sd = split_adapter_state_dict(sd)
    assert set(lora_sd) == {
        "language_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "language_model.model.layers.0.self_attn.q_proj.lora_B.weight",
    }


def test_split_strips_clippable_linear_wrapper():
    """Gemma4ClippableLinear inserts a `.linear.` segment between the
    module and its LoRA weights; the normalizer must remove it.
    vision_tower lives at the top level (no `.model.` swap)."""
    sd = {
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.lora_A.default.weight": "A",
        "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.lora_B.default.weight": "B",
    }
    _, lora_sd = split_adapter_state_dict(sd)
    assert set(lora_sd) == {
        "vision_tower.encoder.layers.0.self_attn.q_proj.lora_A.weight",
        "vision_tower.encoder.layers.0.mlp.gate_proj.lora_B.weight",
    }


def test_normalize_is_idempotent():
    # Already-normalized keys (in vLLM module shape) pass through
    # unchanged.
    for already_normalized in [
        "language_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "vision_tower.encoder.layers.0.mlp.down_proj.lora_B.weight",
        "embed_vision.embedding_projection.lora_A.weight",
    ]:
        assert normalize_lora_key(already_normalized) == already_normalized


def test_normalize_collision_raises():
    # Two training-side keys that normalize to the same target is a
    # bug upstream of us; fail loudly rather than silently dropping
    # one.
    sd = {
        "q_proj.lora_A.default.weight": "A1",
        "q_proj.linear.lora_A.weight": "A2",
    }
    with pytest.raises(ValueError, match="collision"):
        normalize_lora_state_dict(sd)


def test_split_hybrid():
    # LoRA inputs are in HF shape and normalize on the way out.
    # Tokenformer keys don't go through normalize_lora_key.
    sd = {
        "model.layers.0.mlp.tokenformer_k": "tk",
        "model.language_model.layers.0.self_attn.q_proj.lora_A.weight": "A",
        "model.language_model.layers.0.self_attn.q_proj.lora_B.weight": "B",
        "model.embed_tokens.weight": "emb",
    }
    tk_sd, lora_sd = split_adapter_state_dict(sd)
    assert set(tk_sd) == {
        "model.layers.0.mlp.tokenformer_k",
        "model.embed_tokens.weight",
    }
    assert set(lora_sd) == {
        "language_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "language_model.model.layers.0.self_attn.q_proj.lora_B.weight",
    }


def test_split_is_not_destructive():
    sd = {
        "model.layers.0.mlp.tokenformer_k": "tk",
        "model.layers.0.self_attn.q_proj.lora_A.weight": "A",
    }
    sd_copy = dict(sd)
    _ = split_adapter_state_dict(sd)
    assert sd == sd_copy  # input not mutated


# --- .pt loader ---------------------------------------------------------


def test_load_adapter_from_pt_dispatches_to_classify_and_split(monkeypatch, tmp_path):
    """Without torch, mock the low-level checkpoint reader to assert the
    high-level function wires the pure pieces together correctly."""
    fake_sd = {
        "model.layers.0.mlp.tokenformer_k": "tk",
        "model.embed_tokens.weight": "emb",
        "model.language_model.layers.0.self_attn.q_proj.lora_A.weight": "A",
        "model.language_model.layers.0.self_attn.q_proj.lora_B.weight": "B",
    }
    fake_metadata = {"lora_alpha": 64}
    monkeypatch.setattr(
        "vllm.tokenformer.adapter_format._load_adapter_checkpoint",
        lambda *a, **kw: (fake_sd, fake_metadata),
    )
    result = load_adapter_from_pt(tmp_path)
    assert isinstance(result, LoadedAdapter)
    assert result.kind == "hybrid"
    assert set(result.tokenformer_sd) == {
        "model.layers.0.mlp.tokenformer_k",
        "model.embed_tokens.weight",
    }
    assert set(result.lora_sd) == {
        "language_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "language_model.model.layers.0.self_attn.q_proj.lora_B.weight",
    }
    assert result.source_path == tmp_path.resolve()
    assert result.metadata == {"lora_alpha": 64}


def test_load_adapter_from_pt_rejects_unrelated_only(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "vllm.tokenformer.adapter_format._load_adapter_checkpoint",
        lambda *a, **kw: ({"model.embed_tokens.weight": "emb"}, {}),
    )
    with pytest.raises(ValueError, match="neither tokenformer"):
        load_adapter_from_pt(tmp_path)


def test_metadata_defaults_to_empty_dict(monkeypatch, tmp_path):
    """Older .pt files without a `metadata` key still load, with
    metadata={} in the result."""
    fake_sd = {"model.layers.0.mlp.tokenformer_k": "tk"}
    monkeypatch.setattr(
        "vllm.tokenformer.adapter_format._load_adapter_checkpoint",
        lambda *a, **kw: (fake_sd, {}),
    )
    result = load_adapter_from_pt(tmp_path)
    assert result.metadata == {}


def test_load_adapter_metadata_real_pt(tmp_path):
    torch = pytest.importorskip("torch")
    from vllm.tokenformer.adapter_format import load_adapter_metadata

    sd = {"model.layers.0.mlp.tokenformer_k": torch.zeros(2)}
    torch.save(
        {"model_state_dict": sd, "metadata": {"lora_alpha": 32, "use_rslora": True}},
        tmp_path / "a.pt",
    )
    md = load_adapter_metadata(tmp_path)
    assert md == {"lora_alpha": 32, "use_rslora": True}


def test_load_adapter_metadata_missing_returns_empty(tmp_path):
    torch = pytest.importorskip("torch")
    from vllm.tokenformer.adapter_format import load_adapter_metadata

    torch.save({"model_state_dict": {"x": torch.zeros(1)}}, tmp_path / "a.pt")
    assert load_adapter_metadata(tmp_path) == {}


def test_metadata_wrong_type_raises(tmp_path):
    torch = pytest.importorskip("torch")
    torch.save(
        {"model_state_dict": {"x": torch.zeros(1)}, "metadata": "not-a-dict"},
        tmp_path / "a.pt",
    )
    with pytest.raises(ValueError, match="metadata"):
        load_adapter_state_dict(tmp_path)


def test_load_adapter_state_dict_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .pt file"):
        load_adapter_state_dict(tmp_path)


# --- real torch round-trip (skipped if torch not installed) -------------


def _torch_or_skip():
    torch = pytest.importorskip("torch")
    return torch


def test_load_adapter_state_dict_real_pt(tmp_path):
    torch = _torch_or_skip()
    sd = {
        "model.layers.0.mlp.tokenformer_k": torch.zeros(4, 4),
        "model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(4, 8),
    }
    torch.save({"model_state_dict": sd}, tmp_path / "adapter.pt")
    loaded = load_adapter_state_dict(tmp_path)
    assert set(loaded) == set(sd)


def test_load_adapter_from_pt_real_pt_hybrid(tmp_path):
    torch = _torch_or_skip()
    sd = {
        "model.layers.0.mlp.tokenformer_p": torch.zeros(4, 4),
        "model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(8, 4),
        "lm_head.weight": torch.zeros(16, 4),
    }
    torch.save({"model_state_dict": sd}, tmp_path / "adapter.pt")
    result = load_adapter_from_pt(tmp_path)
    assert result.kind == "hybrid"
    assert set(result.tokenformer_sd) == {
        "model.layers.0.mlp.tokenformer_p",
        "lm_head.weight",
    }
    assert set(result.lora_sd) == {
        "language_model.model.layers.0.self_attn.q_proj.lora_B.weight",
    }


def test_load_adapter_rejects_checkpoint_without_model_state_dict(tmp_path):
    torch = _torch_or_skip()
    torch.save({"some_other_key": torch.zeros(2)}, tmp_path / "adapter.pt")
    with pytest.raises(ValueError, match="model_state_dict"):
        load_adapter_state_dict(tmp_path)
