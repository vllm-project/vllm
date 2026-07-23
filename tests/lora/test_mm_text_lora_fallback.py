# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for WeightsMapper.with_mm_text_lora_fallback.

Protects the guard that skips gemma4-shaped mappers
(``model.language_model.`` → ``model.``) and models that already carry a bare
``model.`` / ``model`` catch-all. A naive ``"model.language_model." in
prefixes`` check would append the fallback to gemma4 and, because
``_map_name`` applies prefix rules sequentially without breaking, corrupt
``model.language_model.layers...`` → ``model.layers...`` →
``language_model.model.layers...``.
"""

from __future__ import annotations

from vllm.model_executor.models.utils import WeightsMapper

TEXT_LORA_PROBE = "model.layers.0.self_attn.q_proj"
BASE_LM_PROBE = "model.language_model.layers.0.self_attn.q_proj.weight"
BASE_VISUAL_PROBE = "model.visual.blocks.0.attn.qkv.weight"


def _map(mapper: WeightsMapper, key: str) -> str | None:
    return mapper._map_name(key)


def test_gemma4_shaped_mapper_is_noop():
    """gemma4 mounts LM at self.model; destination is not language_model.*."""
    mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.",
        }
    )
    out = mapper.with_mm_text_lora_fallback()
    assert out is mapper
    assert "model." not in out.orig_to_new_prefix
    # Text-LoRA key already matches the text stack path under model.*.
    assert _map(out, TEXT_LORA_PROBE) == TEXT_LORA_PROBE
    # Existing specific rule must not be double-mapped by a catch-all.
    assert _map(out, BASE_LM_PROBE) == "model.layers.0.self_attn.q_proj.weight"


def test_naive_catchall_would_corrupt_gemma4_shaped_base_keys():
    """Document the double-map hazard the destination guard prevents."""
    prefixes = {
        "model.language_model.": "model.",
        # Naive append without destination check — unsafe.
        "model.": "language_model.model.",
    }
    mapper = WeightsMapper(orig_to_new_prefix=prefixes)
    # First rule rewrites to model.layers..., then catch-all rewrites again.
    assert (
        _map(mapper, BASE_LM_PROBE)
        == "language_model.model.layers.0.self_attn.q_proj.weight"
    )


def test_already_has_model_dot_catchall_is_noop():
    """qwen2_vl-shaped: bare model. already present."""
    mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )
    out = mapper.with_mm_text_lora_fallback()
    assert out is mapper
    assert _map(out, TEXT_LORA_PROBE) == "language_model.model.layers.0.self_attn.q_proj"


def test_already_has_model_catchall_without_dot_is_noop():
    """gemma4_mm-shaped: bare 'model' (no trailing dot) already present."""
    mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "model": "language_model.model",
        }
    )
    out = mapper.with_mm_text_lora_fallback()
    assert out is mapper
    assert _map(out, TEXT_LORA_PROBE) == "language_model.model.layers.0.self_attn.q_proj"


def test_qwen3_vl_shaped_bug_gets_fallback():
    """Missing bare catch-all + language_model destination → append last."""
    mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )
    out = mapper.with_mm_text_lora_fallback()
    assert out is not mapper
    assert out.orig_to_new_prefix["model."] == "language_model.model."
    assert list(out.orig_to_new_prefix.keys())[-1] == "model."
    assert _map(mapper, TEXT_LORA_PROBE) == TEXT_LORA_PROBE
    assert _map(out, TEXT_LORA_PROBE) == (
        "language_model.model.layers.0.self_attn.q_proj"
    )
    # Base keys unchanged vs pre-fallback mapper.
    assert _map(out, BASE_LM_PROBE) == _map(mapper, BASE_LM_PROBE)
    assert _map(out, BASE_VISUAL_PROBE) == _map(mapper, BASE_VISUAL_PROBE)
    assert _map(out, BASE_LM_PROBE) == (
        "language_model.model.layers.0.self_attn.q_proj.weight"
    )
    assert _map(out, BASE_VISUAL_PROBE) == "visual.blocks.0.attn.qkv.weight"
