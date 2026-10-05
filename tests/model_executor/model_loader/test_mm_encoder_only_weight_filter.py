# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for --mm-encoder-only safetensors shard filtering."""

import json
import os
import tempfile

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.model_executor.model_loader.weight_utils import (
    filter_mm_encoder_only_safetensors_files,
    resolve_mm_encoder_only_lm_prefixes,
)
from vllm.model_executor.models.utils import WeightsMapper


def _write_index(folder: str, weight_map: dict[str, str]) -> None:
    with open(os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def _qwen25_mapper() -> WeightsMapper:
    """Qwen2.5-VL dual layout mapper (nested + flat catch-all)."""
    return WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )


def _qwen3_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )


def _molmo_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "model.vision_backbone.": "vision_backbone.",
            "model.transformer.blocks.": "model.layers.",
            "model.transformer.ln_f.": "model.norm.",
            "model.transformer.mlp.down_proj.": "lm_head.",
        }
    )


def test_keeps_vision_shards_drops_language_only():
    with tempfile.TemporaryDirectory() as folder:
        files = [
            os.path.join(folder, "model-00001-of-000003.safetensors"),
            os.path.join(folder, "model-00002-of-000003.safetensors"),
            os.path.join(folder, "model-00003-of-000003.safetensors"),
        ]
        for path in files:
            open(path, "wb").close()

        _write_index(
            folder,
            {
                "language_model.layers.0.weight": "model-00001-of-000003.safetensors",
                "language_model.layers.1.weight": "model-00002-of-000003.safetensors",
                "vision_tower.blocks.0.weight": "model-00003-of-000003.safetensors",
                "mm_projector.weight": "model-00003-of-000003.safetensors",
            },
        )

        kept = filter_mm_encoder_only_safetensors_files(
            files,
            folder,
            SAFE_WEIGHTS_INDEX_NAME,
            ("language_model.",),
        )
        assert kept == [files[2]]


def test_keeps_mixed_shard_with_vision_and_lm():
    with tempfile.TemporaryDirectory() as folder:
        mixed = os.path.join(folder, "model-00001-of-000001.safetensors")
        open(mixed, "wb").close()
        _write_index(
            folder,
            {
                "language_model.embed.weight": "model-00001-of-000001.safetensors",
                "vision_tower.patch.weight": "model-00001-of-000001.safetensors",
            },
        )

        kept = filter_mm_encoder_only_safetensors_files(
            [mixed],
            folder,
            SAFE_WEIGHTS_INDEX_NAME,
            ("language_model.",),
        )
        assert kept == [mixed]


def test_no_index_returns_unchanged():
    with tempfile.TemporaryDirectory() as folder:
        files = [os.path.join(folder, "model.safetensors")]
        open(files[0], "wb").close()
        kept = filter_mm_encoder_only_safetensors_files(
            files,
            folder,
            SAFE_WEIGHTS_INDEX_NAME,
            ("language_model.",),
        )
        assert kept == files


def test_resolve_returns_module_prefixes_without_hf_hardcode():
    """No global Qwen HF nest list — just vLLM module prefixes."""
    prefixes = resolve_mm_encoder_only_lm_prefixes(["language_model"])
    assert prefixes == ("language_model.",)
    # With Qwen mapper, still module prefixes (keys classified via mapper).
    assert resolve_mm_encoder_only_lm_prefixes(
        ["language_model"], weights_mapper=_qwen25_mapper()
    ) == ("language_model.",)


def test_resolve_empty_language_model_names_fail_closed():
    """Missing/empty ``_language_model_names`` → None (no language_model. default)."""
    assert resolve_mm_encoder_only_lm_prefixes(None) is None
    assert resolve_mm_encoder_only_lm_prefixes([]) is None
    assert resolve_mm_encoder_only_lm_prefixes(()) is None


def test_resolve_skips_filter_for_shared_hf_root():
    """Shared HF root (Molmo-shaped mapper) → fail-closed, not magic name list."""
    assert (
        resolve_mm_encoder_only_lm_prefixes(["model"], weights_mapper=_molmo_mapper())
        is None
    )
    # Without a mapper, identity layout: module prefix is returned as-is.
    assert resolve_mm_encoder_only_lm_prefixes(["model"]) == ("model.",)
    assert resolve_mm_encoder_only_lm_prefixes(["model."]) == ("model.",)


def test_resolve_keeps_llm_prefix_without_model_broadening():
    prefixes = resolve_mm_encoder_only_lm_prefixes(["llm"])
    assert prefixes == ("llm.",)


def test_qwen3_nested_index_skips_lm_keeps_visual():
    with tempfile.TemporaryDirectory() as folder:
        files = [
            os.path.join(folder, "model-00001-of-000002.safetensors"),
            os.path.join(folder, "model-00002-of-000002.safetensors"),
        ]
        for path in files:
            open(path, "wb").close()
        _write_index(
            folder,
            {
                "model.language_model.layers.0.weight": (
                    "model-00001-of-000002.safetensors"
                ),
                "lm_head.weight": "model-00001-of-000002.safetensors",
                "model.visual.blocks.0.weight": "model-00002-of-000002.safetensors",
            },
        )
        prefixes = resolve_mm_encoder_only_lm_prefixes(
            ["language_model"], weights_mapper=_qwen3_mapper()
        )
        assert prefixes is not None
        kept = filter_mm_encoder_only_safetensors_files(
            files,
            folder,
            SAFE_WEIGHTS_INDEX_NAME,
            prefixes,
            weights_mapper=_qwen3_mapper(),
        )
        assert kept == [files[1]]


def test_qwen25_flat_index_skips_lm_keeps_visual():
    with tempfile.TemporaryDirectory() as folder:
        files = [
            os.path.join(folder, "model-00001-of-000002.safetensors"),
            os.path.join(folder, "model-00002-of-000002.safetensors"),
        ]
        for path in files:
            open(path, "wb").close()
        _write_index(
            folder,
            {
                "model.layers.0.self_attn.q_proj.weight": (
                    "model-00001-of-000002.safetensors"
                ),
                "model.embed_tokens.weight": "model-00001-of-000002.safetensors",
                "model.norm.weight": "model-00001-of-000002.safetensors",
                "lm_head.weight": "model-00001-of-000002.safetensors",
                "visual.blocks.0.weight": "model-00002-of-000002.safetensors",
            },
        )
        prefixes = resolve_mm_encoder_only_lm_prefixes(
            ["language_model"], weights_mapper=_qwen25_mapper()
        )
        assert prefixes is not None
        kept = filter_mm_encoder_only_safetensors_files(
            files,
            folder,
            SAFE_WEIGHTS_INDEX_NAME,
            prefixes,
            weights_mapper=_qwen25_mapper(),
        )
        assert kept == [files[1]]


def test_molmo_shaped_index_not_filtered_when_deny_disabled():
    """Shared-root LM attr → resolve None; callers leave the file list alone."""
    assert (
        resolve_mm_encoder_only_lm_prefixes(["model"], weights_mapper=_molmo_mapper())
        is None
    )
    with tempfile.TemporaryDirectory() as folder:
        files = [
            os.path.join(folder, "vision.safetensors"),
            os.path.join(folder, "lm.safetensors"),
        ]
        for path in files:
            open(path, "wb").close()
        _write_index(
            folder,
            {
                "model.vision_backbone.patch.weight": "vision.safetensors",
                "model.transformer.blocks.0.weight": "lm.safetensors",
            },
        )
        # Document why raw HF startswith("model.") is unsafe without fail-closed:
        # whole-shard deny would also drop vision.
        if_applied_naively = filter_mm_encoder_only_safetensors_files(
            files, folder, SAFE_WEIGHTS_INDEX_NAME, ("model.",)
        )
        assert if_applied_naively == []
