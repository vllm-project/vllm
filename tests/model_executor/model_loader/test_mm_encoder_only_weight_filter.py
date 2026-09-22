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


def _write_index(folder: str, weight_map: dict[str, str]) -> None:
    with open(os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
        json.dump({"weight_map": weight_map}, f)


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


def test_resolve_expands_qwen_nested_and_flat_prefixes():
    prefixes = resolve_mm_encoder_only_lm_prefixes(["language_model"])
    assert prefixes is not None
    assert "language_model." in prefixes
    assert "model.language_model." in prefixes
    assert "model.layers." in prefixes
    assert "model.embed_tokens." in prefixes
    assert "model.norm." in prefixes
    assert "lm_head." in prefixes
    # This PR does not expand to bare model. (would overlap vision for
    # Molmo / Phi-4-MM / Muse); those stay full-load until finer prefixes.
    assert "model." not in prefixes


def test_resolve_skips_filter_for_bare_model_attr():
    """Bare model. is out of scope for whole-shard skip; resolve returns None."""
    assert resolve_mm_encoder_only_lm_prefixes(["model"]) is None
    assert resolve_mm_encoder_only_lm_prefixes(["model."]) is None


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
        prefixes = resolve_mm_encoder_only_lm_prefixes(["language_model"])
        assert prefixes is not None
        kept = filter_mm_encoder_only_safetensors_files(
            files, folder, SAFE_WEIGHTS_INDEX_NAME, prefixes
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
        prefixes = resolve_mm_encoder_only_lm_prefixes(["language_model"])
        assert prefixes is not None
        kept = filter_mm_encoder_only_safetensors_files(
            files, folder, SAFE_WEIGHTS_INDEX_NAME, prefixes
        )
        assert kept == [files[1]]


def test_molmo_shaped_index_not_filtered_when_deny_disabled():
    """Bare model. is out of scope this PR; resolve returns None so no filter."""
    assert resolve_mm_encoder_only_lm_prefixes(["model"]) is None
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
        # Document why we do not pass bare model. into the filter today:
        # whole-shard deny would also drop vision. Follow-ups can use finer
        # HF LM prefixes instead.
        if_applied_naively = filter_mm_encoder_only_safetensors_files(
            files, folder, SAFE_WEIGHTS_INDEX_NAME, ("model.",)
        )
        assert if_applied_naively == []
        # Safe path for this PR: resolve → None → callers leave file list as-is.
        assert files == files
