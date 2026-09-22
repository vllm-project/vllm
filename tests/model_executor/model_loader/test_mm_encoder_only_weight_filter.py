# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for --mm-encoder-only safetensors shard filtering."""

import json
import os
import tempfile

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.model_executor.model_loader.weight_utils import (
    filter_mm_encoder_only_safetensors_files,
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
