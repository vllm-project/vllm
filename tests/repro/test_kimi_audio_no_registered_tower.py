# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression test: Kimi-Audio must not register a tower submodule.

V1 multiprocessing uses a strict missing-weights check in DefaultModelLoader.
If KimiAudioTower is registered as a model submodule and contains parameters not
present in the checkpoint, engine startup will fail.

This test prevents reintroducing `self.audio_tower = ...` style registration.
"""

from __future__ import annotations

from pathlib import Path


def test_kimi_audio_does_not_register_audio_tower_submodule() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm"
        / "model_executor"
        / "models"
        / "kimi_audio_asr.py"
    )
    src = path.read_text(encoding="utf-8")

    assert "self.audio_tower" not in src, (
        "KimiAudioTower must not be assigned to `self.audio_tower` as a model "
        "submodule; use runtime-only instantiation to avoid V1 missing-weights "
        "failures."
    )
