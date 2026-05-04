# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from create_dummy_lora import create_dummy_lora  # noqa: E402


def _write_model(tmp_path: Path, cfg: dict) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_create_dummy_lora_cohere2moe(tmp_path):
    model_dir = _write_model(
        tmp_path,
        {
            "model_type": "cohere2moe",
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
        },
    )
    out = tmp_path / "lora"
    create_dummy_lora(str(model_dir), str(out), lora_rank=4)
    assert (out / "adapter_config.json").is_file()
    assert (out / "adapter_model.safetensors").is_file()


def test_create_dummy_lora_cohere2_vision_uses_text_config(tmp_path):
    model_dir = _write_model(
        tmp_path,
        {
            "model_type": "cohere2_vision",
            "text_config": {
                "model_type": "cohere2moe",
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
            },
        },
    )
    out = tmp_path / "lora"
    create_dummy_lora(str(model_dir), str(out), lora_rank=4)
    assert (out / "adapter_config.json").is_file()
    assert (out / "adapter_model.safetensors").is_file()
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["target_modules"] == ["k_proj", "o_proj", "q_proj", "v_proj"]
