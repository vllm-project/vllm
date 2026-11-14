# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dense-MTP-head detection for pack-quantized checkpoints (#58807).

The whole fix hinges on reading the checkpoint's actual tensor names, so the
tests pin the naming contract: dense means plain `mtp.*.weight` keys with no
quantization artifact suffix anywhere under `mtp.`.
"""

import json
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.qwen3_5_mtp import (
    _checkpoint_ships_dense_mtp,
    _mtp_weight_names,
)

pytestmark = pytest.mark.cpu_test


def _model_config_with_index(monkeypatch, names):
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_5_mtp._mtp_weight_names",
        lambda _config: names,
    )
    return SimpleNamespace()


@pytest.mark.parametrize(
    "names,expected",
    [
        # The reporter's checkpoint: every mtp.* tensor dense.
        (["mtp.fc.weight", "mtp.layers.0.mlp.down_proj.weight"], True),
        # Biases are dense too and must not count as artifacts.
        (["mtp.fc.weight", "mtp.fc.bias"], True),
        # A properly pack-quantized MTP head keeps the target quantization.
        (["mtp.fc.weight_packed", "mtp.fc.weight_scale"], False),
        # Channel-wise quantized MTP head: dense name plus scale companion.
        (["mtp.fc.weight", "mtp.fc.weight_scale"], False),
        # No MTP head in the checkpoint at all.
        (["model.layers.0.self_attn.q_proj.weight"], False),
        # Layout unknown (no index): keep current behavior.
        (None, False),
        ([], False),
    ],
)
def test_dense_mtp_detection(monkeypatch, names, expected):
    config = _model_config_with_index(monkeypatch, names)
    assert _checkpoint_ships_dense_mtp(config) is expected


def test_weight_names_read_from_local_index(tmp_path, monkeypatch):
    index = {
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight_packed": "model-00001.safetensors",
            "mtp.fc.weight": "model-mtp-bf16.safetensors",
            "mtp.layers.0.mlp.down_proj.weight": "model-mtp-bf16.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    config = SimpleNamespace(model=str(tmp_path), revision=None)

    names = _mtp_weight_names(config)

    assert names == ["mtp.fc.weight", "mtp.layers.0.mlp.down_proj.weight"]


def test_weight_names_none_without_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_5_mtp.huggingface_hub.hf_hub_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    config = SimpleNamespace(model=str(tmp_path), revision=None)

    assert _mtp_weight_names(config) is None
