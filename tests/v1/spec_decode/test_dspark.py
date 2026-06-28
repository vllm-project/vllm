# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.config import SpeculativeConfig
from vllm.models.deepseek_v4.nvidia.dspark import (
    _build_dspark_topk_idxs,
    _read_dspark_num_layers,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import _get_target_layer_ids


def _spec_config(method: str) -> SpeculativeConfig:
    config = object.__new__(SpeculativeConfig)
    config.method = method
    config.draft_model_config = None
    return config


def _compute_dspark_hash(layer_ids: list[int]) -> str:
    config = _spec_config("dspark")
    config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(dspark_target_layer_ids=layer_ids)
    )
    return config.compute_hash()


def test_dspark_compile_hash_uses_target_layer_ids():
    base_hash = _compute_dspark_hash([40, 41, 42])
    same_hash = _compute_dspark_hash([40, 41, 42])
    different_hash = _compute_dspark_hash([39, 40, 41])

    assert base_hash == same_hash
    assert base_hash != different_hash


def test_dspark_does_not_require_eagle_prefix_cache_drop():
    assert _spec_config("dspark").use_eagle()
    assert not _spec_config("dspark").requires_eagle_cache_drop()
    assert _spec_config("eagle3").requires_eagle_cache_drop()


def test_dspark_target_layer_ids_from_config():
    config = _spec_config("dspark")
    config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(dspark_target_layer_ids=[40, 41, 42])
    )

    assert _get_target_layer_ids(config) == (40, 41, 42)


def test_dspark_target_layer_ids_reject_missing_config():
    config = _spec_config("dspark")

    with pytest.raises(RuntimeError, match="requires a draft model config"):
        _get_target_layer_ids(config)


def test_dspark_target_layer_ids_reject_invalid_config():
    config = _spec_config("dspark")
    config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(dspark_target_layer_ids=None)
    )

    with pytest.raises(RuntimeError, match="must define dspark_target_layer_ids"):
        _get_target_layer_ids(config)


def test_read_dspark_num_layers_prefers_inference_config(tmp_path):
    inference_dir = tmp_path / "inference"
    inference_dir.mkdir()
    (inference_dir / "config.json").write_text(
        json.dumps({"n_mtp_layers": 3}),
        encoding="utf-8",
    )

    assert _read_dspark_num_layers(str(tmp_path), default=1) == 3


def test_read_dspark_num_layers_falls_back_to_safetensors_index(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "model-00001.safetensors",
                    "mtp.0.block.weight": "model-00002.safetensors",
                    "mtp.2.block.weight": "model-00003.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    assert _read_dspark_num_layers(str(tmp_path), default=1) == 3


def test_read_dspark_num_layers_uses_default_without_metadata(tmp_path):
    assert _read_dspark_num_layers(str(tmp_path), default=1) == 1


def test_build_dspark_topk_idxs_uses_rolling_target_window_then_draft_block():
    positions = torch.tensor([0, 2], dtype=torch.int64)

    idxs = _build_dspark_topk_idxs(
        window_size=4,
        batch_size=2,
        block_size=3,
        positions=positions,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [
                [0, -1, -1, -1, 4, 5, 6],
                [0, -1, -1, -1, 4, 5, 6],
                [0, -1, -1, -1, 4, 5, 6],
            ],
            [
                [0, 1, 2, -1, 4, 5, 6],
                [0, 1, 2, -1, 4, 5, 6],
                [0, 1, 2, -1, 4, 5, 6],
            ],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(idxs, expected)
