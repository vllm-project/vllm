# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationConfig, SnapshotConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase


@pytest.mark.parametrize("snapshot_enabled", [False, True])
def test_rotary_cache_is_persistent_only_for_snapshot(snapshot_enabled):
    vllm_config = SimpleNamespace(
        snapshot_config=SnapshotConfig() if snapshot_enabled else None,
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )
    with set_current_vllm_config(vllm_config):
        layer = RotaryEmbeddingBase(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=16,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
        )

    assert ("cos_sin_cache" in layer.state_dict()) is snapshot_enabled
