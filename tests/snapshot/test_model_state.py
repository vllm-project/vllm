# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import CompilationConfig, SnapshotConfig, set_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase


@pytest.mark.parametrize("snapshot_enabled", [False, True])
def test_mla_derived_weights_are_persistent_only_for_snapshot(snapshot_enabled):
    layer = object.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_b_proj = object()
    layer.kv_lora_rank = 2
    layer.num_heads = 1
    layer.qk_nope_head_dim = 2
    layer.v_head_dim = 2
    layer.is_aiter_triton_fp4_bmm_enabled = False
    layer.is_aiter_triton_fp8_bmm_enabled = False
    layer.quant_config = None
    layer.layer_name = "model.layers.0.self_attn"
    layer._vllm_config = SimpleNamespace(
        snapshot_config=SnapshotConfig() if snapshot_enabled else None
    )

    weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    with (
        patch(
            "vllm.model_executor.layers.attention.mla_attention."
            "get_and_maybe_dequant_weights",
            return_value=weight,
        ),
        patch(
            "vllm.model_executor.layers.attention.mla_attention."
            "set_default_quant_scales"
        ),
    ):
        layer.process_weights_after_loading(torch.float32)

    state = layer.state_dict()
    assert ("W_UV" in state) is snapshot_enabled
    assert ("W_UK_T" in state) is snapshot_enabled


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
