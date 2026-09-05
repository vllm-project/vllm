# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `is_layer_gptq_quantized`'s handling of fused layers.

These exercise the fused-shard reconciliation logic directly (no GPU, no
model loading), including the case where a fused layer's checkpoint is
missing a shard entirely (e.g. Gemma 4's k_eq_v global-attention layers,
which have no `v_proj` at all).
"""

import pytest

from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    is_layer_gptq_quantized,
)

PREFIX = "model.layers.5.self_attn.qkv_proj"
BASE = "model.layers.5.self_attn"
Q_PROJ = f"{BASE}.q_proj"
K_PROJ = f"{BASE}.k_proj"
V_PROJ = f"{BASE}.v_proj"
FUSED_MAPPING = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}


def test_fused_layer_all_shards_quantized():
    quantized_layers = [Q_PROJ, K_PROJ, V_PROJ]
    assert is_layer_gptq_quantized(PREFIX, quantized_layers, FUSED_MAPPING)


def test_fused_layer_all_shards_unquantized():
    assert not is_layer_gptq_quantized(PREFIX, [], FUSED_MAPPING)


def test_fused_layer_real_mismatch_raises():
    # q_proj and k_proj are quantized, v_proj is present but unquantized.
    quantized_layers = [Q_PROJ, K_PROJ]
    with pytest.raises(ValueError):
        is_layer_gptq_quantized(PREFIX, quantized_layers, FUSED_MAPPING)


def test_fused_layer_real_mismatch_raises_with_modules_in_checkpoint():
    # v_proj is genuinely present in the checkpoint, so a mismatch is
    # still a real mismatch even when modules_in_checkpoint is provided.
    quantized_layers = [Q_PROJ, K_PROJ]
    modules_in_checkpoint = {Q_PROJ, K_PROJ, V_PROJ}
    with pytest.raises(ValueError):
        is_layer_gptq_quantized(
            PREFIX,
            quantized_layers,
            FUSED_MAPPING,
            modules_in_checkpoint=modules_in_checkpoint,
        )


def test_fused_layer_absent_shard_is_skipped():
    # v_proj does not exist in the checkpoint at all (e.g. Gemma 4's
    # k_eq_v layers), so it cannot disagree with q_proj/k_proj.
    quantized_layers = [Q_PROJ, K_PROJ]
    modules_in_checkpoint = {Q_PROJ, K_PROJ}
    assert is_layer_gptq_quantized(
        PREFIX,
        quantized_layers,
        FUSED_MAPPING,
        modules_in_checkpoint=modules_in_checkpoint,
    )


def test_fused_layer_no_shard_in_checkpoint_returns_false():
    quantized_layers = [Q_PROJ, K_PROJ, V_PROJ]
    assert not is_layer_gptq_quantized(
        PREFIX,
        quantized_layers,
        FUSED_MAPPING,
        modules_in_checkpoint=set(),
    )
