# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.qwen2_moe import (
    _ceil_to_multiple,
    _dense_mlp_padded_intermediate_multiple,
)
from vllm.model_executor.models.qwen3_5 import _is_qwen3_5_critical_weight_loaded


def test_qwen3_5_dense_mlp_tp3_padding_aligns_awq_group_size():
    intermediate_size = 17408
    tp_size = 3

    padded = _ceil_to_multiple(
        intermediate_size,
        _dense_mlp_padded_intermediate_multiple(tp_size),
    )

    assert padded == 17472
    assert padded % tp_size == 0
    assert (padded // tp_size) % 32 == 0


def test_qwen3_5_critical_weight_audit_accepts_packed_weight():
    loaded = {"language_model.model.layers.0.linear_attn.out_proj.weight_packed"}

    assert _is_qwen3_5_critical_weight_loaded(
        "language_model.model.layers.0.linear_attn.out_proj.weight",
        loaded,
    )
    assert not _is_qwen3_5_critical_weight_loaded(
        "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight",
        {"language_model.model.layers.0.linear_attn.in_proj_qkvz.weight_scale"},
    )


def test_qwen3_5_tied_embeddings_do_not_require_separate_lm_head():
    critical_weights = {
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight",
        "language_model.model.layers.0.linear_attn.out_proj.weight",
    }
    loaded = set(critical_weights)

    missing_critical = sorted(
        name
        for name in critical_weights
        if not _is_qwen3_5_critical_weight_loaded(name, loaded)
    )

    assert missing_critical == []
