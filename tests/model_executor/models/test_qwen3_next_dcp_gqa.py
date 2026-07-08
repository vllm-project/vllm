# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.qwen3_next import (
    _should_replicate_full_attention_heads,
)


def test_dcp_replicates_overlapping_gqa_full_attention_heads():
    should_replicate, use_overlapping_gqa = _should_replicate_full_attention_heads(
        total_num_heads=24,
        total_num_kv_heads=4,
        tp_size=3,
        dcp_size=3,
        enable_dcp_replicated_full_attention=True,
    )

    assert use_overlapping_gqa
    assert should_replicate


def test_overlapping_gqa_without_dcp_keeps_tp_partition():
    should_replicate, use_overlapping_gqa = _should_replicate_full_attention_heads(
        total_num_heads=24,
        total_num_kv_heads=4,
        tp_size=3,
        dcp_size=1,
        enable_dcp_replicated_full_attention=True,
    )

    assert use_overlapping_gqa
    assert not should_replicate


def test_uneven_q_heads_still_replicate_without_dcp():
    should_replicate, use_overlapping_gqa = _should_replicate_full_attention_heads(
        total_num_heads=32,
        total_num_kv_heads=8,
        tp_size=3,
        dcp_size=1,
        enable_dcp_replicated_full_attention=True,
    )

    assert not use_overlapping_gqa
    assert should_replicate
