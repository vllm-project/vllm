# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for ModelConfig.verify_with_parallel_config's DCP/PCP
interaction (https://github.com/vllm-project/vllm/issues/51429).

`ParallelConfig` accepts `decode_context_parallel_size` (DCP) spanning the
`prefill_context_parallel_size` (PCP) axis, but
`ModelConfig.verify_with_parallel_config` only accounted for TP-induced KV
duplication when computing the maximum supported DCP size, so it rejected
topologies that `ParallelConfig` itself considers valid. These tests call
the validator directly with lightweight stand-ins so they run without a
GPU or downloading a real model.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.config import ModelConfig


def _make_model_config(
    *, total_num_attention_heads: int, total_num_kv_heads: int
) -> Any:
    """Build a minimal stand-in exposing only what
    `verify_with_parallel_config` reads before the DCP/PCP branch returns.
    """
    return SimpleNamespace(
        model_arch_config=SimpleNamespace(
            total_num_attention_heads=total_num_attention_heads
        ),
        use_mla=False,
        multimodal_config=None,
        get_total_num_kv_heads=lambda: total_num_kv_heads,
    )


def _make_parallel_config(
    *,
    tensor_parallel_size: int,
    prefill_context_parallel_size: int,
    decode_context_parallel_size: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        tensor_parallel_size=tensor_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
        decode_context_parallel_size=decode_context_parallel_size,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
    )


@pytest.mark.parametrize(
    ("tp", "pcp", "dcp", "num_q_heads", "num_kv_heads"),
    [
        # DCP spans the PCP axis; TP alone would not permit any DCP
        # (tp == kv_heads), but PCP duplication makes it valid.
        (4, 2, 2, 32, 8),
        (8, 2, 2, 32, 8),
    ],
)
def test_dcp_spanning_pcp_axis_is_accepted(tp, pcp, dcp, num_q_heads, num_kv_heads):
    model_config = _make_model_config(
        total_num_attention_heads=num_q_heads, total_num_kv_heads=num_kv_heads
    )
    parallel_config = _make_parallel_config(
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
    )

    # Should not raise.
    ModelConfig.verify_with_parallel_config(model_config, parallel_config)


def test_dcp_without_pcp_still_requires_tp_duplication():
    # pcp=1, tp <= kv_heads: no duplication source at all, must still fail.
    model_config = _make_model_config(
        total_num_attention_heads=32, total_num_kv_heads=8
    )
    parallel_config = _make_parallel_config(
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=2,
    )

    with pytest.raises(ValueError, match="Decode context parallelism"):
        ModelConfig.verify_with_parallel_config(model_config, parallel_config)


def test_dcp_without_pcp_tp_duplication_still_enforces_max():
    # pcp=1, tp=16, kv_heads=8 -> max_dcp_size == 2 (unchanged from before).
    model_config = _make_model_config(
        total_num_attention_heads=32, total_num_kv_heads=8
    )
    parallel_config = _make_parallel_config(
        tensor_parallel_size=16,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=2,
    )

    # Should not raise: matches the pre-existing accepted control case.
    ModelConfig.verify_with_parallel_config(model_config, parallel_config)

    parallel_config_too_large = _make_parallel_config(
        tensor_parallel_size=16,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=4,
    )
    with pytest.raises(ValueError, match="exceeds the maximum"):
        ModelConfig.verify_with_parallel_config(
            model_config, parallel_config_too_large
        )


def test_dcp_with_non_divisible_tp_kv_ratio_is_not_falsely_rejected():
    # pcp=1, tp=12, kv_heads=8: tp > kv_heads so there IS a duplication
    # source, even though floor(tp / kv_heads) == 1. dcp=2 must be rejected
    # for exceeding the (low) max_dcp_size, not misreported as having no
    # duplication source at all.
    model_config = _make_model_config(
        total_num_attention_heads=24, total_num_kv_heads=8
    )
    parallel_config = _make_parallel_config(
        tensor_parallel_size=12,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=2,
    )

    with pytest.raises(ValueError, match="exceeds the maximum"):
        ModelConfig.verify_with_parallel_config(model_config, parallel_config)


def test_dcp_exceeding_combined_tp_pcp_capacity_is_rejected():
    # tp=4, pcp=2 -> max_dcp_size = max(1, 4 // 8) * 2 = 2; dcp=4 must fail.
    model_config = _make_model_config(
        total_num_attention_heads=32, total_num_kv_heads=8
    )
    parallel_config = _make_parallel_config(
        tensor_parallel_size=4,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=4,
    )

    with pytest.raises(ValueError, match="exceeds the maximum"):
        ModelConfig.verify_with_parallel_config(model_config, parallel_config)
