# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_default_config,
    try_get_optimal_moe_config,
)


@pytest.mark.parametrize(
    ("m", "m_is_per_expert", "expected_block_m"),
    [
        (512, False, 16),
        (820, False, 64),
        (512, True, 64),
        (64, True, 16),
    ],
)
def test_fp8_block_default_config_m_tile_respects_m_semantics(
    m: int, m_is_per_expert: bool, expected_block_m: int
):
    with patch(
        "vllm.model_executor.layers.fused_moe.fused_moe.current_platform.is_rocm",
        return_value=False,
    ):
        config = get_default_config(
            M=m,
            E=512,
            N=256,
            K=2048,
            topk=10,
            dtype="fp8_w8a8",
            block_shape=[128, 128],
            m_is_per_expert=m_is_per_expert,
        )

    assert config["BLOCK_SIZE_M"] == expected_block_m


def test_fp8_block_default_config_per_expert_m_flows_through_fallback():
    with (
        patch(
            "vllm.model_executor.layers.fused_moe.get_config",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.fused_moe.get_moe_configs",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.fused_moe.current_platform.is_rocm",
            return_value=False,
        ),
    ):
        config = try_get_optimal_moe_config(
            w1_shape=(512, 512, 2048),
            w2_shape=(512, 2048, 256),
            top_k=10,
            dtype="fp8_w8a8",
            M=512,
            block_shape=[128, 128],
            m_is_per_expert=True,
        )

    assert config["BLOCK_SIZE_M"] == 64
