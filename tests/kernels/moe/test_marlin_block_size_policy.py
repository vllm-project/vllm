# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    _choose_marlin_block_size_m,
)
from vllm.platforms.interface import DeviceCapability
from vllm.scalar_type import scalar_types


def test_gpt_oss_sm89_small_m_uses_decode_like_block_size() -> None:
    block_size_m, policy = _choose_marlin_block_size_m(
        num_tokens=1,
        num_experts=32,
        topk=4,
        hidden_size=2880,
        quant_type=scalar_types.float4_e2m1f,
        input_dtype=None,
        device_capability=DeviceCapability(8, 9),
    )

    assert (block_size_m, policy) == (64, "gpt_oss_sm89_decode_like")


def test_gpt_oss_sm89_large_m_uses_prefill_block_size() -> None:
    block_size_m, policy = _choose_marlin_block_size_m(
        num_tokens=1024,
        num_experts=32,
        topk=4,
        hidden_size=2880,
        quant_type=scalar_types.float4_e2m1f,
        input_dtype=None,
        device_capability=DeviceCapability(8, 9),
    )

    assert (block_size_m, policy) == (32, "gpt_oss_sm89_prefill_like")


def test_non_sm89_gpt_oss_shape_uses_generic_policy() -> None:
    block_size_m, policy = _choose_marlin_block_size_m(
        num_tokens=1,
        num_experts=32,
        topk=4,
        hidden_size=2880,
        quant_type=scalar_types.float4_e2m1f,
        input_dtype=None,
        device_capability=DeviceCapability(9, 0),
    )

    assert (block_size_m, policy) == (8, "auto")


def test_generic_auto_policy_keeps_int8_floor() -> None:
    block_size_m, policy = _choose_marlin_block_size_m(
        num_tokens=16,
        num_experts=32,
        topk=1,
        hidden_size=4096,
        quant_type=scalar_types.uint4,
        input_dtype=torch.int8,
        device_capability=DeviceCapability(8, 9),
    )

    assert (block_size_m, policy) == (16, "auto")
