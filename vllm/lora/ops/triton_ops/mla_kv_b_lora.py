# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicitly routed LoRA corrections for MLA ``kv_b_proj``."""

import torch

from vllm.lora.ops.triton_ops.routed_lora_matmul import routed_lora_two_stage


def mla_kv_b_lora_linear(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
) -> None:
    """Add ``x @ A.T @ B.T`` to a materialized base projection."""
    first_weight = lora_a.transpose(2, 3)
    second_weight = lora_b.transpose(2, 3)
    routed_lora_two_stage(
        x.view(-1, 1, x.shape[-1]),
        first_weight,
        second_weight,
        output.view(-1, 1, output.shape[-1]),
        token_lora_mapping,
        no_lora_flag_cpu,
    )


def mla_kv_b_lora_q(
    q_nope: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    v_head_dim: int,
) -> None:
    """Add the absorbed query correction ``q_nope @ B_K @ A``."""
    num_loras = lora_b.shape[0]
    num_heads = q_nope.shape[1]
    full_head_dim = lora_b.shape[-2] // num_heads
    assert full_head_dim == q_nope.shape[-1] + v_head_dim
    first_weight = lora_b[:, 0].view(
        num_loras, num_heads, full_head_dim, lora_b.shape[-1]
    )[:, :, : q_nope.shape[-1]]
    routed_lora_two_stage(
        q_nope,
        first_weight,
        lora_a,
        output,
        token_lora_mapping,
        no_lora_flag_cpu,
    )


def mla_kv_b_lora_v(
    latent_output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    qk_nope_head_dim: int,
) -> None:
    """Add the absorbed value correction ``latent @ A.T @ B_V.T``."""
    num_loras = lora_b.shape[0]
    num_heads = latent_output.shape[1]
    full_head_dim = lora_b.shape[-2] // num_heads
    assert full_head_dim == qk_nope_head_dim + output.shape[-1]
    first_weight = lora_a.transpose(2, 3)
    second_weight = (
        lora_b[:, 0]
        .view(num_loras, num_heads, full_head_dim, lora_b.shape[-1])[
            :, :, qk_nope_head_dim:
        ]
        .transpose(2, 3)
    )
    routed_lora_two_stage(
        latent_output,
        first_weight,
        second_weight,
        output,
        token_lora_mapping,
        no_lora_flag_cpu,
    )
