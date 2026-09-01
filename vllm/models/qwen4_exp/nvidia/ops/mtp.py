# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA MTP kernels for Qwen4Exp."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _mtp_residual_add_norm_kernel(
    embedding_ptr,
    hidden_ptr,
    weight_ptr,
    residual_ptr,
    norm_ptr,
    stride_embedding,
    stride_hidden_token,
    stride_hidden_stream,
    stride_residual,
    stride_norm,
    HIDDEN_SIZE: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(HIDDEN_SIZE)

    pid = tl.program_id(0)
    stream = pid % HC
    token = pid // HC
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HIDDEN_SIZE
    output_offsets = stream * HIDDEN_SIZE + offsets

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    embedding = tl.load(
        embedding_ptr + token * stride_embedding + offsets,
        mask=mask,
        other=0.0,
    )
    hidden = tl.load(
        hidden_ptr
        + token * stride_hidden_token
        + stream * stride_hidden_stream
        + offsets,
        mask=mask,
        other=0.0,
    )
    residual = (embedding.to(tl.float32) + hidden.to(tl.float32)).to(tl.bfloat16)
    weight = tl.load(weight_ptr + output_offsets, mask=mask, other=0.0)

    residual_fp32 = residual.to(tl.float32)
    rrms = tl.rsqrt(tl.sum(residual_fp32 * residual_fp32) / HIDDEN_SIZE + EPS)
    normalized = residual_fp32 * rrms
    normalized += normalized * weight.to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        residual_ptr + token * stride_residual + output_offsets,
        residual,
        mask=mask,
    )
    tl.store(
        norm_ptr + token * stride_norm + output_offsets,
        normalized,
        mask=mask,
    )


def _mtp_residual_add_norm(
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hc_count, hidden_size = hidden.shape
    assert embedding.shape == (num_tokens, hidden_size)
    assert embedding.stride(1) == 1
    assert hidden.stride(2) == 1
    assert weight.is_contiguous()
    assert weight.numel() == hc_count * hidden_size

    shape = (num_tokens, hc_count * hidden_size)
    residual = hidden.new_empty(shape)
    norm = hidden.new_empty(shape)
    _mtp_residual_add_norm_kernel[(num_tokens * hc_count,)](
        embedding,
        hidden,
        weight,
        residual,
        norm,
        embedding.stride(0),
        hidden.stride(0),
        hidden.stride(1),
        residual.stride(0),
        norm.stride(0),
        hidden_size,
        hc_count,
        EPS=eps,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return residual, norm


def _mtp_residual_add_norm_fake(
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, eps
    shape = (hidden.shape[0], hidden.shape[1] * hidden.shape[2])
    return embedding.new_empty(shape), embedding.new_empty(shape)


direct_register_custom_op(
    op_name="qwen4_exp_mtp_residual_add_norm",
    op_func=_mtp_residual_add_norm,
    fake_impl=_mtp_residual_add_norm_fake,
)


def mtp_residual_add_norm(
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.qwen4_exp_mtp_residual_add_norm(
        embedding, hidden, weight, eps
    )


__all__ = ["mtp_residual_add_norm"]
