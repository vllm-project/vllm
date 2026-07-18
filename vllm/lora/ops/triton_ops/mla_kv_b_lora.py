# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicitly routed LoRA corrections for MLA ``kv_b_proj``.

MLA does not always execute ``kv_b_proj.forward``. Dense prefill keeps a
reference to the original linear layer, while decode uses K/V weights that
were absorbed during model initialization. The helpers in this module apply
the missing low-rank delta with a token-to-adapter mapping supplied by the
caller. Using an explicit mapping is important for mixed prefill/decode
batches and for context tokens gathered from the KV cache.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK_K = 64
_BLOCK_N = 64


@triton.jit
def _routed_matmul_kernel(
    x,
    weight,
    output,
    token_lora_mapping,
    num_tokens,
    num_heads: tl.constexpr,
    contraction_size: tl.constexpr,
    output_size: tl.constexpr,
    x_stride_m,
    x_stride_h,
    x_stride_k,
    weight_stride_l,
    weight_stride_h,
    weight_stride_k,
    weight_stride_n,
    weight_offset,
    output_stride_m,
    output_stride_h,
    output_stride_n,
    add_output: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
):
    token_head = tl.program_id(0)
    token_id = token_head // num_heads
    head_id = token_head % num_heads
    output_offsets = tl.program_id(1) * block_n + tl.arange(0, block_n)
    output_mask = output_offsets < output_size

    lora_id = tl.load(token_lora_mapping + token_id)
    if lora_id == -1:
        return

    accumulator = tl.zeros((block_n,), dtype=tl.float32)
    contraction_offsets = tl.arange(0, block_k)
    for contraction_start in range(0, contraction_size, block_k):
        current_k = contraction_start + contraction_offsets
        contraction_mask = current_k < contraction_size
        x_values = tl.load(
            x + token_id * x_stride_m + head_id * x_stride_h + current_k * x_stride_k,
            mask=contraction_mask,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            weight
            + weight_offset
            + lora_id * weight_stride_l
            + head_id * weight_stride_h
            + current_k[:, None] * weight_stride_k
            + output_offsets[None, :] * weight_stride_n,
            mask=contraction_mask[:, None] & output_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(x_values[:, None] * weight_values, axis=0)

    output_ptr = (
        output
        + token_id * output_stride_m
        + head_id * output_stride_h
        + output_offsets * output_stride_n
    )
    if add_output:
        accumulator += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, accumulator, mask=output_mask)


def _routed_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    weight_stride_h: int,
    weight_stride_k: int,
    weight_stride_n: int,
    weight_offset: int = 0,
    add_output: bool,
) -> None:
    assert x.ndim == output.ndim == 3
    assert x.shape[:2] == output.shape[:2]
    num_tokens, num_heads, contraction_size = x.shape
    output_size = output.shape[-1]
    assert token_lora_mapping.shape[0] >= num_tokens
    grid = (num_tokens * num_heads, triton.cdiv(output_size, _BLOCK_N))
    _routed_matmul_kernel[grid](
        x,
        weight,
        output,
        token_lora_mapping,
        num_tokens,
        num_heads=num_heads,
        contraction_size=contraction_size,
        output_size=output_size,
        x_stride_m=x.stride(0),
        x_stride_h=x.stride(1),
        x_stride_k=x.stride(2),
        weight_stride_l=weight.stride(0),
        weight_stride_h=weight_stride_h,
        weight_stride_k=weight_stride_k,
        weight_stride_n=weight_stride_n,
        weight_offset=weight_offset,
        output_stride_m=output.stride(0),
        output_stride_h=output.stride(1),
        output_stride_n=output.stride(2),
        add_output=add_output,
        block_k=_BLOCK_K,
        block_n=_BLOCK_N,
    )


def _native_routed_matmul(
    x: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    add_output: bool,
) -> None:
    result = torch.zeros_like(output)
    for lora_id in range(weights.shape[0]):
        mask = token_lora_mapping[: x.shape[0]] == lora_id
        if torch.any(mask):
            result[mask] = torch.einsum(
                "mhk,hkn->mhn", x[mask].float(), weights[lora_id].float()
            ).to(output.dtype)
    if add_output:
        output.add_(result)
    else:
        output.copy_(result)


def _linear_weights(
    lora_a: torch.Tensor, lora_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return lora_a[:, 0].transpose(1, 2), lora_b[:, 0].transpose(1, 2)


@torch.inference_mode()
def _mla_kv_b_lora_linear(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
) -> None:
    """Add ``x @ A.T @ B.T`` to a materialized base projection."""
    if no_lora_flag_cpu.item():
        return
    x = x.view(-1, 1, x.shape[-1])
    output = output.view(-1, 1, output.shape[-1])
    rank = lora_a.shape[-2]
    intermediate = torch.empty(
        (x.shape[0], 1, rank), dtype=torch.float32, device=x.device
    )
    if not current_platform.is_cuda_alike():
        a, b = _linear_weights(lora_a, lora_b)
        _native_routed_matmul(
            x, a.unsqueeze(1), intermediate, token_lora_mapping, add_output=False
        )
        _native_routed_matmul(
            intermediate, b.unsqueeze(1), output, token_lora_mapping, add_output=True
        )
    else:
        _routed_matmul(
            x,
            lora_a,
            intermediate,
            token_lora_mapping,
            weight_stride_h=0,
            weight_stride_k=lora_a.stride(3),
            weight_stride_n=lora_a.stride(2),
            add_output=False,
        )
        _routed_matmul(
            intermediate,
            lora_b,
            output,
            token_lora_mapping,
            weight_stride_h=0,
            weight_stride_k=lora_b.stride(3),
            weight_stride_n=lora_b.stride(2),
            add_output=True,
        )


@torch.inference_mode()
def _mla_kv_b_lora_q(
    q_nope: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    v_head_dim: int,
) -> None:
    """Add the absorbed query correction ``q_nope @ B_K @ A``."""
    if no_lora_flag_cpu.item():
        return
    num_heads = q_nope.shape[1]
    rank = lora_a.shape[-2]
    full_head_dim = lora_b.shape[-2] // num_heads
    assert full_head_dim == q_nope.shape[-1] + v_head_dim
    intermediate = torch.empty(
        (*q_nope.shape[:2], rank), dtype=torch.float32, device=q_nope.device
    )
    if not current_platform.is_cuda_alike():
        b = lora_b[:, 0].view(lora_b.shape[0], num_heads, full_head_dim, rank)[
            :, :, : q_nope.shape[-1]
        ]
        a = lora_a[:, 0].unsqueeze(1).expand(-1, num_heads, -1, -1)
        _native_routed_matmul(
            q_nope, b, intermediate, token_lora_mapping, add_output=False
        )
        _native_routed_matmul(
            intermediate, a, output, token_lora_mapping, add_output=True
        )
        return
    _routed_matmul(
        q_nope,
        lora_b,
        intermediate,
        token_lora_mapping,
        weight_stride_h=full_head_dim * lora_b.stride(2),
        weight_stride_k=lora_b.stride(2),
        weight_stride_n=lora_b.stride(3),
        add_output=False,
    )
    _routed_matmul(
        intermediate,
        lora_a,
        output,
        token_lora_mapping,
        weight_stride_h=0,
        weight_stride_k=lora_a.stride(2),
        weight_stride_n=lora_a.stride(3),
        add_output=True,
    )


@torch.inference_mode()
def _mla_kv_b_lora_v(
    latent_output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    qk_nope_head_dim: int,
) -> None:
    """Add the absorbed value correction ``latent @ A.T @ B_V.T``."""
    if no_lora_flag_cpu.item():
        return
    num_heads = latent_output.shape[1]
    rank = lora_a.shape[-2]
    full_head_dim = lora_b.shape[-2] // num_heads
    assert full_head_dim == qk_nope_head_dim + output.shape[-1]
    intermediate = torch.empty(
        (*latent_output.shape[:2], rank),
        dtype=torch.float32,
        device=latent_output.device,
    )
    if not current_platform.is_cuda_alike():
        a = lora_a[:, 0].transpose(1, 2).unsqueeze(1).expand(-1, num_heads, -1, -1)
        b = (
            lora_b[:, 0]
            .view(lora_b.shape[0], num_heads, full_head_dim, rank)[
                :, :, qk_nope_head_dim:
            ]
            .transpose(2, 3)
        )
        _native_routed_matmul(
            latent_output, a, intermediate, token_lora_mapping, add_output=False
        )
        _native_routed_matmul(
            intermediate, b, output, token_lora_mapping, add_output=True
        )
        return
    _routed_matmul(
        latent_output,
        lora_a,
        intermediate,
        token_lora_mapping,
        weight_stride_h=0,
        weight_stride_k=lora_a.stride(3),
        weight_stride_n=lora_a.stride(2),
        add_output=False,
    )
    _routed_matmul(
        intermediate,
        lora_b,
        output,
        token_lora_mapping,
        weight_stride_h=full_head_dim * lora_b.stride(2),
        weight_stride_k=lora_b.stride(3),
        weight_stride_n=lora_b.stride(2),
        weight_offset=qk_nope_head_dim * lora_b.stride(2),
        add_output=True,
    )


def _fake(*args, **kwargs) -> None:
    return None


try:
    for _name, _func in (
        ("mla_kv_b_lora_linear", _mla_kv_b_lora_linear),
        ("mla_kv_b_lora_q", _mla_kv_b_lora_q),
        ("mla_kv_b_lora_v", _mla_kv_b_lora_v),
    ):
        direct_register_custom_op(
            op_name=_name,
            op_func=_func,
            mutates_args=["output"],
            fake_impl=_fake,
            tags=(torch.Tag.flexible_layout,),
        )

    mla_kv_b_lora_linear = torch.ops.vllm.mla_kv_b_lora_linear
    mla_kv_b_lora_q = torch.ops.vllm.mla_kv_b_lora_q
    mla_kv_b_lora_v = torch.ops.vllm.mla_kv_b_lora_v
except AttributeError:
    mla_kv_b_lora_linear = _mla_kv_b_lora_linear
    mla_kv_b_lora_q = _mla_kv_b_lora_q
    mla_kv_b_lora_v = _mla_kv_b_lora_v
