# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rmsnorm_row(
    row_in,
    weight_ptr,
    row_out,
    eps,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # fp32 throughout, single cast at store. BLOCK is next_pow2(SIZE) so narrow
    # Q rows avoid KV-wide reductions that waste ALU on masked lanes.
    block = tl.arange(0, BLOCK)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


@triton.jit
def _fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
):
    # num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
    # CUDA's grid-y/z are capped at 65535, so putting num_tokens there crashes
    # the launch at max-num-batched-tokens >= 65536 with "invalid argument".
    # int64: q_in_stride can be ~24K (128 heads × 192) and overflows int32
    # past num_tokens ~87K under large chunked prefill.
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        _rmsnorm_row(
            q_ptr + token_idx * q_in_stride,
            q_weight_ptr,
            q_out_ptr + token_idx * q_out_stride,
            eps,
            Q_SIZE,
            Q_BLOCK,
        )
    else:
        _rmsnorm_row(
            kv_ptr + token_idx * kv_in_stride,
            kv_weight_ptr,
            kv_out_ptr + token_idx * kv_out_stride,
            eps,
            KV_SIZE,
            KV_BLOCK,
        )


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
        f"token dim mismatch: qr={qr.shape}, kv={kv.shape}"
    )
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    num_tokens = qr.shape[0]
    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return qr_out, kv_out

    q_block = triton.next_power_of_2(q_size)
    kv_block = triton.next_power_of_2(kv_size)
    _fused_q_kv_rmsnorm_kernel[(num_tokens, 2)](
        qr,
        qr_out,
        q_weight,
        qr.stride(0),
        qr_out.stride(0),
        kv,
        kv_out,
        kv_weight,
        kv.stride(0),
        kv_out.stride(0),
        eps,
        Q_SIZE=q_size,
        KV_SIZE=kv_size,
        Q_BLOCK=q_block,
        KV_BLOCK=kv_block,
    )
    return qr_out, kv_out
