# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


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
    BLOCK_SIZE: tl.constexpr,
):
    # num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
    # CUDA's grid-y/z are capped at 65535, so putting num_tokens there crashes
    # the launch at max-num-batched-tokens >= 65536 with "invalid argument".
    # int64: q_in_stride can be ~24K (128 heads × 192) and overflows int32
    # past num_tokens ~87K under large chunked prefill.
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    # RMSNorm in fp32 throughout — matches csrc/layernorm_kernels.cu's
    # `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel, which
    # keep x, rrms, and w all in fp32 and perform a single cast at store.
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


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

    block_size = triton.next_power_of_2(max(q_size, kv_size))
    # One warp is faster for the large prefill batches this fused kernel mainly
    # targets, while Triton's default remains slightly better for tiny decode
    # batches where launch overhead dominates.
    if num_tokens >= 1024:
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
            BLOCK_SIZE=block_size,
            num_warps=1,
        )
    else:
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
            BLOCK_SIZE=block_size,
        )
    return qr_out, kv_out


def fused_qkv_rmsnorm_tilelang(
    qr_kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TileLang variant over DeepSeek-V4's fused q/kv GEMM output."""
    assert qr_kv.ndim == 2
    assert qr_kv.dtype in (torch.bfloat16, torch.float16)
    assert q_weight.dtype == qr_kv.dtype
    assert kv_weight.dtype == qr_kv.dtype
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    q_size = q_weight.shape[0]
    kv_size = kv_weight.shape[0]
    num_tokens = qr_kv.shape[0]
    assert qr_kv.shape[1] == q_size + kv_size
    assert qr_kv.is_contiguous()

    qr_out = torch.empty(
        (num_tokens, q_size), dtype=qr_kv.dtype, device=qr_kv.device
    )
    kv_out = torch.empty(
        (num_tokens, kv_size), dtype=qr_kv.dtype, device=qr_kv.device
    )
    if num_tokens == 0:
        return qr_out, kv_out

    from vllm._tilelang_ops import fused_qkv_rmsnorm_tilelang_kernel

    block_size = triton.next_power_of_2(max(q_size, kv_size))
    # DeepSeek-V4 Flash (q=1024) is best with 32 threads; Pro (q=1536) is
    # generally best with 64 threads on B200 for the large prefill region.
    n_thr = 64 if q_size >= 1536 else 32
    dtype = "bfloat16" if qr_kv.dtype == torch.bfloat16 else "float16"
    fused_qkv_rmsnorm_tilelang_kernel(
        qr_kv,
        qr_out,
        q_weight,
        kv_out,
        kv_weight,
        eps,
        q_size,
        kv_size,
        qr_kv.shape[1],
        block_size,
        dtype,
        n_thr,
    )
    return qr_out, kv_out
