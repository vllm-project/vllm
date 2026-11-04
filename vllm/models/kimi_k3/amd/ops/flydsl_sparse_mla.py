# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
# This file contains code copied from the ROCm/aiter project
# (aiter/ops/flydsl/sparse_mla.py @ a61e02c26a3915011e85b103574c25d457890825).
# The original source code was licensed under the MIT license. It is carried
# in-tree so a stock aiter build still gets the kernel; only the imports and
# this header differ from the original.

"""FlyDSL sparse MLA forward on the fp8_ds_mla cache (gfx950).

Covers the slice of ``rocm_aiter_ops.triton_sparse_mla_fwd`` that
DeepSeek-V3.2 / Kimi-K3 prefill takes: bf16 q, a 656 B fp8_ds_mla record
cache, kv_lora_rank 512, rope 64, at most 16 heads, bf16 dots, no sink, no
second segment, no split-K. ``flydsl_sparse_mla_fp8ds_supported`` says whether
a call is in that slice; ``rocm_aiter_ops`` routes it here when
``VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA=1``.
"""

import functools

import torch

from vllm.models.kimi_k3.amd.ops.flydsl_sparse_mla_gfx950 import (
    KV_LORA_RANK,
    MAX_HEADS,
    QK_ROPE_HEAD_DIM,
    RECORD_BYTES,
    build_sparse_mla_fp8ds_module,
)
from vllm.v1.attention.ops.flydsl_kernels import _run_compiled

# Softmax runs in exp2.
LOG2E = 1.4426950408889634

__all__ = [
    "flydsl_sparse_mla_fp8ds_fwd",
    "flydsl_sparse_mla_fp8ds_supported",
    "flydsl_sparse_mla_fp8ds_warmup",
]

# KV rows are addressed in 64 bits, so the cache has no size cap. The index
# descriptor is sized in bytes (numel * 4) and kept well inside i32.
_MAX_INDICES = 1 << 29


@functools.cache
def _get_launcher(num_heads, has_invalid, return_lse):
    return build_sparse_mla_fp8ds_module(num_heads, has_invalid, return_lse)


@functools.lru_cache(maxsize=16)
def _is_gfx950(device_index):
    try:
        arch = torch.cuda.get_device_properties(device_index).gcnArchName
    except Exception:  # noqa: BLE001
        return False
    return arch.split(":")[0] == "gfx950"


def _kv_rows(kv):
    """The cache as a 2-D [N, 656] uint8 view, or None if it is not flat records.

    Takes [N, 656], [N, 1, 656], [N, 1, 1, 656] and an unpadded paged pool
    [nb, block, 656] of any 1-byte dtype.
    """
    if kv.element_size() != 1 or kv.ndim not in (2, 3, 4):
        return None
    if kv.shape[-1] != RECORD_BYTES or kv.stride(-1) != 1:
        return None
    if kv.ndim == 4 and (kv.shape[1] != 1 or kv.shape[2] != 1):
        return None
    if kv.ndim == 3 and kv.shape[1] != 1:
        # A padded page pitch would need paged addressing, which this kernel
        # does not do.
        if kv.stride(1) != RECORD_BYTES or kv.stride(0) != kv.shape[1] * RECORD_BYTES:
            return None
    elif kv.shape[0] > 1 and kv.stride(0) != RECORD_BYTES:
        return None
    u8 = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
    return u8.reshape(-1, RECORD_BYTES)


def flydsl_sparse_mla_fp8ds_supported(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    kv_scale: torch.Tensor | None = None,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    kv_splits: int | None = None,
    skip_reduce: bool = False,
    has_invalid: bool = False,
    dot_precision: str = "bf16",
    q_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    attn_sink: torch.Tensor | None = None,
    extra_kv: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_kv_scale: torch.Tensor | None = None,
    inv_rope_positions: torch.Tensor | None = None,
    inv_rope_cos_sin_cache: torch.Tensor | None = None,
    out_mxfp8: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> bool:
    """Can ``flydsl_sparse_mla_fp8ds_fwd`` serve this ``sparse_mla_fwd`` call?

    Takes ``sparse_mla_fwd``'s arguments. Reads host metadata only (no device
    reads, no sync), so it is safe under CUDA-graph capture. has_invalid and
    return_lse are accepted with either value.

    True iff: gfx950; q is bf16 [T, H, 576] with 1 <= H <= 16, stride(2) == 1,
    even token/head strides and a 4 B aligned base; kv_buffer has a 16 B
    aligned base and holds 1-byte fp8_ds_mla records at a 656 B pitch
    ([N, 656], [N, 1, 656], [N, 1, 1, 656] or an unpadded [nb, block, 656])
    of any size (rows are addressed in 64 bits; slot ids are int32);
    kv_indptr / kv_indices are 1-D int32 or int64 on q's
    device with T + 1 / < 2^29 entries; out, if given, is bf16 [T, H, >= 512]
    with stride(2) == 1, even strides and a 4 B aligned base; and none of
    kv_scale, q_scale, attn_sink, the extra_* segment, the inverse-RoPE / MXFP8
    epilogue, kv_splits or skip_reduce is used, with dot_precision "bf16".

    Not checkable here (it would need a device read): the per-128 KV scales
    must be powers of two <= 2^7, which vLLM's fp8_ds_mla writer guarantees
    for any latent with |x| <= 57344. See the kernel module docstring.
    """
    del has_invalid, return_lse  # both are served
    if not (torch.is_tensor(q) and q.is_cuda and q.ndim == 3):
        return False
    if q.dtype != torch.bfloat16:
        return False
    num_tokens, num_heads, d_qk = q.shape
    if kv_lora_rank != KV_LORA_RANK or qk_rope_head_dim != QK_ROPE_HEAD_DIM:
        return False
    if d_qk != KV_LORA_RANK + QK_ROPE_HEAD_DIM or not 1 <= num_heads <= MAX_HEADS:
        return False
    if any(
        a is not None
        for a in (
            kv_scale,
            q_scale,
            attn_sink,
            extra_kv,
            extra_indptr,
            extra_indices,
            extra_kv_scale,
            inv_rope_positions,
            inv_rope_cos_sin_cache,
            out_mxfp8,
            kv_splits,
        )
    ):
        return False
    if skip_reduce or dot_precision != "bf16":
        return False
    device_index = q.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    if not _is_gfx950(device_index):
        return False
    if q.stride(2) != 1 or q.stride(0) % 2 or q.stride(1) % 2:
        return False
    if q.data_ptr() % 4:
        return False
    if not torch.is_tensor(kv_buffer) or kv_buffer.device != q.device:
        return False
    kv = _kv_rows(kv_buffer)
    if kv is None:
        return False
    if kv.data_ptr() % 16:
        return False
    for t in (kv_indptr, kv_indices):
        if not torch.is_tensor(t) or t.device != q.device or t.ndim != 1:
            return False
        if t.dtype not in (torch.int32, torch.int64):
            return False
    if kv_indptr.numel() != num_tokens + 1 or kv_indices.numel() >= _MAX_INDICES:
        return False
    if out is not None:
        if out.dtype != torch.bfloat16 or out.device != q.device or out.ndim != 3:
            return False
        if out.shape[0] != num_tokens or out.shape[1] != num_heads:
            return False
        if out.shape[2] < KV_LORA_RANK or out.stride(2) != 1:
            return False
        if out.stride(0) % 2 or out.stride(1) % 2:
            return False
        if out.data_ptr() % 4:
            return False
    return True


def flydsl_sparse_mla_fp8ds_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    *,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    has_invalid: bool = False,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse MLA over an fp8_ds_mla cache, ``sparse_mla_fwd`` semantics.

    Args:
        q: [T, H, 576] bf16, H <= 16.
        kv_buffer: fp8_ds_mla records (512 e4m3 | 4 f32 per-128 scales |
            64 bf16 rope), see ``flydsl_sparse_mla_fp8ds_supported``.
        kv_indptr: [T + 1] CSR offsets into kv_indices. Row q attends
            ``kv_indices[kv_indptr[q] : kv_indptr[q + 1]]``; any length >= 0,
            no alignment needed. int64 is accepted and copied to int32.
        kv_indices: flat GLOBAL slot ids.
        softmax_scale: the layer's softmax scale.
        kv_lora_rank: latent width; only 512 is served.
        qk_rope_head_dim: rope width; only 64 is served.
        has_invalid: entries < 0 anywhere in a row are masked. Without it every
            entry must be a valid slot.
        out: optional [T, H, >= 512] bf16 destination (strided views are
            fine); the first 512 columns are written.
        return_lse: also return a fresh [T, H] f32 natural-log log-sum-exp of
            softmax_scale * q.k over the unmasked keys.

    A row with no unmasked key gives out = 0 and lse = -inf (the Gluon kernel
    returns NaN out for such a row).

    The first call for a new (H, has_invalid, return_lse) JIT-compiles the
    kernel: call ``flydsl_sparse_mla_fp8ds_warmup`` before CUDA-graph capture.
    The launch itself does no host sync.

    Returns:
        (out, lse), lse is None unless return_lse.

    """
    if not flydsl_sparse_mla_fp8ds_supported(
        q,
        kv_buffer,
        kv_indptr,
        kv_indices,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        out=out,
    ):
        raise ValueError(
            "flydsl_sparse_mla_fp8ds_fwd: unsupported call, check "
            "flydsl_sparse_mla_fp8ds_supported"
        )
    num_tokens, num_heads, _ = q.shape
    if out is None:
        out = torch.empty(
            (num_tokens, num_heads, KV_LORA_RANK), dtype=torch.bfloat16, device=q.device
        )
    lse = (
        torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    if num_tokens == 0:
        return out, lse
    kv = _kv_rows(kv_buffer)
    if kv_indptr.dtype != torch.int32 or not kv_indptr.is_contiguous():
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
    if kv_indices.dtype != torch.int32 or not kv_indices.is_contiguous():
        kv_indices = kv_indices.to(torch.int32).contiguous()
    n_indices = kv_indices.numel()
    if n_indices == 0:
        # Every row is empty; the kernel still wants a live index pointer.
        kv_indices = torch.zeros(1, dtype=torch.int32, device=q.device)
    launcher = _get_launcher(num_heads, bool(has_invalid), bool(return_lse))
    _run_compiled(
        launcher,
        q,
        kv,
        kv_indices,
        kv_indptr,
        out,
        lse if lse is not None else out,  # unused without return_lse
        num_tokens,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        float(softmax_scale * LOG2E),
        n_indices * 4,
        # Bound to q's device: current_stream() with no argument returns a
        # stream on the ambient device, which may be a different one.
        torch.cuda.current_stream(q.device),
    )
    return out, lse


def flydsl_sparse_mla_fp8ds_warmup(
    device: torch.device | str | int | None = None,
    heads: tuple[int, ...] = (1, 2, 4, 6, 8, 12, 16),
    has_invalid: tuple[bool, ...] = (False, True),
    return_lse: tuple[bool, ...] = (False, True),
) -> None:
    """Compile (and run once on a one-token problem) every listed configuration,
    so no JIT compile lands inside a CUDA-graph capture."""
    dev = (
        torch.device("cuda", torch.accelerator.current_device_index())
        if device is None
        else torch.device(device)
    )
    kv = torch.zeros(1, RECORD_BYTES, dtype=torch.uint8, device=dev)
    indptr = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    indices = torch.zeros(1, dtype=torch.int32, device=dev)
    for num_heads in heads:
        q = torch.zeros(
            1,
            num_heads,
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=dev,
        )
        for hi in has_invalid:
            for rl in return_lse:
                flydsl_sparse_mla_fp8ds_fwd(
                    q, kv, indptr, indices, 1.0, has_invalid=hi, return_lse=rl
                )
    torch.accelerator.synchronize(dev)
