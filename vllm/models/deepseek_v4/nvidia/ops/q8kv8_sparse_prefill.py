# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch


def _check_output(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tensor.shape}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def sparse_mla_q8kv8_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    *,
    out: torch.Tensor | None = None,
    max_logits: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run native FP8 sparse MLA prefill on Hopper.

    Args:
        q: OCP E4M3 query with shape ``[s_q, 64|128, 512]``.
        kv: OCP E4M3 key/value rows with shape ``[s_kv, 1, 512]``.
        indices: Gathered-row indices with shape ``[s_q, 1, topk]``.
        q_scale: Scalar FP32 query dequantization scale.
        kv_scale: Scalar FP32 key/value dequantization scale.
        attn_sink: Per-query-head FP32 attention sink.
        topk_length: Number of valid indices in each query row.
        sm_scale: Softmax scale.
        out: Optional BF16 output buffer.
        max_logits: Optional FP32 maximum-logit buffer.
        lse: Optional FP32 log-sum-exp buffer.

    Returns:
        The output, maximum logits, and log-sum-exp tensors.
    """
    if q.ndim != 3:
        raise ValueError(f"q must have shape [s_q, h_q, 512], got {q.shape}")
    if kv.ndim != 3:
        raise ValueError(f"kv must have shape [s_kv, 1, 512], got {kv.shape}")
    if indices.ndim != 3:
        raise ValueError(f"indices must have shape [s_q, 1, topk], got {indices.shape}")

    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, kv_dim = kv.shape
    topk = indices.shape[2]
    device = q.device

    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    for name, tensor in (("kv", kv), ("indices", indices)):
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"{name} must be a CUDA tensor on {device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    if q.dtype != torch.float8_e4m3fn:
        raise ValueError(f"q must be float8_e4m3fn, got {q.dtype}")
    if kv.dtype != torch.float8_e4m3fn:
        raise ValueError(f"kv must be float8_e4m3fn, got {kv.dtype}")
    if d_qk != 512 or kv_dim != 512:
        raise ValueError("q and kv head dimensions must be 512")
    if h_q not in (64, 128):
        raise ValueError(f"q head count must be padded to 64 or 128, got {h_q}")
    if h_kv != 1:
        raise ValueError(f"kv must have one head, got {h_kv}")
    if s_q == 0 or s_kv == 0:
        raise ValueError("q and kv must contain at least one row")
    if indices.shape[:2] != (s_q, 1):
        raise ValueError(
            f"indices must have shape [{s_q}, 1, topk], got {indices.shape}"
        )
    if indices.dtype != torch.int32:
        raise ValueError(f"indices must be int32, got {indices.dtype}")
    if topk == 0 or topk % 128:
        raise ValueError(f"topk must be a positive multiple of 128, got {topk}")

    for name, scale in (("q_scale", q_scale), ("kv_scale", kv_scale)):
        if (
            not scale.is_cuda
            or scale.device != device
            or scale.dtype != torch.float32
            or scale.numel() != 1
            or not scale.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous float32 CUDA scalar")
    if (
        attn_sink.shape != (h_q,)
        or attn_sink.dtype != torch.float32
        or attn_sink.device != device
        or not attn_sink.is_contiguous()
    ):
        raise ValueError(f"attn_sink must be float32 with shape [{h_q}]")
    if (
        topk_length.shape != (s_q,)
        or topk_length.dtype != torch.int32
        or topk_length.device != device
        or not topk_length.is_contiguous()
    ):
        raise ValueError(f"topk_length must be int32 with shape [{s_q}]")

    out_shape = (s_q, h_q, 512)
    stat_shape = (s_q, h_q)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bfloat16, device=device)
    else:
        _check_output(out, "out", out_shape, torch.bfloat16, device)
    if max_logits is None:
        max_logits = torch.empty(stat_shape, dtype=torch.float32, device=device)
    else:
        _check_output(max_logits, "max_logits", stat_shape, torch.float32, device)
    if lse is None:
        lse = torch.empty(stat_shape, dtype=torch.float32, device=device)
    else:
        _check_output(lse, "lse", stat_shape, torch.float32, device)

    torch.ops._C.sparse_mla_q8kv8_prefill_sm90(
        q,
        kv,
        indices,
        q_scale,
        kv_scale,
        attn_sink,
        topk_length,
        out,
        max_logits,
        lse,
        sm_scale,
    )
    return out, max_logits, lse
