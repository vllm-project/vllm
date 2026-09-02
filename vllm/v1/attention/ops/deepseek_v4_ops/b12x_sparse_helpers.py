# SPDX-License-Identifier: Apache-2.0
"""Optional B12x sparse-MLA helpers for GLM fp8_ds_mla cache experiments."""

from __future__ import annotations

import os
import math

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_GLM_HEAD_DIM = 576
_GLM_VALUE_DIM = 512
_GLM_RECORD_BYTES = 656


def b12x_mla_enabled() -> bool:
    return os.getenv("GLM52_B12X_MLA", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _profile_enabled() -> bool:
    return os.getenv("GLM52_PREFILL_PROFILE", "0").lower() not in {
        "",
        "0",
        "false",
        "no",
    }


def _cuda_capture_active() -> bool:
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(is_capturing and is_capturing())


class _CudaProfileRegion:
    def __init__(self, region: str, **fields: object) -> None:
        self.region = region
        self.fields = fields
        self.enabled = (
            _profile_enabled()
            and torch.cuda.is_available()
            and not _cuda_capture_active()
        )
        self.start_event: torch.cuda.Event | None = None
        self.end_event: torch.cuda.Event | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        torch.cuda.nvtx.range_push(f"glm52:{self.region}")
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def stop(self) -> None:
        if not self.enabled or self.start_event is None or self.end_event is None:
            return
        self.end_event.record()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        details = " ".join(f"{key}={value}" for key, value in self.fields.items())
        logger.info(
            "GLM52_PREFILL_PROFILE region=%s elapsed_ms=%.3f %s",
            self.region,
            self.start_event.elapsed_time(self.end_event),
            details,
        )


def b12x_glm_mla_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run B12x GLM_NSA sparse MLA over global-slot top-k indices.

    Args:
        q: [batch, seq, heads, 576] bf16 query tensor.
        kv_cache: fp8_ds_mla cache, either [blocks, block, 656] or
            [blocks, block, 1, 656] uint8-compatible storage.
        topk_indices: [batch, seq, topk] int32 global physical slot ids.
        softmax_scale: attention scale.

    Returns:
        (out [batch, seq, heads, 512], lse [batch, heads, seq]) or None when the
        optional B12x package/path is unavailable or the shape is unsupported.
    """
    if not b12x_mla_enabled():
        return None
    if q.dim() != 4 or q.shape[-1] != _GLM_HEAD_DIM:
        return None
    if topk_indices.dim() != 3:
        return None
    if kv_cache.dtype != torch.uint8:
        kv_cache = kv_cache.view(torch.uint8)
    if kv_cache.shape[-1] != _GLM_RECORD_BYTES:
        return None

    batch, seq_len, num_heads, _ = q.shape
    if num_heads % 16 != 0:
        logger.warning_once(
            "B12x GLM sparse MLA skipped: num_heads=%s is not divisible by 16",
            num_heads,
        )
        return None

    try:
        from b12x.attention.mla.prefill_mg import run_unified_prefill_mg
        from b12x.attention.mla.traits import ComputeMode, ModelType, ScaleFormat
    except Exception as exc:
        logger.warning_once("B12x GLM sparse MLA unavailable: %r", exc)
        return None

    q_flat = q.reshape(batch * seq_len, num_heads, _GLM_HEAD_DIM).contiguous()
    indices_flat = topk_indices.reshape(batch * seq_len, topk_indices.shape[-1])
    indices_flat = indices_flat.to(device=q.device, dtype=torch.int32).contiguous()
    kv_flat = kv_cache.reshape(-1, 1, _GLM_RECORD_BYTES).contiguous()
    mg_n_hg = 2 if num_heads % 32 == 0 else 1

    profile = _CudaProfileRegion(
        "attention.b12x_glm_mla",
        q_shape=tuple(q.shape),
        kv_shape=tuple(kv_cache.shape),
        topk=topk_indices.shape[-1],
        mg_n_hg=mg_n_hg,
    )
    profile.start()
    try:
        out, lse = run_unified_prefill_mg(
            q=q_flat,
            kv_cache=kv_flat,
            topk_indices=indices_flat,
            sm_scale=float(softmax_scale),
            page_block_size=1,
            compute_mode=ComputeMode.FP8,
            mg_n_hg=mg_n_hg,
            model_type=ModelType.GLM_NSA,
            scale_format=ScaleFormat.ARBITRARY_FP32,
        )
    finally:
        profile.stop()
    out = out.reshape(batch, seq_len, num_heads, _GLM_VALUE_DIM)
    lse = lse.mul(math.log(2.0))
    lse = lse.reshape(batch, seq_len, num_heads).permute(0, 2, 1).contiguous()
    return out, lse
