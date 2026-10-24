# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER FlashAttention backend for MLA prefill (ROCm).

This backend calls ``aiter.flash_attn_varlen_func`` directly, which natively
supports different q/k and v head dims (qk headdim 192, v headdim 128) without
padding V, and dispatches to the fast ``aiter::fmha_fwd_`` kernel on
gfx942/gfx950 (fp16/bf16).

With ``VLLM_ROCM_MLA_PREFILL_FLYDSL_FP8=1`` on gfx950, attention instead runs
AITER's FlyDSL FP8 kernel on per-tensor dynamically quantized Q/K/V.
"""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)


# Two-pass Q/K/V per-tensor FP8 quant ported from ATOM
# atom/model_ops/triton_fused_qkv_quant.py
# (MIT, Copyright (C) 2026, Advanced Micro Devices, Inc.).
# Sizes and worker counts stay runtime args so new prefill lengths do not
# trigger a Triton recompile.
@triton.jit
def _qkv_amax(Q, K, V, Partial, NQ, NK, NV, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    part = tl.program_id(0)
    kind = tl.program_id(1)
    X = Q if kind == 0 else K if kind == 1 else V
    n = NQ if kind == 0 else NK if kind == 1 else NV
    offsets = part * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for start in range(tl.cdiv(n, PARTS * BLOCK)):
        index = offsets + start * PARTS * BLOCK
        x = tl.load(X + index, index < n, 0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(Partial + kind * PARTS + part, tl.max(acc, 0))


@triton.jit
def _qkv_quant(
    Q,
    K,
    V,
    Q8,
    K8,
    V8,
    Partial,
    Scales,
    NQ,
    NK,
    NV,
    WORKERS,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    worker = tl.program_id(0)
    kind = tl.program_id(1)
    X = Q if kind == 0 else K if kind == 1 else V
    Y = Q8 if kind == 0 else K8 if kind == 1 else V8
    n = NQ if kind == 0 else NK if kind == 1 else NV
    offsets = worker * BLOCK + tl.arange(0, BLOCK)
    if SINGLE_PASS:
        x = tl.load(X + offsets, offsets < n, 0).to(tl.float32)
        amax = tl.max(tl.abs(x), 0)
    else:
        amax = tl.max(tl.load(Partial + kind * PARTS + tl.arange(0, PARTS)), 0)
    # Positive descale for all-zero inputs keeps the FMHA free of NaNs.
    raw_descale = amax * (1.0 / 448.0)
    descale = tl.where(raw_descale > 0, raw_descale, 1e-6)
    inv = 1.0 / descale
    if worker == 0:
        tl.store(Scales + kind, descale)
    if SINGLE_PASS:
        value = tl.minimum(tl.maximum(x * inv, -448.0), 448.0)
        tl.store(Y + offsets, value, offsets < n)
    else:
        for start in range(tl.cdiv(n, WORKERS * BLOCK)):
            index = offsets + start * WORKERS * BLOCK
            x = tl.load(X + index, index < n, 0).to(tl.float32)
            value = tl.minimum(tl.maximum(x * inv, -448.0), 448.0)
            tl.store(Y + index, value, index < n)


def _qkv_config(n: int) -> tuple[int, int, int, int]:
    if n <= 8192:
        return 1, max(256, triton.next_power_of_2(n)), 1, 4
    if n <= 2**24:
        return 256, 4096, min(1024, triton.cdiv(n, 4096)), 4
    if n <= 2**26:
        return 512, 4096, min(2048, triton.cdiv(n, 4096)), 4
    return 512, 8192, min(4096, triton.cdiv(n, 8192)), 8


def _quant_fp8_qkv_per_tensor(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Return ``(q8, k8, v8, q_descale, k_descale, v_descale)``, one scale each.

    Descales are fp32 of shape ``[1]``; an all-zero tensor gets descale 1e-6.
    """
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    nq, nk, nv = q.numel(), k.numel(), v.numel()
    n = max(nq, nk, nv)
    assert n < 2**31, "int32 element offsets"
    parts, block, workers, warps = _qkv_config(n)
    q8, k8, v8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn) for x in (q, k, v))
    scales = torch.empty(3, dtype=torch.float32, device=k.device)
    partial = torch.empty((3, parts), dtype=torch.float32, device=k.device)
    single = n <= 8192
    if not single:
        _qkv_amax[(parts, 3)](
            q, k, v, partial, nq, nk, nv, parts, block, num_warps=warps
        )
    _qkv_quant[(workers, 3)](
        q,
        k,
        v,
        q8,
        k8,
        v8,
        partial,
        scales,
        nq,
        nk,
        nv,
        workers,
        parts,
        block,
        single,
        num_warps=warps,
    )
    return q8, k8, v8, scales[0:1], scales[1:2], scales[2:3]


class AiterFlashAttnPrefillBackend(MLAPrefillBackend):
    """AITER FlashAttention backend for MLA prefill."""

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FA"

    @classmethod
    def supports_compute_capability(
        cls,
        device_capability: "DeviceCapability | None",
    ) -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_mi3xx

        return on_mi3xx()

    @classmethod
    def is_available(cls) -> bool:
        from vllm._aiter_ops import rocm_aiter_ops

        return rocm_aiter_ops.is_enabled()

    @classmethod
    def validate_configuration(cls, device_capability, selector_config):
        invalid_reasons = super().validate_configuration(
            device_capability, selector_config
        )
        if envs.VLLM_BATCH_INVARIANT:
            invalid_reasons.append("batch invariance not supported")
        return invalid_reasons

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

        from aiter import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.fp8_attn_func = None
        if envs.VLLM_ROCM_MLA_PREFILL_FLYDSL_FP8:
            from aiter.ops.flydsl import (
                flydsl_flash_attn_fp8_func,
                flydsl_flash_attn_fp8_supported,
            )

            device = torch.device("cuda", torch.accelerator.current_device_index())
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            # The FP8 kernel always writes a bf16 output.
            if vllm_config.model_config.dtype == torch.bfloat16 and (
                flydsl_flash_attn_fp8_supported(
                    device, num_heads, num_heads, qk_head_dim, v_head_dim
                )
            ):
                self.fp8_attn_func = flydsl_flash_attn_fp8_func
                logger.info_once("MLA prefill uses FlyDSL FP8 attention")
            else:
                logger.warning_once(
                    "VLLM_ROCM_MLA_PREFILL_FLYDSL_FP8=1 but FlyDSL FP8 attention "
                    "does not support dtype=%s, %d heads, qk_dim=%d, v_dim=%d "
                    "on %s; using the default attention.",
                    vllm_config.model_config.dtype,
                    num_heads,
                    qk_head_dim,
                    v_head_dim,
                    device,
                )

    def _fp8_attn(self, q, k, v, **kwargs):
        q8, k8, v8, q_descale, k_descale, v_descale = _quant_fp8_qkv_per_tensor(q, k, v)
        assert self.fp8_attn_func is not None
        return self.fp8_attn_func(
            q8,
            k8,
            v8,
            softmax_scale=self.scale,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            **kwargs,
        )

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert output_scale is None, (
            "AiterFlashAttnPrefillBackend does not support fused quantized output."
        )
        if self.fp8_attn_func is not None:
            return self._fp8_attn(
                q,
                k,
                v,
                cu_seqlens_q=self._prefill_metadata.query_start_loc,
                cu_seqlens_kv=self._prefill_metadata.query_start_loc,
                max_seqlen_q=self._prefill_metadata.max_query_len,
                max_seqlen_kv=self._prefill_metadata.max_query_len,
                cross_seqlen=False,
                causal=True,
                return_lse=return_softmax_lse,
            )
        result = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_lse=return_softmax_lse,
            out=out,
        )

        # aiter returns the bare output tensor when return_lse is False, and
        # (out, softmax_lse) when it is True.
        if return_softmax_lse:
            return result[0], result[1]
        return result

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert out is None, (
            "AiterFlashAttnPrefillBackend does not report supports_out(), so it "
            "is never given a context-chunk `out` to write into."
        )
        if self.fp8_attn_func is not None:
            return self._fp8_attn(
                q,
                k,
                v,
                cu_seqlens_q=chunk.query_start_loc,
                cu_seqlens_kv=chunk.cu_seq_lens,
                max_seqlen_q=chunk.max_query_len,
                max_seqlen_kv=chunk.max_seq_len,
                cross_seqlen=True,
                causal=False,
                return_lse=True,
            )
        out, lse = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=chunk.query_start_loc,
            cu_seqlens_k=chunk.cu_seq_lens,
            max_seqlen_q=chunk.max_query_len,
            max_seqlen_k=chunk.max_seq_len,
            softmax_scale=self.scale,
            causal=False,
            return_lse=True,
        )
        return out, lse
