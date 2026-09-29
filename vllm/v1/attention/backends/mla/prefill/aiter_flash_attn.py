# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER FlashAttention backend for MLA prefill (ROCm).

This backend calls ``aiter.flash_attn_varlen_func`` directly, which natively
supports different q/k and v head dims (qk headdim 192, v headdim 128) without
padding V, and dispatches to the fast ``aiter::fmha_fwd_`` kernel on
gfx942/gfx950 (fp16/bf16).

With ``VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL=1``, BF16 prefill attention runs
AITER's FlyDSL FP8 kernel on dynamically per-tensor quantized Q/K/V instead.
Q is quantized once per forward and reused by every context chunk. Support is
decided by AITER's capability check; anything it rejects uses BF16 attention.
"""

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.ops.triton_per_tensor_fp8_quant import (
    fused_per_tensor_fp8_quant,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)

# FlyDSL FP8 attention packs flattened Q/K/V/O sizes as int32.
_FLYDSL_FP8_MAX_NUMEL = 2**31


@functools.cache
def _get_flydsl_fp8_attn(
    num_heads: int, qk_head_dim: int, v_head_dim: int
) -> Callable | None:
    """Return AITER's FlyDSL FP8 attention if enabled and supported, else None."""
    from vllm._aiter_ops import rocm_aiter_ops

    if not rocm_aiter_ops.is_flydsl_fp8_prefill_enabled():
        return None
    try:
        from aiter.ops.flydsl import (
            flydsl_flash_attn_fp8_func,
            flydsl_flash_attn_fp8_supported,
        )
    except ImportError as e:
        logger.warning(
            "VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL is set, but the installed "
            "AITER lacks FlyDSL FP8 attention or its capability check (%s). "
            "Using BF16 prefill attention.",
            e,
        )
        return None
    device = torch.device(
        current_platform.device_type, torch.accelerator.current_device_index()
    )
    if not flydsl_flash_attn_fp8_supported(
        device,
        num_heads,
        num_heads,
        qk_head_dim,
        v_head_dim,
        dtype=current_platform.fp8_dtype(),
    ):
        logger.warning(
            "VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL is set, but AITER FlyDSL FP8 "
            "attention does not support device=%s, heads=%d, qk_head_dim=%d, "
            "v_head_dim=%d. Using BF16 prefill attention.",
            device,
            num_heads,
            qk_head_dim,
            v_head_dim,
        )
        return None
    logger.info_once("Using AITER FlyDSL FP8 attention for MLA prefill.")
    return flydsl_flash_attn_fp8_func


class AiterFlashAttnPrefillBackend(MLAPrefillBackend):
    """AITER FlashAttention backend for MLA prefill."""

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FA"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_mi3xx

        return on_mi3xx()

    @classmethod
    def is_available(cls) -> bool:
        from vllm._aiter_ops import rocm_aiter_ops

        return rocm_aiter_ops.is_enabled()

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
        # The FP8 kernel always writes BF16 output.
        self.flydsl_fp8_attn = (
            _get_flydsl_fp8_attn(
                num_heads, qk_nope_head_dim + qk_rope_head_dim, v_head_dim
            )
            if vllm_config.model_config.dtype == torch.bfloat16
            else None
        )
        # (source Q, FP8 Q, Q descale) from the current forward's new tokens.
        self._q_fp8: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def use_flydsl_fp8(self) -> bool:
        return self.flydsl_fp8_attn is not None

    def prepare_metadata(self, prefill_metadata: "MLACommonPrefillMetadata") -> None:
        super().prepare_metadata(prefill_metadata)
        self._q_fp8 = None

    def _can_use_fp8(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
        return (
            self.flydsl_fp8_attn is not None
            and q.dtype == torch.bfloat16
            and q.shape[0] > 0
            and k.shape[0] > 0
            and max(q.numel(), k.numel(), v.numel()) < _FLYDSL_FP8_MAX_NUMEL
        )

    def _get_q_fp8(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse the new-token FP8 Q for this chunk's token slice if it matches."""
        if self._q_fp8 is not None:
            q_src, q_fp8, q_descale = self._q_fp8
            q_src = q_src[chunk.token_slice]
            if (
                q_src.data_ptr() == q.data_ptr()
                and q_src.shape == q.shape
                and q_src.stride() == q.stride()
            ):
                return q_fp8[chunk.token_slice], q_descale
        (q_fp8,), (q_descale,) = fused_per_tensor_fp8_quant(
            q, fp8_dtype=current_platform.fp8_dtype()
        )
        return q_fp8, q_descale

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
        if self._can_use_fp8(q, k, v):
            assert self.flydsl_fp8_attn is not None
            (q_fp8, k_fp8, v_fp8), (q_descale, k_descale, v_descale) = (
                fused_per_tensor_fp8_quant(
                    q, k, v, fp8_dtype=current_platform.fp8_dtype()
                )
            )
            self._q_fp8 = (q, q_fp8, q_descale)
            return self.flydsl_fp8_attn(
                q_fp8,
                k_fp8,
                v_fp8,
                causal=True,
                cu_seqlens_q=self._prefill_metadata.query_start_loc,
                cu_seqlens_kv=self._prefill_metadata.query_start_loc,
                max_seqlen_q=self._prefill_metadata.max_query_len,
                max_seqlen_kv=self._prefill_metadata.max_query_len,
                cross_seqlen=False,
                softmax_scale=self.scale,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                out=out,
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
        if self._can_use_fp8(q, k, v):
            assert self.flydsl_fp8_attn is not None
            q_fp8, q_descale = self._get_q_fp8(chunk, q)
            (k_fp8, v_fp8), (k_descale, v_descale) = fused_per_tensor_fp8_quant(
                k, v, fp8_dtype=current_platform.fp8_dtype()
            )
            return self.flydsl_fp8_attn(
                q_fp8,
                k_fp8,
                v_fp8,
                causal=False,
                cu_seqlens_q=chunk.query_start_loc,
                cu_seqlens_kv=chunk.cu_seq_lens,
                max_seqlen_q=chunk.max_query_len,
                max_seqlen_kv=chunk.max_seq_len,
                cross_seqlen=True,
                softmax_scale=self.scale,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
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
