# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER FlashAttention backend for MLA prefill (ROCm).

This backend calls ``aiter.flash_attn_varlen_func`` directly, which natively
supports different q/k and v head dims (qk headdim 192, v headdim 128) without
padding V, and dispatches to the fast ``aiter::fmha_fwd_`` kernel on
gfx942/gfx950 (fp16/bf16).
"""

import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)


class AiterFlashAttnPrefillBackend(MLAPrefillBackend):
    """AITER FlashAttention backend for MLA prefill"""

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
        self._fp8_prefill_enabled = False
        self._fp8_prefill_active = False
        self._fp8_prefill_func = None
        self._fp8_quant_func = None
        self._fp8_v_enabled = (
            os.environ.get("VLLM_ROCM_KIMI_K3_FP8_PREFILL_V", "1") == "1"
        )
        self._fp8_q_cache: tuple[int, torch.Tensor, torch.Tensor] | None = None
        self._fp8_prefill_min_context = int(
            os.environ.get("VLLM_ROCM_KIMI_K3_FP8_PREFILL_MIN_CONTEXT", "786432")
        )

        if os.environ.get("VLLM_ROCM_KIMI_K3_FP8_PREFILL", "0") == "1":
            try:
                from aiter.ops.triton.attention.mha import (
                    kimi_k3_fp8_prefill_gfx942,
                    supports_kimi_k3_fp8_prefill_gfx942,
                )
                from aiter.ops.triton.quant.per_head import (
                    dynamic_per_head_quant_fp8,
                )

                model_dtype = vllm_config.model_config.dtype
                self._fp8_prefill_enabled = (
                    model_dtype == torch.bfloat16
                    and num_heads == 12
                    and qk_nope_head_dim + qk_rope_head_dim == 192
                    and v_head_dim == 128
                    and supports_kimi_k3_fp8_prefill_gfx942()
                )
                if self._fp8_prefill_enabled:
                    self._fp8_prefill_func = kimi_k3_fp8_prefill_gfx942
                    self._fp8_quant_func = dynamic_per_head_quant_fp8
                    logger.info_once(
                        "Enabled opt-in gfx942 Kimi-K3 FP8 context prefill."
                    )
            except (ImportError, RuntimeError):
                logger.warning_once(
                    "Requested gfx942 Kimi-K3 FP8 prefill, but the installed "
                    "AITER build does not provide it; using BF16 ASM."
                )

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        super().prepare_metadata(prefill_metadata)
        self._fp8_q_cache = None
        chunked_context = prefill_metadata.chunked_context
        context_lens = (
            getattr(chunked_context, "context_lens_list", None)
            if chunked_context is not None
            else None
        )
        if context_lens is None and chunked_context is not None:
            seq_tot = getattr(chunked_context, "seq_tot", [])
            context_lens = [sum(seq_tot)] if seq_tot else []
        self._fp8_prefill_active = self._fp8_prefill_enabled and (
            max(context_lens or [], default=0) >= self._fp8_prefill_min_context
        )

    def _quantize_per_head(
        self, value: torch.Tensor, fp8_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fp8_quant_func is not None:
            return self._fp8_quant_func(value, fp8_dtype)
        fp8_max = torch.finfo(fp8_dtype).max
        descale = value.float().abs().amax(dim=(0, 2)) / fp8_max
        descale = descale.clamp_min(torch.finfo(torch.float32).tiny)
        quantized = (value.float() / descale[None, :, None]).clamp(-fp8_max, fp8_max)
        return quantized.to(fp8_dtype), descale[None, :].contiguous()

    def _quantize_q_once(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = q.data_ptr()
        cached = self._fp8_q_cache
        if cached is None or cached[0] != key:
            fp8_dtype = current_platform.fp8_dtype()
            q_fp8, q_descale = self._quantize_per_head(q, fp8_dtype)
            cached = (key, q_fp8, q_descale)
            self._fp8_q_cache = cached
        return cached[1], cached[2]

    def _disable_fp8_prefill(self, error: Exception) -> None:
        self._fp8_prefill_enabled = False
        self._fp8_prefill_active = False
        self._fp8_quant_func = None
        self._fp8_q_cache = None
        logger.warning_once(
            "gfx942 Kimi-K3 FP8 prefill failed (%s); using BF16 ASM.",
            error,
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
        if self._fp8_prefill_active and self._fp8_prefill_func is not None:
            try:
                fp8_dtype = current_platform.fp8_dtype()
                q_fp8, q_descale = self._quantize_q_once(q)
                k_fp8, k_descale = self._quantize_per_head(k, fp8_dtype)
                if self._fp8_v_enabled:
                    v_input, v_descale = self._quantize_per_head(v, fp8_dtype)
                else:
                    v_input = v
                    v_descale = None
                return self._fp8_prefill_func(
                    q=q_fp8,
                    k=k_fp8,
                    v=v_input,
                    cu_seqlens_q=chunk.query_start_loc,
                    cu_seqlens_k=chunk.cu_seq_lens,
                    max_seqlen_q=chunk.max_query_len,
                    max_seqlen_k=chunk.max_seq_len,
                    softmax_scale=self.scale,
                    causal=False,
                    descale_q=q_descale,
                    descale_k=k_descale,
                    descale_v=v_descale,
                )
            except Exception as error:
                self._disable_fp8_prefill(error)

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
