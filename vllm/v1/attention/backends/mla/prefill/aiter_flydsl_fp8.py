# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER FlyDSL FP8 backend for MLA prefill (ROCm gfx950).

Runs AITER's FlyDSL FP8 flash attention on dynamically per-tensor quantized
Q/K/V. Q is quantized once per forward and reused by every context chunk; each
chunk quantizes only its own K/V. Whether the kernel serves the model's head
dims is decided by AITER's ``flydsl_flash_attn_fp8_supported`` at selection
time. Inputs the kernel cannot take (empty, or at least 2**31 elements) use the
parent's BF16 AITER flash attention.
"""

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import (
    AiterFlashAttnPrefillBackend,
)
from vllm.v1.attention.ops.triton_per_tensor_fp8_quant import (
    fused_per_tensor_fp8_quant,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.prefill.selector import (
        MLAPrefillSelectorConfig,
    )

# FlyDSL FP8 attention packs flattened Q/K/V/O sizes as int32.
_FLYDSL_FP8_MAX_NUMEL = 2**31


class AiterFlyDSLFP8PrefillBackend(AiterFlashAttnPrefillBackend):
    """AITER FlyDSL FP8 flash attention backend for MLA prefill."""

    # The FP8 kernel always writes BF16 output.
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FLYDSL_FP8"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_gfx950

        return on_gfx950()

    @classmethod
    def is_available(cls) -> bool:
        if not super().is_available():
            return False
        try:
            from aiter.ops.flydsl import (  # noqa: F401
                flydsl_flash_attn_fp8_func,
                flydsl_flash_attn_fp8_supported,
            )
        except ImportError:
            return False
        return True

    @classmethod
    def validate_configuration(
        cls,
        device_capability: "DeviceCapability",
        selector_config: "MLAPrefillSelectorConfig",
    ) -> list[str]:
        invalid_reasons = super().validate_configuration(
            device_capability, selector_config
        )
        if invalid_reasons:
            return invalid_reasons
        from aiter.ops.flydsl import flydsl_flash_attn_fp8_supported

        dims = selector_config.mla_dimensions
        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )
        # MLA prefill is MHA, so only the head dims matter to the check.
        if not flydsl_flash_attn_fp8_supported(
            device,
            1,
            1,
            dims.qk_nope_head_dim + dims.qk_rope_head_dim,
            dims.v_head_dim,
            dtype=current_platform.fp8_dtype(),
        ):
            invalid_reasons.append(
                f"AITER FlyDSL FP8 attention does not support {dims} on {device}"
            )
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

        from aiter.ops.flydsl import flydsl_flash_attn_fp8_func

        self.flydsl_flash_attn_fp8_func = flydsl_flash_attn_fp8_func
        # (source Q, FP8 Q, Q descale) from the current forward's new tokens.
        self._q_fp8: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def prepare_metadata(self, prefill_metadata: "MLACommonPrefillMetadata") -> None:
        super().prepare_metadata(prefill_metadata)
        self._q_fp8 = None

    @staticmethod
    def _can_use_fp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
        return (
            q.dtype == torch.bfloat16
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
        if not self._can_use_fp8(q, k, v):
            return super().run_prefill_new_tokens(
                q, k, v, return_softmax_lse, out=out, output_scale=output_scale
            )
        assert output_scale is None, (
            "AiterFlyDSLFP8PrefillBackend does not support fused quantized output."
        )
        (q_fp8, k_fp8, v_fp8), (q_descale, k_descale, v_descale) = (
            fused_per_tensor_fp8_quant(q, k, v, fp8_dtype=current_platform.fp8_dtype())
        )
        self._q_fp8 = (q, q_fp8, q_descale)
        return self.flydsl_flash_attn_fp8_func(
            q_fp8,
            k_fp8,
            v_fp8,
            causal=True,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_kv=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_kv=self._prefill_metadata.max_query_len,
            # Bit-identical to cross_seqlen=False for equal q/kv lengths, and
            # covered by AITER's AOT-built FP8 FMHA variants.
            cross_seqlen=True,
            softmax_scale=self.scale,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            out=out,
            return_lse=return_softmax_lse,
        )

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._can_use_fp8(q, k, v):
            return super().run_prefill_context_chunk(chunk, q, k, v, out=out)
        assert out is None, (
            "AiterFlyDSLFP8PrefillBackend does not report supports_out(), so it "
            "is never given a context-chunk `out` to write into."
        )
        q_fp8, q_descale = self._get_q_fp8(chunk, q)
        (k_fp8, v_fp8), (k_descale, v_descale) = fused_per_tensor_fp8_quant(
            k, v, fp8_dtype=current_platform.fp8_dtype()
        )
        return self.flydsl_flash_attn_fp8_func(
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
