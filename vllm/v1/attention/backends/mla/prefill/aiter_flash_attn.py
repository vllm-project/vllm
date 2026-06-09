# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER FlashAttention backend for MLA prefill (ROCm).

This backend calls ``aiter.flash_attn_varlen_func`` directly, which natively
supports different q/k and v head dims (qk headdim 192, v headdim 128) without
padding V, and dispatches to the fast ``aiter::fmha_fwd_`` kernel on
gfx950. It also returns ``softmax_lse`` shaped ``(nheads, total_q)`` -- exactly
what ``merge_attn_states`` expects -- so no LSE transpose is required.

This is the fp16/bf16 (non-FP8) prefill path: the kernel runs in the model's
compute dtype, and ``supported_dtypes`` (inherited as ``[float16, bfloat16]``)
restricts selection accordingly. It is distinct from the FP8 ``fmha_fwd_``
varlen variant, which does not support returning ``softmax_lse`` for the MLA
head dims and so cannot serve the chunked-context merge path.
"""

from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.platforms.interface import DeviceCapability


def is_aiter_flash_attn_varlen_func_available() -> bool:
    try:
        from aiter import flash_attn_varlen_func  # noqa: F401

        return True
    except ImportError:
        return False


if is_aiter_flash_attn_varlen_func_available():
    from aiter import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None  # type: ignore[assignment]


class AiterFlashAttnPrefillBackend(MLAPrefillBackend):
    """AITER FlashAttention backend for MLA prefill (ROCm/gfx950)."""

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FA"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        # The fast fmha_fwd_* varlen path is validated on gfx950. Non-gfx950
        # ROCm falls back to FLASH_ATTN via the selector priorities.
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_gfx950

        return on_gfx950()

    @classmethod
    def is_available(cls) -> bool:
        import vllm.envs as envs
        from vllm._aiter_ops import is_aiter_found_and_supported

        return (
            envs.VLLM_ROCM_USE_AITER
            and is_aiter_flash_attn_varlen_func_available()
            and is_aiter_found_and_supported()
        )

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

        assert flash_attn_varlen_func is not None, (
            "AiterFlashAttnPrefillBackend requires aiter.flash_attn_varlen_func. "
            "Ensure AiterFlashAttnPrefillBackend.is_available() is checked first."
        )
        # aiter natively supports diff q/k vs v head dims (no V padding) and
        # returns softmax_lse as (nheads, total_q) already (no transpose).
        self.flash_attn_varlen_func = flash_attn_varlen_func

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.flash_attn_varlen_func(
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
        )

        # aiter returns the bare output tensor when return_lse is False, and
        # (out, softmax_lse) when it is True. softmax_lse is already shaped
        # (nheads, total_q), so no transpose is needed.
        if return_softmax_lse:
            return out[0], out[1]
        return out

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._prefill_metadata.chunked_context is not None
        chunked = self._prefill_metadata.chunked_context
        out, lse = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=chunked.cu_seq_lens[chunk_idx],
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=chunked.max_seq_lens[chunk_idx],
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_lse=True,
        )
        return out, lse
