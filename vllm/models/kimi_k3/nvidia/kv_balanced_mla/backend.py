# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KIMI_K3_KV_BALANCED_MLA: the TOKENSPEED_MLA backend with its FP8 decode on the
KV-balanced kernel.

Selected with ``--attention-backend KIMI_K3_KV_BALANCED_MLA``; it is never chosen
automatically. Everything but the decode attention is TOKENSPEED_MLA's: the KV cache
layout, the decode metadata, the FP8 query, the log2 LSE the DCP combine merges, and
prefill. A step the kernel doesn't run (``decode.py``) uses TokenSpeed's decode.
"""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.tokenspeed_mla import (
    TokenspeedMLABackend,
    TokenspeedMLAImpl,
)

from .decode import LATENT, KvBalancedDecode


class KvBalancedMLABackend(TokenspeedMLABackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["fp8", "fp8_e4m3"]

    @staticmethod
    def get_name() -> str:
        return "KIMI_K3_KV_BALANCED_MLA"

    @staticmethod
    def get_impl_cls() -> type["KvBalancedMLAImpl"]:
        return KvBalancedMLAImpl

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        reason = super().supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )
        if reason is None:
            from vllm.config import get_current_vllm_config

            model_config = get_current_vllm_config().model_config
            kv_lora_rank = (
                getattr(model_config.hf_text_config, "kv_lora_rank", LATENT)
                if model_config is not None
                else LATENT
            )
            if kv_lora_rank != LATENT:
                reason = f"needs kv_lora_rank {LATENT}, got {kv_lora_rank}"
        return reason


class KvBalancedMLAImpl(TokenspeedMLAImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kv_balanced = KvBalancedDecode(self.num_heads)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.kv_balanced(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
        if out is None:
            out = super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        return out
