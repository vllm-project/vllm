# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KIMI_K3_KV_BALANCED_MLA: an FP8 MLA decode backend on the KV-balanced kernel.

Selected with ``--attention-backend KIMI_K3_KV_BALANCED_MLA`` (and the speculative
config's ``attention_backend`` for the DSpark draft); it is never chosen
automatically. Its cache layout, decode metadata and LSE convention are those the
kernel was written for (TokenSpeed MLA's): an FP8 cache of 576 values per token in
pages of 32 or 64, uniform decode query lengths, and a log2 LSE for the DCP combine.
Every decode step runs on the kernel (``decode.py``), which raises for a step it
can't run. Prefill uses the configured MLA prefill backend.
"""

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache

from .decode import LATENT, ROPE, KvBalancedDecode

if TYPE_CHECKING:
    from vllm.config import VllmConfig

NOPE, V = 128, 128


class KimiK3KvBalancedMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    # The kernel takes an explicit causal flag, so the DSpark draft's non-causal
    # block decodes fused, with or without DCP.
    supports_non_causal_multi_token_decode: ClassVar[bool] = True
    supports_non_causal_multi_token_dcp: ClassVar[bool] = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            MLACommonMetadata,
            supports_dcp_with_varlen=True,
        )


class KimiK3KvBalancedMLABackend(MLACommonBackend):
    # The kernel writes a BF16 latent.
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["fp8", "fp8_e4m3"]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [32, 64]

    @staticmethod
    def get_name() -> str:
        return "KIMI_K3_KV_BALANCED_MLA"

    @staticmethod
    def get_impl_cls() -> type["KimiK3KvBalancedMLAImpl"]:
        return KimiK3KvBalancedMLAImpl

    @staticmethod
    def get_builder_cls() -> type[KimiK3KvBalancedMLAMetadataBuilder]:
        return KimiK3KvBalancedMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

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
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        if model_config is None:
            return None
        text_config = model_config.hf_text_config
        dims = tuple(
            getattr(text_config, name, 0)
            for name in (
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
                "kv_lora_rank",
            )
        )
        if dims != (NOPE, ROPE, V, LATENT):
            return (
                "needs MLA dimensions (qk_nope_head_dim, qk_rope_head_dim, "
                f"v_head_dim, kv_lora_rank) = {(NOPE, ROPE, V, LATENT)}, got {dims}"
            )
        return None


class KimiK3KvBalancedMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True
    # The kernel returns the LSE in log2 units.
    lse_base_on_e: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        if alibi_slopes or sliding_window or logits_soft_cap:
            raise ValueError(
                "KIMI_K3_KV_BALANCED_MLA does not support alibi_slopes, "
                "sliding_window or logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise ValueError("KIMI_K3_KV_BALANCED_MLA supports decoder attention only")
        if not is_quantized_kv_cache(self.kv_cache_dtype):
            raise ValueError(
                "KIMI_K3_KV_BALANCED_MLA requires an FP8 KV cache, got "
                f"kv_cache_dtype={self.kv_cache_dtype!r}"
            )
        self.kv_balanced = KvBalancedDecode(self.num_heads)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.kv_balanced(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
