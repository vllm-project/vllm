# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA attention backend for CUDA SM80+ (including Ampere)."""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    align_blocks_to_rows,
    flat_kv_row_view,
)
from vllm.v1.attention.ops.metadata import compute_token_to_req_indices
from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd
from vllm.v1.kv_cache_interface import AttentionSpec


class TritonMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        return align_blocks_to_rows(spec)

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl[Any]]:
        return TritonMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

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
        if kv_cache_dtype not in (None, "auto", "float16", "bfloat16"):
            return (
                "TRITON_MLA_SPARSE supports only FP16/BF16 KV cache "
                "(use --kv-cache-dtype auto/bfloat16/float16)"
            )

        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            if vllm_config.parallel_config.decode_context_parallel_size > 1:
                return "TRITON_MLA_SPARSE does not support DCP"

            hf_text_config = vllm_config.model_config.hf_text_config
            if not hasattr(hf_text_config, "index_topk"):
                return "TRITON_MLA_SPARSE requires a model with index_topk"
            kv_lora_rank = getattr(hf_text_config, "kv_lora_rank", None)
            qk_rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", None)
            if kv_lora_rank != 512:
                return (
                    f"TRITON_MLA_SPARSE requires kv_lora_rank=512, got {kv_lora_rank}"
                )
            if qk_rope_head_dim not in (0, 64):
                return (
                    "TRITON_MLA_SPARSE requires qk_rope_head_dim in (0, 64), "
                    f"got {qk_rope_head_dim}"
                )
        return None


@dataclass
class TritonMLASparseMetadata(SparseMLACommonMetadata):
    pass


class TritonMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[TritonMLASparseMetadata]
):
    metadata_cls = TritonMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        num_q_heads = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {16: 128, 32: 128, 64: 256, 128: 256}.get(num_q_heads, 256)
        self._init_reorder_batch_threshold(threshold, supports_spec_as_decode=True)
        self.supports_draft_decode_metadata_update = self.dcp_world_size == 1

    def update_draft_decode_metadata(self, metadata: SparseMLACommonMetadata) -> None:
        num_tokens = metadata.num_decode_tokens
        if num_tokens == 0:
            return
        compute_token_to_req_indices(
            metadata.query_start_loc,
            metadata.req_id_per_token,
            num_tokens,
            num_tokens,
        )


class TritonMLASparseImpl(SparseMLACommonImpl[TritonMLASparseMetadata]):
    group_conversion_decode_only: ClassVar[bool] = True
    lse_base_on_e: bool = True

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
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "TritonMLASparseImpl does not support alibi_slopes, "
                "sliding_window, or logits_soft_cap."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "TritonMLASparseImpl only supports decoder self-attention."
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                "TritonMLASparseImpl supports only FP16/BF16 KV cache; "
                f"got kv_cache_dtype={kv_cache_dtype!r}."
            )

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
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        assert self.topk_indices_buffer is not None, (
            "Indexer or topk_indices_buffer required for sparse MLA"
        )
        self.supports_quant_query_input = False

    def _forward_mqa_kernel(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        valid_counts: torch.Tensor,
        *,
        layer: AttentionLayer,
        block_size: int,
        is_decode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q_nope, q_rope = q
            q_tensor = (
                q_nope
                if self.qk_rope_head_dim == 0
                else torch.cat((q_nope, q_rope), dim=-1)
            )
        else:
            q_tensor = q

        kv_rows, _ = flat_kv_row_view(kv_cache, block_size)
        return triton_mla_sparse_fwd(
            q_tensor,
            kv_rows,
            topk_indices,
            sm_scale=self.scale,
            kv_lora_rank=self.kv_lora_rank,
            return_lse=self.need_to_return_lse_for_decode,
        )
