# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass
from functools import cache
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
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
    FlashInferMLASparseTRTLLMBackend,
    FlashInferMLASparseTRTLLMMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    align_blocks_to_rows,
    flat_kv_row_view,
)
from vllm.v1.attention.ops.metadata import compute_token_to_req_indices
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


class FlashAttnMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        return align_blocks_to_rows(spec)

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLASparseMetadataBuilder"]:
        return FlashAttnMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl[Any]]:
        return FlashAttnMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9

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
                "FlashAttention MLA Sparse currently supports only FP16/BF16 KV cache"
            )

        if not flash_attn_supports_mla():
            return "FlashAttention MLA not supported on this device"

        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            if vllm_config.parallel_config.decode_context_parallel_size > 1:
                return "FlashAttention MLA Sparse does not support DCP for now"

            hf_text_config = vllm_config.model_config.hf_text_config
            if not hasattr(hf_text_config, "index_topk"):
                return "FlashAttention MLA Sparse requires model with index_topk"
        return None


@dataclass
class FlashAttnMLASparseMetadata(SparseMLACommonMetadata):
    pass


class FlashAttnMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashAttnMLASparseMetadata]
):
    metadata_cls = FlashAttnMLASparseMetadata
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
        # Everything else is a view of runner buffers; the per-token request
        # map is a builder buffer that build() filled for the captured batch.
        compute_token_to_req_indices(
            metadata.query_start_loc,
            metadata.req_id_per_token,
            num_tokens,
            num_tokens,
        )


class FlashAttnMLASparseImpl(SparseMLACommonImpl[FlashAttnMLASparseMetadata]):
    group_conversion_decode_only: ClassVar[bool] = True

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
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl does not support alibi, sliding window, "
                "or logits soft cap."
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl currently supports only FP16/BF16 KV cache."
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
        self.cu_seqlens_q_buffer = torch.arange(
            self.topk_indices_buffer.shape[0] + 1,
            dtype=torch.int32,
            device=self.topk_indices_buffer.device,
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
        if not isinstance(q, tuple):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl expects split (q_nope, q_rope) input."
            )
        q_nope, q_rope = q
        kv_rows, _ = flat_kv_row_view(kv_cache, block_size)

        cu_seqlens_q = self.cu_seqlens_q_buffer[: q_rope.shape[0] + 1]
        v_cache = kv_rows[:, : self.kv_lora_rank].unsqueeze(1).unsqueeze(1)
        if self.qk_rope_head_dim == 0:
            # FA3's QV path requires the 64-wide Q/K specialization. For NoPE
            # MLA, zero Q preserves QV-only attention scores while providing a
            # valid TMA shape.
            _FA3_QV_HEAD_DIM = 64
            q_rope = q_nope.new_zeros(*q_nope.shape[:-1], _FA3_QV_HEAD_DIM)
            k_cache = v_cache[..., :_FA3_QV_HEAD_DIM]
        else:
            k_cache = kv_rows[:, self.kv_lora_rank :].unsqueeze(1).unsqueeze(1)

        out = flash_attn_varlen_func(
            q=q_rope,
            k=k_cache,
            v=v_cache,
            q_v=q_nope,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=topk_indices.shape[1],
            seqused_k=valid_counts,
            block_table=topk_indices,
            softmax_scale=self.scale,
            causal=True,
            fa_version=3,
        )
        return out, None


def _fa4_cute_mla_available() -> str | None:
    """Reason the FA4 cute-DSL MLA entry point is unusable, or None."""
    from vllm.vllm_flash_attn import flash_attn_interface as fa

    if not fa.is_fa_version_supported(4):
        return f"FA4 unavailable: {fa.fa_version_unsupported_reason(4)}"
    try:
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd  # noqa: F401
    except Exception as e:  # cute-DSL / cutlass import chain
        return f"FA4 cute-DSL entry point failed to import: {e!r}"
    return None


class FlashAttnMLASparseFA4Backend(FlashInferMLASparseTRTLLMBackend):
    """FA4 for is_decode kernel calls; trtllm-gen for the rest."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # The qv kernel asserts every descale is None: BF16 KV cache only.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA_SPARSE_FA4"

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLASparseFA4MetadataBuilder"]:
        return FlashAttnMLASparseFA4MetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl[Any]]:
        return FlashAttnMLASparseFA4Impl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 512 latent + 64 RoPE; split back apart for the qv kernel.
        return [576]

    @classmethod
    def supports_combination(cls, *args: Any, **kwargs: Any) -> str | None:
        from vllm.config import get_current_vllm_config_or_none
        from vllm.utils.flashinfer import has_flashinfer

        if not has_flashinfer():
            return "FA4 sparse MLA runs its prefill batches on FlashInfer's kernel"
        if (reason := _fa4_cute_mla_available()) is not None:
            return reason
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return None
        if vllm_config.model_config is not None:
            hf_config = vllm_config.model_config.hf_text_config
            topk = getattr(hf_config, "index_topk", None)
            # gather_kv_indices' last dim must be a whole number of n-tiles.
            if topk is None or topk % 128 != 0:
                return f"FA4 sparse MLA requires index_topk % 128 == 0, got {topk}"
            dims = (
                getattr(hf_config, "kv_lora_rank", None),
                getattr(hf_config, "qk_rope_head_dim", None),
            )
            if dims != (512, 64):
                return (
                    "FA4 sparse MLA requires (kv_lora_rank, qk_rope_head_dim) "
                    f"== (512, 64), got {dims}"
                )
            # Under DCP the query is all-gathered: the kernel sees heads * dcp_size.
            dcp_size = vllm_config.parallel_config.decode_context_parallel_size
            num_heads = vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            )
            # 128 gathered heads take FA's prefill kernel, unvalidated under DCP.
            head_counts = (8, 16, 32, 64) if dcp_size > 1 else (8, 16, 32, 64, 128)
            if num_heads * dcp_size not in head_counts:
                return (
                    f"FA4 sparse MLA requires {head_counts} gathered query heads, "
                    f"got num_heads={num_heads} * dcp_size={dcp_size}"
                )
        return super().supports_combination(*args, **kwargs)


class FlashAttnMLASparseFA4MetadataBuilder(FlashInferMLASparseTRTLLMMetadataBuilder):
    """The trtllm-gen builder, for the prefill lane's DCP workspace pre-size.

    Its UNIFORM_BATCH support and varlen decode bound also fit FA4, which runs
    one query row per token and bakes the row count into its varlen scalars.
    """


@cache
def _fa4_varlen_scalars(
    num_tokens: int, num_kv_rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token varlen scalars; never evict, CUDA graphs bake in the pointers."""
    cu_seqlens_q = torch.arange(num_tokens + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(num_tokens + 1, dtype=torch.int32, device=device)
    seqused_k = torch.full((num_tokens,), num_kv_rows, dtype=torch.int32, device=device)
    return cu_seqlens_q, cu_seqlens_k, seqused_k


class FlashAttnMLASparseFA4Impl(FlashInferMLASparseImpl):
    # FA4 emits a natural-log LSE; prefill converts trtllm-gen's log2 LSE.
    lse_base_on_e: bool = True

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
        if not is_decode:
            out, lse = super()._forward_mqa_kernel(
                q,
                kv_cache,
                topk_indices,
                valid_counts,
                layer=layer,
                block_size=block_size,
                is_decode=is_decode,
            )
            return out, None if lse is None else lse * math.log(2.0)
        if isinstance(q, tuple):
            ql_nope, q_pe = q
        else:
            # 16B-aligned halves of the fused query; FA4 reads them in place.
            ql_nope, q_pe = q[..., : self.kv_lora_rank], q[..., self.kv_lora_rank :]
        kv_rows, _ = flat_kv_row_view(kv_cache, block_size)
        num_tokens = q_pe.shape[0]
        num_kv_rows = kv_rows.shape[0]
        assert self.topk_indices_buffer is not None
        cu_seqlens_q, cu_seqlens_k, seqused_k = _fa4_varlen_scalars(
            self.topk_indices_buffer.shape[0], num_kv_rows, kv_rows.device
        )
        # An all-sentinel row yields (0, -inf), the DCP merge identity; no masking.
        kernel_out = flash_attn_varlen_func(
            q=q_pe,
            k=kv_rows[:, self.kv_lora_rank :].unsqueeze(1),
            v=kv_rows[:, : self.kv_lora_rank].unsqueeze(1),
            q_v=ql_nope,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q[: num_tokens + 1],
            max_seqlen_k=num_kv_rows,
            cu_seqlens_k=cu_seqlens_k[: num_tokens + 1],
            seqused_k=seqused_k[:num_tokens],
            gather_kv_indices=topk_indices,
            # Part of FA4's compile key: pass it unconditionally.
            gather_kv_valid_length=valid_counts,
            softmax_scale=self.scale,
            # Causality is in the top-k list; causal=True would clamp flat rows.
            causal=False,
            fa_version=4,
            return_softmax_lse=self.need_to_return_lse_for_decode,
        )
        return kernel_out if self.need_to_return_lse_for_decode else (kernel_out, None)
