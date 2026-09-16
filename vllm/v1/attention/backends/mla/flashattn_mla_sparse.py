# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA3 (page-size-1 block table, Hopper) and FA4 (native top-k gather, Blackwell)."""

import math
from dataclasses import dataclass
from functools import cache
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
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
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

# gather_kv_indices' last dim must be a whole number of n-tiles.
FA4_GATHER_TOPK_MULTIPLE = 128


class FlashAttnMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

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


class FlashAttnMLASparseImpl(SparseMLACommonImpl[FlashAttnMLASparseMetadata]):
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
        self.supports_quant_query_input = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(q, tuple):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl expects split (q_nope, q_rope) input."
            )
        q_nope, q_rope = q
        num_actual_toks = q_rope.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        index_group = self.index_group
        if isinstance(index_group, HiSparseMLAIndexGroup):
            num_decode_tokens = attn_metadata.num_decode_tokens
            outputs = []
            if num_decode_tokens:
                physical_topk, valid_counts = (
                    index_group.convert_decode_logical_to_physical_topk(
                        self.index_group_index,
                        topk_indices[:num_decode_tokens],
                        attn_metadata,
                        return_valid_counts=True,
                    )
                )
                outputs.append(
                    self._run_mqa_kernel(
                        q_nope[:num_decode_tokens],
                        q_rope[:num_decode_tokens],
                        index_group.physical_kv_cache(self.index_group_index).view(
                            kv_c_and_k_pe_cache.dtype
                        ),
                        physical_topk,
                        valid_counts,
                        attn_metadata.block_size,
                    )
                )
            if num_decode_tokens < num_actual_toks:
                cache = index_group.cache(self.index_group_index)
                if num_decode_tokens == 0 and cache.all_context_pages_resident:
                    physical_topk, valid_counts = (
                        index_group.convert_logical_to_physical_topk(
                            self.index_group_index,
                            topk_indices,
                            attn_metadata,
                            block_stride_rows=None,
                            return_valid_counts=True,
                        )
                    )
                    prefill_cache = index_group.physical_kv_cache(
                        self.index_group_index
                    ).view(kv_c_and_k_pe_cache.dtype)
                else:
                    prefill_cache, block_table, req_ids = (
                        index_group.stage_prefill_rows(
                            self.index_group_index,
                            kv_c_and_k_pe_cache,
                            attn_metadata,
                        )
                    )
                    physical_topk, valid_counts = (
                        triton_convert_req_index_to_global_index(
                            req_ids,
                            block_table,
                            topk_indices[num_decode_tokens:],
                            BLOCK_SIZE=attn_metadata.block_size,
                            NUM_TOPK_TOKENS=topk_indices.shape[1],
                            return_valid_counts=True,
                        )
                    )
                outputs.append(
                    self._run_mqa_kernel(
                        q_nope[num_decode_tokens:],
                        q_rope[num_decode_tokens:],
                        prefill_cache,
                        physical_topk,
                        valid_counts,
                        attn_metadata.block_size,
                    )
                )
            return torch.cat(outputs) if len(outputs) > 1 else outputs[0], None

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        topk_indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        return (
            self._run_mqa_kernel(
                q_nope,
                q_rope,
                kv_rows,
                topk_indices,
                valid_counts,
                attn_metadata.block_size,
                cache_is_flat=True,
            ),
            None,
        )

    def _run_mqa_kernel(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        valid_counts: torch.Tensor,
        block_size: int,
        *,
        cache_is_flat: bool = False,
    ) -> torch.Tensor:
        kv_rows = (
            kv_cache if cache_is_flat else flat_kv_row_view(kv_cache, block_size)[0]
        )

        cu_seqlens_q = torch.arange(
            0, q_rope.shape[0] + 1, dtype=torch.int32, device=q_rope.device
        )
        k_cache = kv_rows[:, self.kv_lora_rank :].unsqueeze(1).unsqueeze(1)
        v_cache = kv_rows[:, : self.kv_lora_rank].unsqueeze(1).unsqueeze(1)

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
        return out


def _fa4_cute_mla_available() -> str | None:
    """Reason the FA4 cute-DSL MLA entry point is unusable, or None."""
    from vllm.vllm_flash_attn.flash_attn_interface import (
        fa_version_unsupported_reason,
        is_fa_version_supported,
    )

    if not is_fa_version_supported(4):
        return f"FA4 unavailable: {fa_version_unsupported_reason(4)}"
    try:
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd  # noqa: F401
    except Exception as e:  # cute-DSL / cutlass import chain
        return f"FA4 cute-DSL entry point failed to import: {e!r}"
    return None


class FlashAttnMLASparseFA4Backend(AttentionBackend):
    """Uniform decode batches on FA4; a batch with prefill runs whole on FlashInfer."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # The qv kernel asserts every descale is None: BF16 KV cache only.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

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
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

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
        if kv_cache_dtype not in (None, "auto", "bfloat16"):
            return "FA4 sparse MLA supports only a BF16 KV cache"

        from vllm.utils.flashinfer import has_flashinfer

        if not has_flashinfer():
            return "FA4 sparse MLA runs its prefill batches on FlashInfer's kernel"

        if (reason := _fa4_cute_mla_available()) is not None:
            return reason

        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return None

        # The prefill lane is FlashInfer's trtllm-gen sparse kernel, which
        # cannot run rank-local PCP prefill queries over a DCP-sharded cache.
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            and parallel_config.decode_context_parallel_size > 1
        ):
            return (
                "FA4 sparse MLA does not support combined PCP+DCP; "
                "use FLASHMLA_SPARSE, which gathers each DCP KV shard before "
                "running the rank-local PCP prefill queries"
            )

        if vllm_config.model_config is None:
            return None

        hf_config = vllm_config.model_config.hf_text_config
        topk = getattr(hf_config, "index_topk", None)
        if topk is None:
            return "FA4 sparse MLA requires a model with index_topk"
        if topk % FA4_GATHER_TOPK_MULTIPLE != 0:
            return (
                f"FA4 sparse MLA requires index_topk divisible by "
                f"{FA4_GATHER_TOPK_MULTIPLE}, got {topk}"
            )

        kv_lora_rank = getattr(hf_config, "kv_lora_rank", None)
        qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", None)
        if (kv_lora_rank, qk_rope_head_dim) != (512, 64):
            return (
                "FA4 sparse MLA requires kv_lora_rank=512 and qk_rope_head_dim=64, "
                f"got {kv_lora_rank} and {qk_rope_head_dim}"
            )

        # The prefill lane's trtllm-gen sparse kernel takes only these.
        qk_nope_head_dim = getattr(hf_config, "qk_nope_head_dim", None)
        if qk_nope_head_dim not in (128, 192):
            return (
                "FA4 sparse MLA requires qk_nope_head_dim in [128, 192], "
                f"got {qk_nope_head_dim}"
            )

        # Under DCP the query is all-gathered first: the kernel sees heads * dcp_size.
        num_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        # 128 gathered heads take FA's prefill kernel, which is unvalidated under DCP.
        head_counts = (8, 16, 32, 64) if dcp_size > 1 else (8, 16, 32, 64, 128)
        if num_heads * dcp_size not in head_counts:
            counts = ", ".join(str(h) for h in head_counts)
            if dcp_size == 1:
                return (
                    f"FA4 sparse MLA requires {counts} query heads per rank, "
                    f"got {num_heads}"
                )
            return (
                "FA4 sparse MLA requires the DCP-gathered head count "
                f"(num_heads * dcp_size) to be one of {counts}, got "
                f"{num_heads} * {dcp_size} = {num_heads * dcp_size}"
            )
        return None


class FlashAttnMLASparseFA4MetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashAttnMLASparseMetadata]
):
    metadata_cls = FlashAttnMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # Without this the shared builder makes every HiSparse MTP step a prefill batch.
    hisparse_supports_multi_token_decode: ClassVar[bool] = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        if vllm_config.attention_config.hisparse_config is not None:
            # Ragged decode windows would take the resolver's per-step Python loop.
            self.require_uniform_decodes = True  # type: ignore[misc]

        num_q_heads = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
            num_q_heads, 1024
        )
        # The DCP index filter is per token, so multi-token decode rows are DCP-safe.
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )


@cache
def _fa4_varlen_scalars(
    num_tokens: int, num_kv_rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token varlen scalars; never evict, CUDA graphs bake in the pointers."""
    cu_seqlens_q = torch.arange(num_tokens + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(num_tokens + 1, dtype=torch.int32, device=device)
    seqused_k = torch.full((num_tokens,), num_kv_rows, dtype=torch.int32, device=device)
    return cu_seqlens_q, cu_seqlens_k, seqused_k


class FlashAttnMLASparseFA4Impl(SparseMLACommonImpl[FlashAttnMLASparseMetadata]):
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True
    # FA4 emits a natural-log LSE; ``_prefill`` converts trtllm-gen's log2 LSE.
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
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLASparseFA4Impl does not support alibi, sliding "
                "window, or logits soft cap."
            )
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "FlashAttnMLASparseFA4Impl supports only a BF16 KV cache."
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

        vllm_config = get_current_vllm_config()
        self.max_varlen_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._workspace_buffer: torch.Tensor | None = None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q_fused: torch.Tensor | None = None
        if isinstance(q, tuple):
            ql_nope, q_pe = q
        else:
            # The fused query's halves are 16B-aligned slices; FA4 does not copy them.
            assert q.shape[-1] == self.kv_lora_rank + self.qk_rope_head_dim
            q_fused = q
            ql_nope = q[..., : self.kv_lora_rank]
            q_pe = q[..., self.kv_lora_rank :]
        num_actual_toks = q_pe.shape[0]
        num_decode_toks = attn_metadata.num_decode_tokens

        assert self.topk_indices_buffer is not None
        if isinstance(index_group := self.index_group, HiSparseMLAIndexGroup):
            return self._forward_mqa_hisparse(
                index_group, ql_nope, q_pe, kv_c_and_k_pe_cache, attn_metadata
            )

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        if self.dcp_world_size > 1:
            topk_indices, valid_counts = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                self.topk_indices_buffer[:num_actual_toks],
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=attn_metadata.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=self.topk_indices_buffer.shape[1],
                return_valid_counts=True,
            )
        else:
            topk_indices, valid_counts = self._convert_logical_to_physical_topk(
                self.topk_indices_buffer[:num_actual_toks],
                attn_metadata,
                block_stride_rows=block_stride_rows,
                return_valid_counts=True,
            )

        if num_decode_toks >= num_actual_toks:
            out, lse = self._decode(ql_nope, q_pe, kv_rows, topk_indices, valid_counts)
        else:
            out, lse = self._prefill(
                q_fused if q_fused is not None else torch.cat((ql_nope, q_pe), dim=-1),
                kv_c_and_k_pe_cache,
                topk_indices,
                valid_counts,
            )
        assert lse is None or lse.shape == (out.shape[0], out.shape[1])
        return out, lse

    def _forward_mqa_hisparse(
        self,
        index_group: HiSparseMLAIndexGroup,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Both lanes over HiSparse's hot cache, or a staged copy of the host rows."""
        assert self.dcp_world_size == 1
        assert not self.need_to_return_lse_for_decode
        assert self.topk_indices_buffer is not None
        layer_index = self.index_group_index
        num_actual_toks = q_pe.shape[0]
        num_decode_toks = attn_metadata.num_decode_tokens
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        hot_cache = index_group.physical_kv_cache(layer_index).view(
            kv_c_and_k_pe_cache.dtype
        )

        outputs: list[torch.Tensor] = []
        if num_decode_toks > 0:
            # Once per forward; pass to FA4 unsliced, CUDA graphs capture the pointers.
            physical_topk, valid_counts = (
                index_group.convert_decode_logical_to_physical_topk(
                    layer_index,
                    topk_indices[:num_decode_toks],
                    attn_metadata,
                    return_valid_counts=True,
                )
            )
            hot_rows, block_stride_rows = flat_kv_row_view(
                hot_cache, attn_metadata.block_size
            )
            # The flat row view matches the resolver only while blocks are contiguous.
            assert block_stride_rows == attn_metadata.block_size
            out, _ = self._decode(
                ql_nope[:num_decode_toks],
                q_pe[:num_decode_toks],
                hot_rows,
                physical_topk,
                valid_counts,
            )
            outputs.append(out)

        if num_decode_toks < num_actual_toks:
            cache = index_group.cache(layer_index)
            if num_decode_toks == 0 and cache.all_context_pages_resident:
                physical_topk, valid_counts = (
                    index_group.convert_logical_to_physical_topk(
                        layer_index,
                        topk_indices,
                        attn_metadata,
                        block_stride_rows=None,
                        return_valid_counts=True,
                    )
                )
                prefill_cache = hot_cache
            else:
                prefill_cache, block_table, req_ids = index_group.stage_prefill_rows(
                    layer_index, kv_c_and_k_pe_cache, attn_metadata
                )
                # The staged tensor is dense, so its blocks need no row stride.
                physical_topk, valid_counts = triton_convert_req_index_to_global_index(
                    req_ids,
                    block_table,
                    topk_indices[num_decode_toks:],
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                    return_valid_counts=True,
                )
            out, _ = self._prefill(
                torch.cat((ql_nope[num_decode_toks:], q_pe[num_decode_toks:]), dim=-1),
                prefill_cache,
                physical_topk,
                valid_counts,
            )
            outputs.append(out)

        return (torch.cat(outputs) if len(outputs) > 1 else outputs[0]), None

    def _decode(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_rows: torch.Tensor,
        topk_indices: torch.Tensor,
        valid_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_tokens = q_pe.shape[0]
        num_kv_rows = kv_rows.shape[0]
        cu_seqlens_q, cu_seqlens_k, seqused_k = _fa4_varlen_scalars(
            self.max_varlen_tokens, num_kv_rows, kv_rows.device
        )
        # An all-sentinel row yields (0, -inf), the DCP merge identity; no post-masking.
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
        if self.need_to_return_lse_for_decode:
            assert isinstance(kernel_out, tuple)
            out, lse = kernel_out
            return out, lse
        assert isinstance(kernel_out, torch.Tensor)
        return kernel_out, None

    def _prefill(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        valid_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``kv_cache`` is any contiguous ``[blocks, block_size, 576]`` BF16 tensor."""
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(query.device)
        # BF16 KV only, so bmm1 is the plain softmax scale and bmm2 is 1.0.
        kernel_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query.unsqueeze(1),
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices.unsqueeze(1),
            seq_lens=valid_counts,
            max_seq_len=topk_indices.shape[1],
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=topk_indices.shape[1],
            return_lse=self.need_to_return_lse_for_decode,
        )
        if self.need_to_return_lse_for_decode:
            assert isinstance(kernel_out, tuple)
            o, lse = kernel_out
        else:
            assert isinstance(kernel_out, torch.Tensor)
            o, lse = kernel_out, None

        out = o.view(-1, o.shape[-2], self.kv_lora_rank)
        if lse is not None:
            lse = FlashInferMLASparseImpl._normalize_lse(
                lse, out.shape[0], out.shape[1]
            )
            # trtllm-gen returns a base-2 LSE; this impl declares base e.
            lse = lse * math.log(2.0)
            # Rows this rank owns no slot for: the DCP merge identity.
            empty_rows = valid_counts == 0
            out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
            lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return out, lse

    def autotune_hisparse_decode(self, layer: AttentionLayer) -> None:
        """Compile the FA4 decode kernel before graph capture; dummy runs skip it."""
        assert isinstance(self.index_group, HiSparseMLAIndexGroup)
        assert self.topk_indices_buffer is not None
        runtime = self.index_group.cache(self.index_group_index).runtime
        hot = runtime.hot
        kv_rows, _ = flat_kv_row_view(hot.attention_cache, hot.block_size)
        num_tokens = runtime.max_num_reqs
        topk_tokens = self.topk_indices_buffer.shape[1]
        device, dtype = kv_rows.device, kv_rows.dtype
        ql_nope = torch.zeros(
            (num_tokens, self.num_heads, self.kv_lora_rank),
            dtype=dtype,
            device=device,
        )
        q_pe = torch.zeros(
            (num_tokens, self.num_heads, self.qk_rope_head_dim),
            dtype=dtype,
            device=device,
        )
        topk_indices = (
            torch.arange(topk_tokens, dtype=torch.int32, device=device)
            .expand(num_tokens, -1)
            .contiguous()
        )
        valid_counts = torch.full(
            (num_tokens,), topk_tokens, dtype=torch.int32, device=device
        )
        self._decode(ql_nope, q_pe, kv_rows, topk_indices, valid_counts)
