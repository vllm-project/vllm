# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse MLA attention backend."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
    from vllm.v1.attention.backend import CommonAttentionMetadata

logger = init_logger(__name__)


class _FlashInferMLASparseBackendBase(AttentionBackend):
    """Common metadata for concrete FlashInfer sparse MLA backends."""

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseMetadataBuilder"]:
        return FlashInferMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 576 = 512 NoPE + 64 RoPE (with-rope layout); 512 = 512 NoPE only
        # (no-rope layout, qk_rope_head_dim == 0). Both share D_V = 512 and are
        # served by the TRTLLM-GEN sparse MLA kernel.
        return [512, 576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class FlashInferMLASparseTRTLLMBackend(_FlashInferMLASparseBackendBase):
    """FlashInfer sparse MLA backend using the TRTLLM-gen launcher."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [32, 64]

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return FlashInferMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseTRTLLMMetadataBuilder"]:
        return FlashInferMLASparseTRTLLMMetadataBuilder

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
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if kv_cache_dtype == "fp8_ds_mla":
            return (
                "FLASHINFER_MLA_SPARSE SM10 does not support fp8_ds_mla kv-cache dtype"
            )

        # FlashInfer MLA sparse SM10 kernel requires qk_nope_head_dim in [128, 192].
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = hf_text_config.qk_nope_head_dim
            qk_rope_head_dim = hf_text_config.qk_rope_head_dim
            kv_lora_rank = hf_text_config.kv_lora_rank
            if qk_rope_head_dim == 0:
                # Native no-rope MLA: FlashInfer only ships the one shape
                # (nope_mla_dimensions in flashinfer.mla._core).
                if qk_nope_head_dim != 256 or kv_lora_rank != 512:
                    return (
                        "FlashInfer native no-rope MLA requires "
                        "qk_nope_head_dim=256 and kv_lora_rank=512, but got "
                        f"qk_nope_head_dim={qk_nope_head_dim}, "
                        f"kv_lora_rank={kv_lora_rank}"
                    )
            elif qk_nope_head_dim not in [128, 192]:
                return (
                    "FlashInfer MLA Sparse kernel requires qk_nope_head_dim "
                    f"in [128, 192], but got {qk_nope_head_dim}"
                )
            # Check for index_topk which indicates sparse model
            if not hasattr(hf_text_config, "index_topk"):
                return "FlashInfer MLA Sparse requires model with index_topk config"
        return None


class FlashInferMLASparseSM120Backend(_FlashInferMLASparseBackendBase):
    """FlashInfer sparse MLA backend for SM120."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_SM120"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64, 256]

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
            FlashInferMLASparseSM120Impl,
        )

        return FlashInferMLASparseSM120Impl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

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
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            return (
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API"
            )
        if dtype != torch.bfloat16:
            return "dtype not supported"
        if kv_cache_dtype not in (
            None,
            "auto",
            "fp8",
            "fp8_e4m3",
            "fp8_ds_mla",
        ):
            return "kv_cache_dtype not supported"
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            index_topk = getattr(hf_text_config, "index_topk", None)
            if index_topk is None:
                return (
                    "FLASHINFER_MLA_SPARSE_SM120 requires a model with "
                    "index_topk config"
                )
            if int(index_topk) != 2048:
                return (
                    "FLASHINFER_MLA_SPARSE_SM120 requires index_topk=2048; "
                    f"got {index_topk}"
                )
        return None


@dataclass
class FlashInferMLASparseMetadata(SparseMLACommonMetadata):
    """Attention metadata for FlashInfer MLA Sparse backend."""


class FlashInferMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashInferMLASparseMetadata]
):
    """Builder for FlashInfer MLA Sparse attention metadata."""

    metadata_cls = FlashInferMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
            num_q_heads, 1024
        )
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )


class FlashInferMLASparseTRTLLMMetadataBuilder(FlashInferMLASparseMetadataBuilder):
    """Metadata builder for the SM100 TRT-LLM sparse MLA kernel."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    hisparse_supports_multi_token_decode: ClassVar[bool] = True

    def _build_req_id_per_token(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> torch.Tensor:
        return common_attn_metadata.token_to_req_indices(self.req_id_per_token_buffer)


# Global workspace buffer (lazily initialized)
_fi_sparse_workspace: torch.Tensor | None = None


def _get_workspace_buffer(device: torch.device) -> torch.Tensor:
    global _fi_sparse_workspace
    if _fi_sparse_workspace is None:
        # FlashInfer's CuteDSL MLA-decode tactic requires an int8 workspace;
        # the trtllm-gen path views it as uint8, so int8 is safe for all backends.
        _fi_sparse_workspace = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.int8,
            device=device,
        )
    return _fi_sparse_workspace


class FlashInferMLASparseImpl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
    """FlashInfer MLA Sparse implementation.

    Uses the TRT-LLM MLA kernel with sparse_mla_top_k parameter for
    sparse attention computation.
    """

    can_return_lse_for_decode: bool = True
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
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLASparseImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLASparseImpl"
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

        self._workspace_buffer: torch.Tensor | None = None
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None

        # Native no-rope MLA additionally requires a per-query-token active
        # top-k length tensor.
        self.is_nope_mla = self.qk_rope_head_dim == 0

        # fp8 query quantization is required when using fp8 kv_cache,
        # as the TRTLLM-GEN sparse MLA kernel requires matching dtypes
        # for query and kv_cache (mixed bf16+fp8 is not supported).
        self.supports_quant_query_input = True

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        self._prepare_mqa_kernel(layer, q.device)

        index_group = self.index_group
        if isinstance(index_group, HiSparseMLAIndexGroup):
            num_decode_tokens = attn_metadata.num_decode_tokens
            decode_out: torch.Tensor | None = None
            decode_lse: torch.Tensor | None = None
            if num_decode_tokens > 0:
                physical_topk, valid_counts = (
                    index_group.convert_decode_logical_to_physical_topk(
                        self.index_group_index,
                        topk_indices[:num_decode_tokens],
                        attn_metadata,
                        return_valid_counts=True,
                    )
                )
                decode_out, decode_lse = self._run_mqa_kernel(
                    q[:num_decode_tokens],
                    index_group.physical_kv_cache(self.index_group_index).view(
                        kv_c_and_k_pe_cache.dtype
                    ),
                    physical_topk,
                    valid_counts,
                )
                if num_decode_tokens == num_actual_toks:
                    return decode_out, decode_lse

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
                return self._run_mqa_kernel(
                    q,
                    index_group.physical_kv_cache(self.index_group_index).view(
                        kv_c_and_k_pe_cache.dtype
                    ),
                    physical_topk,
                    valid_counts,
                )

            prefill_cache, block_table, req_ids = index_group.stage_prefill_rows(
                self.index_group_index, kv_c_and_k_pe_cache, attn_metadata
            )
            prefill_indices, prefill_lens = triton_convert_req_index_to_global_index(
                req_ids,
                block_table,
                topk_indices[num_decode_tokens:],
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
            prefill_out, prefill_lse = self._run_mqa_kernel(
                q[num_decode_tokens:],
                prefill_cache,
                prefill_indices,
                prefill_lens,
            )
            if decode_out is None:
                return prefill_out, prefill_lse
            output = torch.cat((decode_out, prefill_out))
            if decode_lse is None:
                return output, None
            assert prefill_lse is not None
            return output, torch.cat((decode_lse, prefill_lse))

        _, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )

        if self.dcp_world_size > 1:
            topk_indices_physical, seq_lens = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            topk_indices_physical, seq_lens = self._convert_logical_to_physical_topk(
                topk_indices,
                attn_metadata,
                block_stride_rows=block_stride_rows,
                return_valid_counts=True,
            )

        return self._run_mqa_kernel(
            q,
            kv_c_and_k_pe_cache,
            topk_indices_physical,
            seq_lens,
        )

    def _prepare_mqa_kernel(
        self,
        layer: AttentionLayer,
        device: torch.device,
    ) -> None:
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(device)

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._k_scale_float

    def autotune_hisparse_decode(self, layer: AttentionLayer) -> None:
        """Autotune the largest legal HiSparse decode batch."""
        assert isinstance(self.index_group, HiSparseMLAIndexGroup)
        cache = self.index_group.cache(self.index_group_index)
        assert self.topk_indices_buffer is not None

        runtime = cache.runtime
        kv_cache = runtime.hot.attention_cache
        num_tokens = runtime.max_num_reqs
        topk_tokens = self.topk_indices_buffer.shape[1]
        self._prepare_mqa_kernel(layer, kv_cache.device)

        q = torch.zeros(
            (num_tokens, self.num_heads, self.qk_head_dim),
            dtype=kv_cache.dtype,
            device=kv_cache.device,
        )
        topk_indices = (
            torch.arange(
                topk_tokens,
                dtype=torch.int32,
                device=kv_cache.device,
            )
            .expand(num_tokens, -1)
            .contiguous()
        )
        seq_lens = torch.full(
            (num_tokens,),
            topk_tokens,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        self._run_mqa_kernel(q, kv_cache, topk_indices, seq_lens)

    def _run_mqa_kernel(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self._workspace_buffer is not None
        assert self.bmm1_scale is not None
        assert self.bmm2_scale is not None

        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        kv_cache = kv_cache.view(q.dtype)

        # Single-token sparse decode. trtllm-gen requires the q_len_per_request
        # dim, but the sparse attention mask is fully per-token (each query token
        # carries its own top-k index row), so unsqueeze is sufficient and
        # correct. The MTP/multi-token q_len grouping is a perf-only layout and is
        # deferred until MTP is validated end-to-end for this backend.
        query = q.unsqueeze(1)
        block_tables = topk_indices.unsqueeze(1)
        seq_lens_arg = seq_lens

        # page_table width = topk buffer width, which kpool widens past
        # index_topk (topk_tokens) and rounds up to a multiple of 128. The
        # kernel treats sparse_mla_top_k as the page-table *capacity* and bounds
        # the active per-query length by ``seq_lens`` (the compacted valid
        # count), so the -1 padding slots past seq_lens are never attended to.
        # Use the actual buffer width instead of the fixed topk_tokens, which
        # mismatches the page_table when index_kpool > 1.
        sparse_topk_capacity = topk_indices.shape[1]

        extra_kwargs: dict[str, torch.Tensor] = {}
        empty_rows: torch.Tensor | None = None
        if self.is_nope_mla:
            # The native no-rope kernel takes the active top-k length per query
            # token (``seq_lens`` here is already the compacted per-token valid
            # count, int32) and rejects zero-length rows. Point empty rows at a
            # single valid dummy slot with length 1 and zero their output after
            # the launch. ``triton_convert_req_index_to_global_index`` packs the
            # valid indices into a contiguous prefix, which is what the kernel
            # requires of the page table.
            empty_rows = seq_lens == 0
            topk_indices[:, 0] = topk_indices[:, 0].masked_fill(empty_rows, 0)
            extra_kwargs["sparse_mla_top_k_lens"] = seq_lens.clamp(min=1)

        kernel_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_arg,
            max_seq_len=sparse_topk_capacity,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            sparse_mla_top_k=sparse_topk_capacity,
            return_lse=self.need_to_return_lse_for_decode,
            **extra_kwargs,
        )
        if self.need_to_return_lse_for_decode:
            assert isinstance(kernel_out, tuple)
            o, lse = kernel_out
        else:
            assert isinstance(kernel_out, torch.Tensor)
            o = kernel_out
            lse = None

        out = o.view(-1, o.shape[-2], o.shape[-1])
        if lse is not None:
            lse = self._normalize_lse(lse, out.shape[0], out.shape[1])
        if empty_rows is None and lse is not None:
            empty_rows = (topk_indices == -1).all(dim=-1)
        if empty_rows is not None:
            out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
            if lse is not None:
                lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return out, lse

    @staticmethod
    def _normalize_lse(
        lse: torch.Tensor,
        num_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        # FlashInfer returns the decode LSE either as 2D (num_tokens, num_heads)
        # or 3D ((num_tokens, num_heads, 1) / (num_tokens, 1, num_heads)).
        # Collapse all of these to the (num_tokens, num_heads) the shared DCP
        # reducer expects.
        if lse.dim() == 3:
            if lse.shape[-1] == 1:
                lse = lse.squeeze(-1)
            elif lse.shape[1] == 1:
                lse = lse.squeeze(1)
            elif lse.shape[0] * lse.shape[1] == num_tokens:
                lse = lse.reshape(num_tokens, lse.shape[-1])
        if lse.shape != (num_tokens, num_heads):
            raise RuntimeError(
                "Unexpected FlashInfer sparse MLA LSE shape: "
                f"{tuple(lse.shape)}, expected ({num_tokens}, {num_heads})."
            )
        return lse
