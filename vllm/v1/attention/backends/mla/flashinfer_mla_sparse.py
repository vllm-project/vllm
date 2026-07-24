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
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.owner_compute import (
    validate_owner_compute_scope,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    filter_peer_slots_to_owner_local,
    phys_shadow,
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

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
        return [576]

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
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
            if qk_nope_head_dim not in [128, 192]:
                return (
                    "FlashInfer MLA Sparse kernel requires qk_nope_head_dim "
                    f"in [128, 192], but got {qk_nope_head_dim}"
                )
            # Check for index_topk which indicates sparse model
            if not hasattr(hf_text_config, "index_topk"):
                return "FlashInfer MLA Sparse requires model with index_topk config"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        return "HND"


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

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
            # fp8_ds_mla packed layout: 512 NoPE + 16 scales + 128 RoPE.
            return (num_blocks, block_size, 656)
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        return None


@dataclass
class FlashInferMLASparseMetadata(AttentionMetadata):
    """Attention metadata for FlashInfer MLA Sparse backend."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int

    # Query start locations
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    # Sequence lengths for all requests (context + query)
    seq_lens: torch.Tensor
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None

    # Sparse-specific
    block_size: int = 64
    topk_tokens: int = 2048
    cp_kv_cache_interleave_size: int = 1


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


# Global workspace buffer (lazily initialized)
_fi_sparse_workspace: torch.Tensor | None = None
_fi_sparse_owner_compute_workspace: torch.Tensor | None = None


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


def _get_owner_compute_workspace_buffer(device: torch.device) -> torch.Tensor:
    global _fi_sparse_owner_compute_workspace
    if _fi_sparse_owner_compute_workspace is None:
        # Owner compute presents P times as many query rows to FlashInfer as
        # ordinary PCP attention. Its TRT-LLM softmax scratch demand exceeds
        # the default sparse workspace at production shapes (for example,
        # 538 MiB for 16K routed rows). Keep a separate shared buffer so decode
        # and non-owner sparse attention retain their existing memory contract.
        _fi_sparse_owner_compute_workspace = torch.zeros(
            2 * envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.int8,
            device=device,
        )
    return _fi_sparse_owner_compute_workspace


class FlashInferMLASparseImpl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
    """FlashInfer MLA Sparse implementation.

    Uses the TRT-LLM MLA kernel with sparse_mla_top_k parameter for
    sparse attention computation.
    """

    can_return_lse_for_decode: bool = True
    lse_base_on_e: bool = False
    supports_owner_compute: bool = True

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
        self._owner_compute_workspace_buffer: torch.Tensor | None = None
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None

        # fp8 query quantization is required when using fp8 kv_cache,
        # as the TRTLLM-GEN sparse MLA kernel requires matching dtypes
        # for query and kv_cache (mixed bf16+fp8 is not supported).
        self.supports_quant_query_input = True

    def _init_sparse_mla_scales(self, layer: AttentionLayer) -> None:
        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._k_scale_float

    def _forward_owner_compute(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, None]:
        """Run pure-prefill sparse MLA against one local history shard."""
        if attn_metadata.num_decodes != 0:
            raise RuntimeError(
                "Owner-local sparse MLA MVP supports pure-prefill batches only."
            )

        from vllm.distributed import get_dcp_group, get_pcp_group

        dcp_group = get_dcp_group()
        pcp_group = get_pcp_group()
        validate_owner_compute_scope(
            pcp_world_size=pcp_group.world_size,
            dcp_world_size=dcp_group.world_size,
            pcp_rank=pcp_group.rank_in_group,
            dcp_rank=dcp_group.rank_in_group,
            cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
            block_size=attn_metadata.block_size,
        )
        source_stride = getattr(layer, "pcp_owner_compute_source_stride", None)
        if not isinstance(source_stride, int) or source_stride <= 0:
            raise RuntimeError(
                "Owner-local sparse MLA requires a validated padded source stride."
            )
        if q.shape[0] != source_stride:
            raise RuntimeError(
                "Owner-local sparse MLA Q rows do not match the fixed source "
                f"stride: q_rows={q.shape[0]}, stride={source_stride}."
            )
        assert self.topk_indices_buffer is not None
        if self.topk_indices_buffer.shape[0] < source_stride:
            raise RuntimeError(
                "Owner-local sparse MLA top-k buffer is shorter than the fixed "
                f"source stride: rows={self.topk_indices_buffer.shape[0]}, "
                f"stride={source_stride}."
            )
        if self._owner_compute_workspace_buffer is None:
            self._owner_compute_workspace_buffer = _get_owner_compute_workspace_buffer(
                q.device
            )
        self._init_sparse_mla_scales(layer)
        assert self.bmm1_scale is not None
        assert self.bmm2_scale is not None

        num_actual = attn_metadata.num_actual_tokens
        if not 0 <= num_actual <= source_stride:
            raise RuntimeError(
                "Owner-local sparse MLA actual-token count exceeds the padded "
                f"source stride: actual={num_actual}, stride={source_stride}."
            )
        if attn_metadata.req_id_per_token.shape[0] != num_actual:
            raise RuntimeError(
                "Owner-local sparse MLA request IDs do not match actual Q rows."
            )

        peer_block_stride = getattr(layer, "pcp_peer_block_stride", None)
        if not isinstance(peer_block_stride, int) or peer_block_stride <= 0:
            raise RuntimeError(
                "Owner-local sparse MLA requires a peer-cache block stride."
            )
        topk = self.topk_indices_buffer.shape[1]
        if topk != attn_metadata.topk_tokens:
            raise RuntimeError(
                "Owner-local sparse MLA top-k width does not match attention "
                f"metadata: buffer={topk}, metadata={attn_metadata.topk_tokens}."
            )
        owner_peer_slot_cache = getattr(layer, "owner_peer_slot_cache", None)
        if owner_peer_slot_cache is None:
            raise RuntimeError(
                "Owner-local sparse MLA requires the shared peer-slot cache."
            )
        # PCPManager pads every source rank to source_stride. Global selected
        # IDs are localized once at their origin, then exchanged together with
        # the fixed-shape query payload. This route is independent of which
        # model layers own an Indexer module.
        routed_q = dcp_group.all_gather(q, dim=0)
        expected_rows = dcp_group.world_size * source_stride
        if routed_q.shape[0] != expected_rows:
            raise RuntimeError(
                "Owner-local sparse MLA gathered queries do not have fixed "
                "rank-major source shape."
            )

        def build_owner_local_slots(
            padded_peer_slots: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            routed_peer_slots = dcp_group.all_gather(padded_peer_slots, dim=0)
            if routed_peer_slots.shape[0] != expected_rows:
                raise RuntimeError(
                    "Owner-local sparse MLA gathered slots do not have fixed "
                    "rank-major source shape."
                )
            return filter_peer_slots_to_owner_local(
                routed_peer_slots,
                owner_rank=dcp_group.rank_in_group,
                dcp_world_size=dcp_group.world_size,
                blocks_per_peer=peer_block_stride,
                block_size=attn_metadata.block_size,
            )

        local_slots, local_selected_counts = (
            owner_peer_slot_cache.get_or_build_owner_local(
                num_actual,
                attn_metadata.block_table,
                source_stride=source_stride,
                owner_rank=dcp_group.rank_in_group,
                dcp_world_size=dcp_group.world_size,
                blocks_per_peer=peer_block_stride,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                block_size=attn_metadata.block_size,
                build=build_owner_local_slots,
            )
        )

        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        kernel_out = trtllm_batch_decode_with_kv_cache_mla(
            query=routed_q.unsqueeze(1),
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=self._owner_compute_workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=local_slots.unsqueeze(1),
            seq_lens=local_selected_counts,
            max_seq_len=attn_metadata.topk_tokens,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            sparse_mla_top_k=attn_metadata.topk_tokens,
            return_lse=True,
        )
        if not isinstance(kernel_out, tuple):
            raise RuntimeError(
                "Owner-local sparse MLA requires FlashInfer output and LSE."
            )
        output, lse = kernel_out
        output = output.view(-1, output.shape[-2], output.shape[-1])
        lse = self._normalize_lse(lse, output.shape[0], output.shape[1])
        empty_rows = local_selected_counts == 0
        # The LSE correction kernel explicitly maps zero-weight partials to
        # zero, even when FlashInfer leaves NaNs in an empty row. Avoid a full
        # dense output sweep solely to clear those rows.
        lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))

        # Every owner has dense partials for the same rank-major query rows.
        # First apply the exact base-2 LSE weight for this owner, then sum and
        # scatter the batch dimension so each origin receives its S rows.
        from vllm.v1.attention.ops.common import cp_lse_ag_out_rs_batch

        local_output = cp_lse_ag_out_rs_batch(
            output,
            lse,
            dcp_group,
            is_lse_base_on_e=False,
        )
        if local_output.shape[0] != source_stride:
            raise RuntimeError(
                "Owner-local sparse MLA reduce-scatter returned an invalid "
                f"source stride: {local_output.shape[0]} != {source_stride}."
            )
        return local_output[:num_actual], None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        if getattr(layer, "pcp_owner_compute", False):
            return self._forward_owner_compute(
                q,
                kv_c_and_k_pe_cache,
                attn_metadata,
                layer,
            )

        num_actual_toks = q.shape[0]

        # PCP can legitimately assign no local tokens to a rank for a short
        # prefill. TRT-LLM's TMA descriptor builder rejects a zero query
        # dimension, so preserve collective participation in the caller and
        # skip only the local sparse-attention kernel.
        if num_actual_toks == 0:
            return q.new_empty((0, self.num_heads, self.kv_lora_rank)), None

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        use_owner_history = getattr(layer, "pcp_owner_history_direct", False)
        if use_owner_history:
            if self.dcp_world_size <= 1:
                raise RuntimeError(
                    "Owner-sharded PCP history requires decode context parallelism."
                )
            peer_block_stride = getattr(layer, "pcp_peer_block_stride", None)
            if not isinstance(peer_block_stride, int):
                raise RuntimeError(
                    "Owner-sharded PCP history requires a peer-cache block stride."
                )
            owner_peer_slot_cache = getattr(layer, "owner_peer_slot_cache", None)
            if owner_peer_slot_cache is None:
                raise RuntimeError(
                    "Owner-sharded PCP history requires the shared peer-slot cache."
                )
            topk_indices_physical, seq_lens = owner_peer_slot_cache.get(
                num_actual_toks,
                attn_metadata.block_table,
                dcp_size=self.dcp_world_size,
                blocks_per_peer=peer_block_stride,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                block_size=attn_metadata.block_size,
            )
        elif self.dcp_world_size > 1:
            topk_indices_physical, seq_lens = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        elif (shadow := phys_shadow(self.topk_indices_buffer)) is not None:
            # skip_topk layers reuse the previous fresh layer's PHYSICAL
            # indices: the logical top-k indices are shared (one buffer for
            # the whole forward) and block_table/req_id are per-step constant,
            # so re-converting them per layer produces identical output. This
            # covers both the backbone (index_topk_freq > 1) and MTP draft
            # steps 1+ (index_share_for_mtp_iteration sets skip_topk; the
            # draft compact re-arranges the shadow together with the logical
            # buffer). The fresh/skip split is fixed per captured graph, so
            # the branch is cudagraph-safe.
            phys_buf, seq_buf = shadow
            wrote_fresh = getattr(layer, "indexer", None) is not None and not getattr(
                layer, "skip_topk", False
            )
            if wrote_fresh:
                topk_indices_physical, seq_lens = (
                    triton_convert_req_index_to_global_index(
                        attn_metadata.req_id_per_token[:num_actual_toks],
                        attn_metadata.block_table,
                        topk_indices,
                        BLOCK_SIZE=attn_metadata.block_size,
                        NUM_TOPK_TOKENS=topk_indices.shape[1],
                        return_valid_counts=True,
                        out=phys_buf[:num_actual_toks],
                        valid_counts_out=seq_buf[:num_actual_toks],
                    )
                )
            else:
                topk_indices_physical = phys_buf[:num_actual_toks]
                seq_lens = seq_buf[:num_actual_toks]
        else:
            topk_indices_physical, seq_lens = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        self._init_sparse_mla_scales(layer)
        assert self.bmm1_scale is not None
        assert self.bmm2_scale is not None

        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        # Single-token sparse decode. trtllm-gen requires the q_len_per_request
        # dim, but the sparse attention mask is fully per-token (each query token
        # carries its own top-k index row), so unsqueeze is sufficient and
        # correct. The MTP/multi-token q_len grouping is a perf-only layout and is
        # deferred until MTP is validated end-to-end for this backend.
        query = q.unsqueeze(1)
        block_tables = topk_indices_physical.unsqueeze(1)
        seq_lens_arg = seq_lens

        return_lse = self.need_to_return_lse_for_decode and not use_owner_history
        kernel_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_arg,
            max_seq_len=attn_metadata.topk_tokens,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            sparse_mla_top_k=attn_metadata.topk_tokens,
            return_lse=return_lse,
        )
        if return_lse:
            assert isinstance(kernel_out, tuple)
            o, lse = kernel_out
        else:
            assert isinstance(kernel_out, torch.Tensor)
            o = kernel_out
            lse = None

        out = o.view(-1, o.shape[-2], o.shape[-1])
        empty_rows = (topk_indices_physical == -1).all(dim=-1)
        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        if lse is not None:
            lse = self._normalize_lse(lse, out.shape[0], out.shape[1])
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
