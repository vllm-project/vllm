# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.owner_compute import (
    get_owner_prefill_mode,
    validate_owner_compute_scope,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    build_rotated_dcp_peer_block_table,
    filter_peer_slots_to_owner_local,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import (
    reshape_attn_output_for_spec_decode,
    reshape_query_for_spec_decode,
    split_prefill_chunks,
)
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# For FP8 sparse attention we have two implementations:
# 1. Mixed batch mode: use the FP8 decode kernel for both prefill and decode this is
#    done by treating all tokens as single batch.
# 2. Separate prefill and decode mode: use the BF16 prefill kernel for prefill
#    (upconverting the FP8 cache to BF16 then calling the prefill kernel) and using
#    the FP8 decode kernel for decode.
# Currently we use #1 when the number of heads per rank is low (i.e. TP) since the BF16
# prefill kernel requires padding the number of heads to 128 while the decode does not
# so when the per-rank head count is below MIN_HEADS_FOR_BF16_PREFILL we use the mixed
# batch mode (#1).
MIN_HEADS_FOR_BF16_PREFILL = 32

"""
NOTE: FlashMLA Sparse uses an fp8 cache with the following format

For DeepSeek V3.2, in the "FP8 with scale" format, each token's KV cache is 656
Bytes, structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512
    `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values.
    The first `float32` is the scale for the first 128 `float8_e4m3` values,
    the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.

For DeepSeek V4, in the "FP8 with scale" format, each token's KV cache is 584
Bytes, structured as:
-   **First 448 bytes:** The "quantized NoPE" part, containing 448
    `float8_e4m3` values.
-   **Next 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.
-   **Last 8 bytes:** Scale factors, containing 7 `ue8m0` values + 1B pad.
    The first `ue8m0` is the scale for the first 64 `float8_e4m3` values,
    the second for the next 64, and so on.
"""


class FlashMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashMLASparseMetadataBuilder"]:
        return FlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLASparseImpl"]:
        return FlashMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V3.2 layout: 512 NoPE + 64 RoPE = 576.
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [9, 10]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # V3.2 main MLA: 656-byte custom storage format. See module docstring.
            return (num_blocks, block_size, 656)
        else:
            return (num_blocks, block_size, head_size)


@dataclass
class FlashMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048

    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    seq_lens: torch.Tensor | None = None
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None
    cp_kv_cache_interleave_size: int = 1

    @dataclass
    class FP8KernelMetadata:
        scheduler_metadata: FlashMLASchedMeta
        dummy_block_table: torch.Tensor
        cache_lens: torch.Tensor

    @dataclass
    class FP8SeparatePrefillDecode:
        @dataclass
        class Decode:
            seq_lens: torch.Tensor
            kernel_metadata: "FlashMLASparseMetadata.FP8KernelMetadata"
            decode_query_len: int  # needed for reshape in spec decode

        @dataclass
        class Prefill:
            # Request ID for each token: -1 for decode tokens, request index
            # (0, 1, 2, ...) for prefill tokens.
            # Shape: [num_actual_tokens]
            request_ids: torch.Tensor

            # Workspace start offsets for all prefill requests
            # Shape: [num_prefill_reqs], adjusted in-place per chunk to be
            # 0-indexed within each chunk. Used to map prefill tokens to workspace
            # offsets in convert_logical_index_to_physical_index
            workspace_starts: torch.Tensor

            @dataclass
            class Chunk:
                """Metadata for a chunk of prefill requests.

                Prefill requests may be chunked to fit within the fixed workspace size.
                """

                tokens_slice: slice
                block_table: torch.Tensor
                req_start_idx: int
                workspace_starts: torch.Tensor
                chunk_tot_seqlen: int
                pcp_peer_block_table: torch.Tensor | None = None
                pcp_peer_block_table_key: tuple[int, int, int] | None = None

            chunks: list[Chunk]

        num_prefills: int = 0
        num_decodes: int = 0
        num_prefill_tokens: int = 0
        num_decode_tokens: int = 0

        decode: Decode | None = None
        prefill: Prefill | None = None

    fp8_extra_metadata: FP8SeparatePrefillDecode | FP8KernelMetadata | None = None
    fp8_use_mixed_batch: bool = False


def get_prefill_workspace_size(max_model_len: int):
    # NOTE(Lucas): 5 is a magic number for controlling the prefill buffer size.
    # May be tuned later.
    # Memory usage: 5 * max_model_len * 576 * 2 bytes
    #   Example: DeepSeek-V3.2 with max_model_len=163840 ->
    #            5 * 163840 * 576 * 2 = ~900 MB
    # This fits nicely below the typical MoE workspace size of >2GB so this is "free"
    return max_model_len * 5


class FlashMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    require_uniform_decodes: ClassVar[bool] = True
    metadata_cls = FlashMLASparseMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config

        num_q_heads = self.model_config.get_num_attention_heads(parallel_config)
        if current_platform.is_device_capability_family(100):
            threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
                num_q_heads, 1024
            )
        else:
            threshold = {16: 128, 32: 128, 64: 256, 128: 256}.get(num_q_heads, 256)
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
        )

        sm_count = num_compute_units(device.index)

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        # FP8 decode kernel only supports h_q = 64 or 128, so we need to pad
        self.fp8_decode_padded_heads = (
            FlashMLASparseImpl._compute_fp8_decode_padded_heads(self.num_heads)
        )

        self.use_fp8_kv_cache = cache_config.cache_dtype == "fp8_ds_mla"
        self.use_peer_pcp_fp8 = (
            envs.VLLM_USE_PCP_OWNER_HISTORY
            and parallel_config.prefill_context_parallel_size > 1
        )
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        # Shape: [max_num_seqs], all elements = topk_tokens (constant for full-CG)
        self.topk_tokens_tensor = torch.full(
            (max_num_seqs,), self.topk_tokens, device=device, dtype=torch.int32
        )
        # Shape: [max_num_seqs], all elements = max_model_len
        self.max_model_len_tensor = torch.full(
            (max_num_seqs,),
            self.model_config.max_model_len,
            device=device,
            dtype=torch.int32,
        )
        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty(
            (max_num_seqs, 1), dtype=torch.int32, device=self.device
        )

        # Equation taken from FlashMLA/csrc/api/sparse_decode.h
        # For sparse FP8 decode, the formula depends on architecture:
        # - SM90 (Hopper): num_sm_parts = num_sms / s_q / (h_q/64)
        # - SM100 (Blackwell head64/head64x2): num_sm_parts = num_sms / s_q
        # - SM100 (Blackwell head128): num_sm_parts = num_sms / s_q / 2
        # For max buffer size, use s_q = 1 (the case that produces largest output)
        # Use padded head count since that's what will be passed to the kernel
        h_q = self.fp8_decode_padded_heads
        if current_platform.is_device_capability_family(100):
            # SM100 head64 or head64x2 uses full SM count
            max_num_sm_parts = sm_count
        else:
            # SM90 uses h_q/64 divisor
            max_num_sm_parts = sm_count // max(1, h_q // 64)
        self.tile_scheduler_metadata_buffer = torch.empty(
            # TileSchedulerMetaDataSize = 8
            # see: FlashMLA/csrc/params.h
            (max_num_sm_parts, 8),
            dtype=torch.int32,
            device=device,
        )
        # Sized for per-request batching (num_decodes + 1)
        self.num_splits_buffer = torch.empty(
            (max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        )

    def _build_fp8_mixed_decode_prefill(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> "FlashMLASparseMetadata.FP8KernelMetadata":
        """Build FP8 metadata treating MQA tokens as one batch.

        The scheduler initializes lazily from the runtime query shape, which may
        be the full batch or only decodes when prefills use dense MHA. This avoids
        the BF16 prefill kernel's head-padding overhead at high TP.
        """
        num_tokens = common_attn_metadata.num_actual_tokens

        # Use padded head count since that's what the kernel will see
        padded_heads = self.fp8_decode_padded_heads

        # Build metadata for all tokens as a single batch
        scheduler_metadata, _ = get_mla_metadata(
            cache_seqlens=self.topk_tokens_tensor[:1],  # Single batch
            num_q_tokens_per_head_k=num_tokens * padded_heads,
            topk=self.topk_tokens,
            num_heads_q=padded_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )

        fp8_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=scheduler_metadata,
            cache_lens=self.max_model_len_tensor[:1],
            dummy_block_table=self.dummy_block_table[:1],
        )

        return fp8_metadata

    def _build_fp8_separate_prefill_decode(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        metadata: FlashMLASparseMetadata,
    ) -> "FlashMLASparseMetadata.FP8SeparatePrefillDecode":
        num_tokens = common_attn_metadata.num_actual_tokens

        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            metadata.num_decodes,
            metadata.num_prefills,
            metadata.num_decode_tokens,
            num_tokens - metadata.num_decode_tokens,
        )

        decode_query_len = 0
        active_num_decodes = num_decodes
        if num_decodes > 0:
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            decode_query_len = (query_start_loc_cpu[1] - query_start_loc_cpu[0]).item()
            assert decode_query_len > 0
            active_num_decodes = num_decode_tokens // decode_query_len
            assert active_num_decodes * decode_query_len == num_decode_tokens

        FP8Meta = FlashMLASparseMetadata.FP8SeparatePrefillDecode
        fp8_metadata = FP8Meta(
            num_decodes=active_num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
        )

        # Extract prefill sequence lengths (context + query, not just query)
        # Decode requests come first in the batch, prefill requests follow
        prefill_request_id = None
        prefill_workspace_starts = None
        prefill_chunks = None

        # For pure decode batches, prefill_request_id will be None
        # For mixed batches, it will have -1 for decode and request_id for prefill
        if num_prefills > 0:
            # Upper bound is exact for prefill rows (the `[num_decodes:]`
            # slice below), so no D2H sync is needed.
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

            prefill_seq_lens_cpu = seq_lens_cpu[num_decodes:]

            # Build prefill_request_id: -1 for decode, request index for
            # prefill. This enables a single
            # convert_logical_index_to_physical_index call for all tokens
            prefill_request_id = torch.full(
                (num_tokens,), -1, dtype=torch.int32, device=self.device
            )
            # Map prefill tokens to their request IDs (0, 1, 2, ...)
            for req_idx in range(num_prefills):
                # Get query token range for this prefill request
                global_req_idx = num_decodes + req_idx
                req_query_start = query_start_loc_cpu[global_req_idx]
                req_query_end = query_start_loc_cpu[global_req_idx + 1]
                prefill_request_id[req_query_start:req_query_end] = req_idx

            # will be adjusted by chunk loop
            prefill_workspace_starts_cpu = torch.zeros(
                num_prefills, dtype=torch.int32, pin_memory=True
            )
            prefill_workspace_starts_cpu[1:] = torch.cumsum(
                prefill_seq_lens_cpu[:-1], dim=0
            )
            # populated by non-blocking copy after prefill_workspace_starts_cpu is
            # updated by each chunk
            prefill_workspace_starts = torch.empty(
                num_prefills, dtype=torch.int32, device=self.device
            )

            # Chunk prefill requests to fit within workspace size
            max_prefill_buffer_size = get_prefill_workspace_size(
                self.vllm_config.model_config.max_model_len
            )
            chunk_bounds = split_prefill_chunks(
                prefill_seq_lens_cpu, max_prefill_buffer_size
            )

            prefill_chunks = []
            for chunk_start, chunk_end in chunk_bounds:
                # Adjust workspace_starts in-place per chunk to be
                # 0-indexed within each chunk
                # Example: seq_lens=[10,15,20,5], chunks=[[0,2],[2,4]]
                #   Initial: workspace_starts=[0,10,25,45]
                #   After:   workspace_starts=[0,10,0,20]
                #           (chunk 0 starts at 0, chunk 1 starts at 0)
                offset = prefill_workspace_starts_cpu[chunk_start].item()
                prefill_workspace_starts_cpu[chunk_start:chunk_end] -= offset

                chunk_tot_seqlen = prefill_seq_lens_cpu[chunk_start:chunk_end].sum()
                token_start = query_start_loc_cpu[num_decodes + chunk_start].item()
                token_end = query_start_loc_cpu[num_decodes + chunk_end].item()
                tokens_slice = slice(token_start, token_end)

                # Create chunk view of gpu tensor
                chunk_workspace_starts = prefill_workspace_starts[chunk_start:chunk_end]
                chunk_block_table = common_attn_metadata.block_table_tensor[
                    num_decodes + chunk_start : num_decodes + chunk_end
                ]

                prefill_chunks.append(
                    FP8Meta.Prefill.Chunk(
                        tokens_slice=tokens_slice,
                        block_table=chunk_block_table,
                        req_start_idx=chunk_start,
                        workspace_starts=chunk_workspace_starts,
                        chunk_tot_seqlen=chunk_tot_seqlen,
                    )
                )

            prefill_workspace_starts.copy_(
                prefill_workspace_starts_cpu, non_blocking=True
            )

            fp8_metadata.prefill = FP8Meta.Prefill(
                request_ids=prefill_request_id,
                workspace_starts=prefill_workspace_starts,
                chunks=prefill_chunks,
            )

        if num_decodes > 0:
            # Use padded head count since that's what the kernel will see
            scheduler_metadata, _ = get_mla_metadata()

            kernel_meta = FlashMLASparseMetadata.FP8KernelMetadata(
                scheduler_metadata=scheduler_metadata,
                dummy_block_table=self.dummy_block_table[:active_num_decodes],
                cache_lens=self.max_model_len_tensor[:active_num_decodes],
            )
            fp8_metadata.decode = FP8Meta.Decode(
                seq_lens=common_attn_metadata.seq_lens[:active_num_decodes],
                kernel_metadata=kernel_meta,
                decode_query_len=decode_query_len,
            )

        return fp8_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashMLASparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

        materialize_owner_prefill = (
            self.use_peer_pcp_fp8
            and get_owner_prefill_mode() == "materialize"
            and metadata.num_prefills > 0
            and metadata.num_decodes == 0
        )
        fp8_use_mixed_batch = not materialize_owner_prefill and (
            self.num_heads < MIN_HEADS_FOR_BF16_PREFILL
            or (self.use_peer_pcp_fp8 and metadata.num_prefills > 0)
        )
        metadata.fp8_use_mixed_batch = fp8_use_mixed_batch
        if self.use_fp8_kv_cache:
            if fp8_use_mixed_batch:
                metadata.fp8_extra_metadata = self._build_fp8_mixed_decode_prefill(
                    common_attn_metadata
                )
            else:
                metadata.fp8_extra_metadata = self._build_fp8_separate_prefill_decode(
                    common_attn_metadata, metadata
                )

        return metadata


class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
    owner_compute_returns_projected_values: bool = True
    supports_owner_history_prefill_materialization: bool = True

    @staticmethod
    def _compute_fp8_decode_padded_heads(num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128
        # Compute padded head count for decode
        return 64 if num_heads <= 64 else 128

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
        self.softmax_scale = scale
        # Prefill BF16 kernel requires 64 on Hopper, 128 on Blackwell
        self.prefill_padding = (
            128 if current_platform.is_device_capability_family(100) else 64
        )
        # FlashMLA V3.2 accepts per-row top-k lengths only on SM100. Hopper's
        # sparse FP8 kernel rejects topk_length, so retain direct peer reads
        # there instead of selecting an unsupported owner-compute path.
        self.supports_owner_compute = (
            kv_cache_dtype == "fp8_ds_mla"
            and current_platform.is_device_capability_family(100)
        )
        self.fp8_decode_padded_heads = self._compute_fp8_decode_padded_heads(num_heads)

        vllm_config = get_current_vllm_config()
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        q_concat_shape = (max_tokens, num_heads, head_size)
        if is_quantized_kv_cache(kv_cache_dtype):
            assert kv_cache_dtype == "fp8_ds_mla", (
                "FlashMLA Sparse Attention backend fp8 only supports "
                "fp8_ds_mla kv-cache dtype"
            )

        if kv_cache_dtype == "fp8_ds_mla":
            # Reserve workspace during initialization
            assert vllm_config is not None and vllm_config.model_config is not None
            prefill_workspace_size = get_prefill_workspace_size(
                vllm_config.model_config.max_model_len
            )
            self.prefill_workspace_shape = (prefill_workspace_size, head_size)
            reserve_owner_materialization_padded_q = (
                envs.VLLM_USE_PCP_OWNER_HISTORY
                and vllm_config.parallel_config.prefill_context_parallel_size > 1
                and get_owner_prefill_mode() == "materialize"
                and num_heads % self.prefill_padding != 0
            )
            q_workspace_shape = (
                (max_tokens, self.prefill_padding, head_size)
                if reserve_owner_materialization_padded_q
                else q_concat_shape
            )
            q_workspace, self.prefill_bf16_workspace = (
                current_workspace_manager().get_simultaneous(
                    (q_workspace_shape, torch.bfloat16),
                    (self.prefill_workspace_shape, torch.bfloat16),
                )
            )
            if reserve_owner_materialization_padded_q:
                self.q_padded_buffer = q_workspace
                # These aliases intentionally overlap but are never live
                # together: direct FP8 writes the contiguous packed view,
                # while materialized prefill writes the padded view. Flattening
                # before reshaping keeps the direct view contiguous.
                self.q_concat_buffer = q_workspace.view(-1)[
                    : max_tokens * num_heads * head_size
                ].view(q_concat_shape)
            else:
                self.q_padded_buffer = None
                self.q_concat_buffer = q_workspace
        else:
            self.q_padded_buffer = None
            (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
                (q_concat_shape, torch.bfloat16),
            )

    def _forward_owner_compute(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, None]:
        """Run packed FP8 FlashMLA against one owner-local history shard."""
        if not current_platform.is_device_capability_family(100):
            raise RuntimeError(
                "Owner-local FlashMLA requires SM100 per-row top-k length support."
            )
        if attn_metadata.num_decodes != 0:
            raise RuntimeError(
                "Owner-local FlashMLA supports pure-prefill batches only."
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
                "Owner-local FlashMLA requires a validated padded source stride."
            )
        if q.shape[0] != source_stride:
            raise RuntimeError(
                "Owner-local FlashMLA Q rows do not match the fixed source "
                f"stride: q_rows={q.shape[0]}, stride={source_stride}."
            )
        w_uv, v_head_dim = self._validate_owner_value_projection(q, layer)
        assert self.topk_indices_buffer is not None
        if self.topk_indices_buffer.shape[0] < source_stride:
            raise RuntimeError(
                "Owner-local FlashMLA top-k buffer is shorter than the fixed "
                f"source stride: rows={self.topk_indices_buffer.shape[0]}, "
                f"stride={source_stride}."
            )

        num_actual = attn_metadata.num_actual_tokens
        if not 0 <= num_actual <= source_stride:
            raise RuntimeError(
                "Owner-local FlashMLA actual-token count exceeds the padded "
                f"source stride: actual={num_actual}, stride={source_stride}."
            )
        if attn_metadata.req_id_per_token.shape[0] != num_actual:
            raise RuntimeError(
                "Owner-local FlashMLA request IDs do not match actual Q rows."
            )

        peer_block_stride = getattr(layer, "pcp_peer_block_stride", None)
        if not isinstance(peer_block_stride, int) or peer_block_stride <= 0:
            raise RuntimeError(
                "Owner-local FlashMLA requires a peer-cache block stride."
            )
        topk = self.topk_indices_buffer.shape[1]
        if topk != attn_metadata.topk_tokens:
            raise RuntimeError(
                "Owner-local FlashMLA top-k width does not match attention "
                f"metadata: buffer={topk}, metadata={attn_metadata.topk_tokens}."
            )
        owner_peer_slot_cache = getattr(layer, "owner_peer_slot_cache", None)
        if owner_peer_slot_cache is None:
            raise RuntimeError(
                "Owner-local FlashMLA requires the shared peer-slot cache."
            )

        routed_q = dcp_group.all_gather(q, dim=0)
        expected_rows = dcp_group.world_size * source_stride
        if routed_q.shape[0] != expected_rows:
            raise RuntimeError(
                "Owner-local FlashMLA gathered queries do not have fixed "
                "rank-major source shape."
            )

        def build_owner_local_slots(
            padded_peer_slots: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            routed_peer_slots = dcp_group.all_gather(padded_peer_slots, dim=0)
            if routed_peer_slots.shape[0] != expected_rows:
                raise RuntimeError(
                    "Owner-local FlashMLA gathered slots do not have fixed "
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

        fp8_metadata = attn_metadata.fp8_extra_metadata
        if not isinstance(fp8_metadata, FlashMLASparseMetadata.FP8KernelMetadata):
            raise RuntimeError(
                "Owner-local FlashMLA requires mixed FP8 kernel metadata."
            )
        # FlashMLA scheduling depends on the per-owner valid counts. Reuse it
        # only across shared attention layers consuming this exact Indexer
        # epoch; OwnerPeerSlotCache invalidates it on the next top-k refresh.
        scheduler_metadata = owner_peer_slot_cache.get_or_build_owner_local_metadata(
            (
                "flashmla",
                expected_rows,
                self.fp8_decode_padded_heads,
                topk,
            ),
            lambda: get_mla_metadata()[0],
        )
        assert isinstance(scheduler_metadata, FlashMLASchedMeta)
        owner_kernel_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=scheduler_metadata,
            dummy_block_table=fp8_metadata.dummy_block_table,
            cache_lens=fp8_metadata.cache_lens,
        )
        output, lse = self._fp8_flash_mla_kernel(
            q=routed_q.unsqueeze(1),
            kv_c_and_k_pe_cache=kv_cache,
            topk_indices=local_slots.unsqueeze(1),
            kernel_metadata=owner_kernel_metadata,
            topk_length=local_selected_counts,
        )
        output = output.reshape(expected_rows, self.num_heads, self.kv_lora_rank)
        # W_UV is head-local and linear, so it commutes with the cross-owner
        # LSE-weighted sum. Projecting each partial first reduces GLM-5.2's
        # correction and reduce-scatter width from 512 to 256. This changes
        # BF16 rounding order, so the path is guarded by the explicit output
        # contract and covered by engine-level numerical qualification.
        projected_output = output.new_empty((expected_rows, self.num_heads, v_head_dim))
        torch.bmm(
            output.transpose(0, 1),
            w_uv,
            out=projected_output.transpose(0, 1),
        )
        output = projected_output
        lse = self._normalize_owner_lse(
            lse,
            batch_size=expected_rows,
            seq_len=1,
            num_heads=self.num_heads,
        )
        empty_rows = local_selected_counts == 0
        lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))

        from vllm.v1.attention.ops.common import cp_lse_ag_out_rs_batch

        local_output = cp_lse_ag_out_rs_batch(
            output,
            lse,
            dcp_group,
            is_lse_base_on_e=True,
        )
        if local_output.shape[0] != source_stride:
            raise RuntimeError(
                "Owner-local FlashMLA reduce-scatter returned an invalid source "
                f"stride: {local_output.shape[0]} != {source_stride}."
            )
        return local_output[:num_actual], None

    def _validate_owner_value_projection(
        self,
        q: torch.Tensor,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, int]:
        """Validate the replicated MLA value projection used by each owner."""
        w_uv = getattr(layer, "W_UV", None)
        v_head_dim = getattr(layer, "v_head_dim", None)
        expected_prefix = (self.num_heads, self.kv_lora_rank)
        if (
            not isinstance(w_uv, torch.Tensor)
            or w_uv.ndim != 3
            or tuple(w_uv.shape[:2]) != expected_prefix
            or not isinstance(v_head_dim, int)
            or v_head_dim <= 0
            or w_uv.shape[2] != v_head_dim
        ):
            actual_shape = tuple(w_uv.shape) if isinstance(w_uv, torch.Tensor) else None
            raise RuntimeError(
                "Owner-local FlashMLA requires W_UV with shape "
                f"({self.num_heads}, {self.kv_lora_rank}, v_head_dim), got "
                f"W_UV={actual_shape} and v_head_dim={v_head_dim}."
            )
        if w_uv.device != q.device or w_uv.dtype != q.dtype:
            raise RuntimeError(
                "Owner-local FlashMLA requires W_UV to match the query device "
                f"and dtype, got W_UV=({w_uv.device}, {w_uv.dtype}) and "
                f"query=({q.device}, {q.dtype})."
            )
        return w_uv, v_head_dim

    @staticmethod
    def _normalize_owner_lse(
        lse: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        num_heads: int,
    ) -> torch.Tensor:
        """Normalize FlashMLA variants to the DCP reducer's [rows, heads]."""
        if lse.shape[:2] == (batch_size, seq_len) and lse.shape[2] >= num_heads:
            return lse[:, :, :num_heads].reshape(-1, num_heads)
        if (
            lse.shape[0] == batch_size
            and lse.shape[1] >= num_heads
            and lse.shape[2] == seq_len
        ):
            return lse[:, :num_heads, :].transpose(1, 2).reshape(-1, num_heads)
        raise RuntimeError(
            "Unexpected owner-local FlashMLA LSE shape: "
            f"{tuple(lse.shape)}, expected ({batch_size}, {seq_len}, H) or "
            f"({batch_size}, H, {seq_len}) with H >= {num_heads}."
        )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill). req_id_per_token covers the whole batch; slice it
        # to the MQA tokens (q may exclude prefill tokens routed to dense MHA).
        topk_indices, topk_length = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[: topk_indices.shape[0]],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        return self._bf16_flash_mla_kernel(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            topk_length,
        )

    def _forward_fp8_kv_separate_prefill_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        fp8_metadata = attn_metadata.fp8_extra_metadata
        assert isinstance(fp8_metadata, FlashMLASparseMetadata.FP8SeparatePrefillDecode)
        num_decodes = fp8_metadata.num_decodes
        num_mqa_tokens = q.shape[0]
        num_decode_tokens = fp8_metadata.num_decode_tokens
        num_prefill_tokens = num_mqa_tokens - num_decode_tokens
        assert num_prefill_tokens in (0, fp8_metadata.num_prefill_tokens), (
            "FP8 sparse MLA expects either the decode subset or the full batch"
        )

        prefill_request_ids = None
        prefill_workspace_starts = None
        has_prefill_workspace = False
        if num_prefill_tokens > 0:
            assert fp8_metadata.prefill is not None
            prefill_request_ids = fp8_metadata.prefill.request_ids
            prefill_workspace_starts = fp8_metadata.prefill.workspace_starts
            has_prefill_workspace = True

        use_owner_history = getattr(layer, "pcp_owner_history_direct", False)
        if use_owner_history:
            peer_block_stride = getattr(layer, "pcp_peer_block_stride", None)
            if not isinstance(peer_block_stride, int) or peer_block_stride <= 0:
                raise RuntimeError(
                    "Owner-sharded FlashMLA requires a peer-cache block stride."
                )
            owner_peer_slot_cache = getattr(layer, "owner_peer_slot_cache", None)
            if owner_peer_slot_cache is None:
                raise RuntimeError(
                    "Owner-sharded FlashMLA requires the shared peer-slot cache."
                )

        if not use_owner_history or num_prefill_tokens > 0:
            # Convert per-request indices to global slots (decode) or workspace
            # offsets (prefill). Prefill workspace starts are chunk-relative.
            topk_indices, topk_length = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[: topk_indices.shape[0]],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                HAS_PREFILL_WORKSPACE=has_prefill_workspace,
                prefill_workspace_request_ids=prefill_request_ids,
                prefill_workspace_starts=prefill_workspace_starts,
                return_valid_counts=True,
            )
        else:
            topk_indices, topk_length = owner_peer_slot_cache.get(
                topk_indices.shape[0],
                attn_metadata.block_table,
                dcp_size=self.dcp_world_size,
                blocks_per_peer=peer_block_stride,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                block_size=attn_metadata.block_size,
            )

        fp8_metadata = attn_metadata.fp8_extra_metadata
        assert isinstance(fp8_metadata, FlashMLASparseMetadata.FP8SeparatePrefillDecode)

        def _fp8_decode(
            q: torch.Tensor,
            topk_indices: torch.Tensor,
        ) -> torch.Tensor:
            # Reshape q: (num_decode_tokens, num_heads, head_dim)
            #         -> (num_decodes, seq_len, num_heads, head_dim)
            q = reshape_query_for_spec_decode(q, num_decodes)
            seq_len = q.shape[1]
            # Reshape topk_indices: (num_decode_tokens, topk)
            #                    -> (num_decodes, seq_len, topk)
            topk_indices = topk_indices.view(num_decodes, seq_len, -1)
            assert fp8_metadata.decode is not None
            attn_out, _ = self._fp8_flash_mla_kernel(
                q=q,
                kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
                topk_indices=topk_indices,
                kernel_metadata=fp8_metadata.decode.kernel_metadata,
            )
            # Reshape output: (num_decodes, seq_len, num_heads, head_dim_v)
            #              -> (num_decode_tokens, num_heads, head_dim_v)
            return reshape_attn_output_for_spec_decode(attn_out)

        # Pure decode: direct call without allocation
        if num_decode_tokens > 0 and num_prefill_tokens == 0:
            assert fp8_metadata.decode is not None
            attn_out = _fp8_decode(q, topk_indices)
        else:
            # Mixed or pure prefill: allocate output tensor
            attn_out = q.new_empty(
                (num_mqa_tokens, self.num_heads, self.kv_lora_rank),
                dtype=q.dtype,
                device=q.device,
            )

            if num_decode_tokens > 0:
                attn_out[:num_decode_tokens] = _fp8_decode(
                    q[:num_decode_tokens],
                    topk_indices[:num_decode_tokens],
                )

            assert fp8_metadata.prefill is not None
            for chunk in fp8_metadata.prefill.chunks:
                chunk_workspace = self.prefill_bf16_workspace[: chunk.chunk_tot_seqlen]
                read_block_table = chunk.block_table
                if use_owner_history:
                    peer_block_stride = getattr(layer, "pcp_peer_block_stride", None)
                    if not isinstance(peer_block_stride, int) or peer_block_stride <= 0:
                        raise RuntimeError(
                            "Owner-sharded FlashMLA materialization requires a "
                            "positive peer-cache block stride."
                        )
                    peer_block_table_key = (
                        peer_block_stride,
                        attn_metadata.cp_kv_cache_interleave_size,
                        attn_metadata.block_size,
                    )
                    read_block_table = chunk.pcp_peer_block_table
                    if (
                        read_block_table is None
                        or chunk.pcp_peer_block_table_key != peer_block_table_key
                    ):
                        owner_block_tables = chunk.block_table.unsqueeze(0).expand(
                            self.dcp_world_size, -1, -1
                        )
                        read_block_table = build_rotated_dcp_peer_block_table(
                            owner_block_tables,
                            local_rank=0,
                            peer_block_stride=peer_block_stride,
                            cp_kv_cache_interleave_size=(
                                attn_metadata.cp_kv_cache_interleave_size
                            ),
                            block_size=attn_metadata.block_size,
                        )
                        chunk.pcp_peer_block_table = read_block_table
                        chunk.pcp_peer_block_table_key = peer_block_table_key
                ops.cp_gather_and_upconvert_fp8_kv_cache(
                    kv_c_and_k_pe_cache,
                    chunk_workspace,
                    read_block_table,
                    chunk.workspace_starts,
                    len(chunk.block_table),
                )

                chunk_q = q[chunk.tokens_slice]
                chunk_topk_indices_workspace = topk_indices[chunk.tokens_slice]
                chunk_topk_length = topk_length[chunk.tokens_slice]

                attn_out[chunk.tokens_slice] = self._bf16_flash_mla_kernel(
                    chunk_q,
                    chunk_workspace,
                    chunk_topk_indices_workspace,
                    chunk_topk_length,
                )

        return attn_out

    def _forward_fp8_kv_mixed_batch(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """Mixed batch FP8 forward path that treats all tokens as one batch.

        This is equivalent to main branch's approach and avoids the BF16
        prefill kernel which has head padding overhead when num_heads is small.
        Used when use_mixed_batch is True.
        """
        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill).
        if getattr(layer, "pcp_owner_history_direct", False):
            peer_block_stride = getattr(layer, "pcp_peer_block_stride", None)
            if not isinstance(peer_block_stride, int) or peer_block_stride <= 0:
                raise RuntimeError(
                    "Owner-sharded FlashMLA requires a peer-cache block stride."
                )
            owner_peer_slot_cache = getattr(layer, "owner_peer_slot_cache", None)
            if owner_peer_slot_cache is None:
                raise RuntimeError(
                    "Owner-sharded FlashMLA requires the shared peer-slot cache."
                )
            topk_indices, _ = owner_peer_slot_cache.get(
                topk_indices.shape[0],
                attn_metadata.block_table,
                dcp_size=self.dcp_world_size,
                blocks_per_peer=peer_block_stride,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                block_size=attn_metadata.block_size,
            )
        else:
            topk_indices = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[: topk_indices.shape[0]],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            )

        assert attn_metadata.fp8_extra_metadata is not None
        assert isinstance(
            attn_metadata.fp8_extra_metadata, FlashMLASparseMetadata.FP8KernelMetadata
        )
        fp8_metadata = attn_metadata.fp8_extra_metadata

        _attn_out, _ = self._fp8_flash_mla_kernel(
            q=q.unsqueeze(0),  # unsqueeze to add batch_dim: (T, H, D) -> (1, T, H, D)
            kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
            topk_indices=topk_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
            kernel_metadata=fp8_metadata,
        )

        # Output is (1, T, H, D_v), squeeze back to (T, H, D_v)
        return _attn_out.squeeze(0)

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
        topk_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q shape: (batch, seq_len, num_heads, head_dim)
        actual_num_heads = q.size(2)
        padded_num_heads = self.fp8_decode_padded_heads

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=topk_indices,
            topk_length=topk_length,
            softmax_scale=self.softmax_scale,
        )

        # Slice output back to actual head count if we padded
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]

        return out, lse

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # NOTE(Chen): kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell
        if self.num_heads % self.prefill_padding != 0:
            assert self.prefill_padding % self.num_heads == 0
            logger.warning_once(
                f"Padding num_heads from {self.num_heads} to "
                f"{self.prefill_padding} for BF16 sparse prefill kernel"
            )
            reserved_q_padded = getattr(self, "q_padded_buffer", None)
            reserved_q_start = None
            if (
                reserved_q_padded is not None
                and reserved_q_padded.shape[0] >= q.shape[0]
                and reserved_q_padded.shape[1:]
                == (
                    self.prefill_padding,
                    q.shape[2],
                )
                and q.stride()
                == reserved_q_padded[: q.shape[0], : self.num_heads].stride()
                and q.untyped_storage().data_ptr()
                == reserved_q_padded.untyped_storage().data_ptr()
            ):
                storage_delta = q.storage_offset() - reserved_q_padded.storage_offset()
                row_stride = reserved_q_padded.stride(0)
                if storage_delta >= 0 and storage_delta % row_stride == 0:
                    candidate_start = storage_delta // row_stride
                    if (
                        candidate_start + q.shape[0] <= reserved_q_padded.shape[0]
                        and q.data_ptr()
                        == reserved_q_padded[
                            candidate_start, : self.num_heads
                        ].data_ptr()
                    ):
                        reserved_q_start = candidate_start
            if reserved_q_start is not None:
                q_padded = reserved_q_padded[
                    reserved_q_start : reserved_q_start + q.shape[0]
                ]
                q_padded[:, self.num_heads :, :].zero_()
            else:
                # Non-owner callers may provide an external query. Keep that
                # fallback correct while owner-history materialization uses the
                # bounded initialization-time workspace above.
                q_padded = q.new_zeros((q.shape[0], self.prefill_padding, q.shape[2]))
                q_padded[:, : self.num_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            topk_length=topk_length,
        )[0]

        output = output[:, : self.num_heads, :]
        return output

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            use_reserved_padded_q = (
                getattr(layer, "pcp_owner_history_direct", False)
                and self.kv_cache_dtype == "fp8_ds_mla"
                and not attn_metadata.fp8_use_mixed_batch
                and attn_metadata.num_decodes == 0
                and self.num_heads % self.prefill_padding != 0
            )
            if use_reserved_padded_q:
                assert self.q_padded_buffer is not None
                q_padded = self.q_padded_buffer[: ql_nope.shape[0]]
                q = q_padded[:, : self.num_heads, :]
            else:
                q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = q.shape[0]

        if getattr(layer, "pcp_owner_compute", False):
            return self._forward_owner_compute(
                q,
                kv_c_and_k_pe_cache,
                attn_metadata,
                layer,
            )

        # PCP can assign no local query tokens to a rank for short prefills.
        # FlashMLA rejects a zero sequence dimension, while the specialized
        # DeepSeek-V3.2 layer can safely continue with an empty local output.
        if num_actual_toks == 0:
            return q.new_empty((0, self.num_heads, self.kv_lora_rank)), None

        # Get topk indices
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"

        if not use_fp8_cache:
            attn_out = self._forward_bf16_kv(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        elif attn_metadata.fp8_use_mixed_batch:
            attn_out = self._forward_fp8_kv_mixed_batch(
                q,
                kv_c_and_k_pe_cache,
                topk_indices,
                attn_metadata,
                layer,
            )
        else:
            attn_out = self._forward_fp8_kv_separate_prefill_decode(
                q,
                kv_c_and_k_pe_cache,
                topk_indices,
                attn_metadata,
                layer,
            )

        return attn_out, None
