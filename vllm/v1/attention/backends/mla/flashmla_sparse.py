# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
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
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    run_length_regions,
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
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
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.pcp_manager import get_current_pcp_schedule
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)


def _neutralize_rows_without_local_kv(
    out: torch.Tensor,
    lse: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    """Rows where this rank owns none of the selected tokens (all indices
    -1) have undefined out/lse; (0, -inf) is the identity element of the
    cross-rank LSE merge, so it drops this rank from those rows.
    """
    empty_rows = (topk_indices == -1).all(dim=-1)
    out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
    lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))


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

In the "nvfp4_ds_mla" format (SM100 only, DeepSeek V3.2 geometry), each
token's KV cache is 352 Bytes, structured as:
-   **First 256 bytes:** 512 `e2m1` NoPE values packed 2/byte (low nibble =
    even element).
-   **Next 64 bytes:** 64 `float8_e4m3` RoPE values. These carry no scale
    factor: `e4m3`'s 4 exponent bits span the RoPE magnitude range unaided.
-   **Last 32 bytes:** 32 `float8_e4m3` NoPE scale factors, one per 16
    elements, stored permuted (an 8x4 -> 4x8 transpose: the scale for element
    block `s` lives at byte `8 * (s & 3) + (s >> 2)`) so that the 8 scales one
    FlashMLA dequant thread needs are contiguous. See the layout comment in
    `csrc/libtorch_stable/cache_kernels.cu`.

"""

# Quantized DS-MLA cache formats served by the FP8/NVFP4 sparse decode kernel
# path (as opposed to the plain bf16 cache). FlashMLA infers which of these the
# cache holds from its bytes-per-token, so nothing else needs to be passed down.
QUANTIZED_DS_MLA_CACHE_FORMATS: frozenset[str] = frozenset(
    {"fp8_ds_mla", "nvfp4_ds_mla"}
)


class FlashMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
        "nvfp4_ds_mla",  # NVFP4 NoPE + FP8 RoPE (SM100 only)
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

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype == "nvfp4_ds_mla" and device_capability.major != 10:
            return (
                f"FLASHMLA_SPARSE only supports the {kv_cache_dtype} kv-cache "
                "dtype on SM100 (Blackwell)"
            )
        return None


@dataclass(frozen=True)
class GatheredPrefillLayout:
    """Layout of one step's DCP-gathered sparse-prefill workspace."""

    region_of_row: np.ndarray
    region_first_row: np.ndarray
    rows_per_rank: np.ndarray
    workspace_starts: np.ndarray
    chunk_bounds: list[tuple[int, int]]


def plan_gathered_prefill(
    row_global_req_idx: np.ndarray,
    scheduled_seq_lens: np.ndarray,
    dcp_world_size: int,
    max_gathered_rows: int,
) -> GatheredPrefillLayout:
    """Lay out the DCP-gathered KV workspace for this step's prefill rows.

    PCP splits a prefill request into ``2W`` chunks and gives this rank two of
    them, so several local rows share one request - and one context. They share
    ONE workspace region.
    """
    assert dcp_world_size > 1, "the gathered layout needs a DCP group to gather"
    assert row_global_req_idx.ndim == 1 and row_global_req_idx.size > 0
    assert np.all(np.diff(row_global_req_idx) >= 0), (
        "PCP+DCP prefill needs rows grouped by ascending global request; got "
        f"{row_global_req_idx.tolist()}. _reorder_segments should guarantee it."
    )

    region_of_row, region_first_row = run_length_regions(row_global_req_idx)
    region_of_row = region_of_row.astype(np.int32)
    region_first_row = region_first_row.astype(np.int32)

    extents = scheduled_seq_lens[row_global_req_idx[region_first_row]].astype(np.int64)
    assert np.all(extents > 0), (
        f"PCP+DCP prefill got an empty scheduled context: {extents.tolist()}"
    )
    rows_per_rank = (extents + dcp_world_size - 1) // dcp_world_size

    per_rank_budget = max_gathered_rows // dcp_world_size
    chunk_bounds = split_prefill_chunks(
        torch.from_numpy(rows_per_rank.astype(np.int32)), per_rank_budget
    )

    workspace_starts = np.zeros(len(rows_per_rank), dtype=np.int64)
    np.cumsum(rows_per_rank[:-1], out=workspace_starts[1:])
    for chunk_start, chunk_stop in chunk_bounds:
        workspace_starts[chunk_start:chunk_stop] -= workspace_starts[chunk_start]

    return GatheredPrefillLayout(
        region_of_row=region_of_row,
        region_first_row=region_first_row,
        rows_per_rank=rows_per_rank,
        workspace_starts=workspace_starts.astype(np.int32),
        chunk_bounds=chunk_bounds,
    )


@dataclass
class GatheredPrefillMetadata:
    """Handles for the DCP-gathered sparse-prefill path."""

    @dataclass
    class Chunk:
        tokens_slice: slice
        block_table: torch.Tensor
        workspace_starts: torch.Tensor
        shard_rows: int

    region_ids: torch.Tensor
    workspace_starts: torch.Tensor
    chunks: list[Chunk]


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

            chunks: list[Chunk]

        num_prefills: int = 0
        num_decodes: int = 0
        num_prefill_tokens: int = 0
        num_decode_tokens: int = 0

        decode: Decode | None = None
        prefill: Prefill | None = None

    fp8_extra_metadata: FP8SeparatePrefillDecode | FP8KernelMetadata | None = None
    fp8_use_mixed_batch: bool = False

    pcp_dcp_kv_gather: bool = False
    gathered_prefill: GatheredPrefillMetadata | None = None


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
        # Varlen decodes are safe under DCP: causality comes from the
        # indexer's top-k indices, not from the kernel metadata.
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=(parallel_config.cp_kv_cache_interleave_size == 1),
        )

        sm_count = num_compute_units(device.index)

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        # FP8 decode kernel only supports h_q = 64 or 128, so we need to pad
        self.fp8_decode_padded_heads = (
            FlashMLASparseImpl._compute_fp8_decode_padded_heads(self.num_heads)
        )

        self.use_fp8_kv_cache = (
            cache_config.cache_dtype in QUANTIZED_DS_MLA_CACHE_FORMATS
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

        self.fp8_use_mixed_batch = self.num_heads < MIN_HEADS_FOR_BF16_PREFILL

        self.pcp_dcp_kv_gather = self.use_pcp and self.dcp_world_size > 1
        self.max_gathered_prefill_rows = 0
        if self.pcp_dcp_kv_gather:
            self.max_gathered_prefill_rows = get_prefill_workspace_size(
                self.model_config.max_model_len
            )

        if parallel_config.decode_context_parallel_size > 1:
            if parallel_config.dcp_comm_backend != "ag_rs":
                raise NotImplementedError(
                    "DCP for FlashMLA sparse is only validated with the "
                    "default 'ag_rs' DCP comm backend; got "
                    f"'{parallel_config.dcp_comm_backend}'"
                )
            if self.pcp_dcp_kv_gather and cache_config.cache_dtype != "fp8_ds_mla":
                raise NotImplementedError(
                    "PCP+DCP sparse prefill gathers the KV through the fp8_ds_mla "
                    f"upconvert; got a {cache_config.cache_dtype} cache"
                )
            if not self.fp8_use_mixed_batch and not self.pcp_dcp_kv_gather:
                raise NotImplementedError(
                    "DCP for FlashMLA sparse is only supported on the "
                    "mixed-batch fp8 path (num_heads < "
                    f"{MIN_HEADS_FOR_BF16_PREFILL}); the separate "
                    "prefill/decode path returns the LSE for decode tokens "
                    "only, while the DCP merge needs it for every token"
                )
            # Head padding (and the tile-scheduler metadata sized from it) is
            # computed from the local head count, but the kernel runs on the
            # DCP-gathered heads.
            if self.use_pcp:
                gathered_num_heads = (
                    self.num_heads * parallel_config.tensor_parallel_size
                    if self.dcp_world_size > self.pcp_world_size
                    else self.num_heads
                )
            else:
                gathered_num_heads = (
                    self.num_heads * parallel_config.decode_context_parallel_size
                )
            gathered_padded_heads = FlashMLASparseImpl._compute_fp8_decode_padded_heads(
                gathered_num_heads
            )
            if self.fp8_decode_padded_heads != gathered_padded_heads:
                raise NotImplementedError(
                    "DCP for FlashMLA sparse requires the local and "
                    "DCP-gathered head counts to pad to the same fp8 decode "
                    f"kernel envelope; got {self.num_heads} local heads "
                    f"(pad to {self.fp8_decode_padded_heads}) vs "
                    f"{gathered_num_heads} gathered heads (pad to "
                    f"{gathered_padded_heads})"
                )

    def _build_gathered_prefill(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        metadata: FlashMLASparseMetadata,
    ) -> GatheredPrefillMetadata | None:
        """Plan the DCP-gathered KV workspace for this step's prefill rows."""
        num_decodes = metadata.num_decodes
        num_prefills = metadata.num_prefills
        if num_prefills == 0:
            return None

        pcp_schedule = get_current_pcp_schedule()
        assert pcp_schedule is not None, (
            "PCP+DCP sparse prefill needs the schedule published by "
            "PCPManager.partition_batch; got None."
        )
        row_global_req_idx = pcp_schedule.local_to_global_req_idx_np[num_decodes:]

        plan = plan_gathered_prefill(
            row_global_req_idx,
            pcp_schedule.seq_lens_np,
            self.dcp_world_size,
            self.max_gathered_prefill_rows,
        )

        qsl_cpu = common_attn_metadata.query_start_loc_cpu.numpy()
        first_prefill_token = int(qsl_cpu[num_decodes])
        last_prefill_token = int(qsl_cpu[num_decodes + num_prefills])
        assert last_prefill_token <= common_attn_metadata.num_actual_tokens, (
            f"prefill rows end at token {last_prefill_token}, past the "
            f"{common_attn_metadata.num_actual_tokens} tokens in the batch"
        )
        region_ids_np = np.full(
            common_attn_metadata.num_actual_tokens, -1, dtype=np.int32
        )
        region_ids_np[first_prefill_token:last_prefill_token] = np.repeat(
            plan.region_of_row,
            np.diff(qsl_cpu[num_decodes : num_decodes + num_prefills + 1]),
        )

        region_ids = async_copy_to_gpu(region_ids_np, device=self.device)
        workspace_starts = async_copy_to_gpu(plan.workspace_starts, device=self.device)
        # One block table row per region: rows of the same request share a
        # request, so they share its blocks - take the first row's.
        region_first_row = (num_decodes + plan.region_first_row).astype(np.int64)
        region_block_tables = common_attn_metadata.block_table_tensor.index_select(
            0, async_copy_to_gpu(region_first_row, device=self.device)
        )

        num_regions = len(plan.region_first_row)
        chunks = []
        for chunk_start, chunk_stop in plan.chunk_bounds:
            first_row = int(plan.region_first_row[chunk_start])
            last_row = (
                int(plan.region_first_row[chunk_stop])
                if chunk_stop < num_regions
                else num_prefills
            )
            chunks.append(
                GatheredPrefillMetadata.Chunk(
                    tokens_slice=slice(
                        int(qsl_cpu[num_decodes + first_row]),
                        int(qsl_cpu[num_decodes + last_row]),
                    ),
                    block_table=region_block_tables[chunk_start:chunk_stop],
                    workspace_starts=workspace_starts[chunk_start:chunk_stop],
                    shard_rows=int(plan.rows_per_rank[chunk_start:chunk_stop].sum()),
                )
            )

        return GatheredPrefillMetadata(
            region_ids=region_ids,
            workspace_starts=workspace_starts,
            chunks=chunks,
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
        return self._build_fp8_mixed_kernel_metadata(num_tokens)

    def _build_fp8_mixed_kernel_metadata(
        self,
        num_tokens: int,
    ) -> "FlashMLASparseMetadata.FP8KernelMetadata":
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

        metadata.fp8_use_mixed_batch = self.fp8_use_mixed_batch
        metadata.pcp_dcp_kv_gather = self.pcp_dcp_kv_gather
        if self.pcp_dcp_kv_gather:
            metadata.gathered_prefill = self._build_gathered_prefill(
                common_attn_metadata, metadata
            )
            # Only the decode rows reach the fp8 kernel on this path, so the
            # tile scheduler is sized for them alone.
            if metadata.num_decode_tokens > 0:
                metadata.fp8_extra_metadata = self._build_fp8_mixed_kernel_metadata(
                    metadata.num_decode_tokens
                )
        elif self.use_fp8_kv_cache:
            if self.fp8_use_mixed_batch:
                metadata.fp8_extra_metadata = self._build_fp8_mixed_decode_prefill(
                    common_attn_metadata
                )
            else:
                metadata.fp8_extra_metadata = self._build_fp8_separate_prefill_decode(
                    common_attn_metadata, metadata
                )

        return metadata


class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
    can_return_lse_for_decode: bool = True

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
        self.fp8_decode_padded_heads = self._compute_fp8_decode_padded_heads(num_heads)

        vllm_config = get_current_vllm_config()
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        q_concat_heads = num_heads
        if not is_quantized_kv_cache(kv_cache_dtype):
            q_concat_heads = (
                (num_heads + self.prefill_padding - 1)
                // self.prefill_padding
                * self.prefill_padding
            )
        q_concat_shape = (max_tokens, q_concat_heads, head_size)
        if is_quantized_kv_cache(kv_cache_dtype):
            assert kv_cache_dtype in QUANTIZED_DS_MLA_CACHE_FORMATS, (
                "FlashMLA Sparse Attention backend only supports the "
                f"{sorted(QUANTIZED_DS_MLA_CACHE_FORMATS)} quantized kv-cache "
                f"dtypes, got {kv_cache_dtype}"
            )

        if self.need_to_return_lse_for_decode and not is_quantized_kv_cache(
            kv_cache_dtype
        ):
            raise NotImplementedError(
                "DCP for FlashMLA sparse requires an fp8_ds_mla kv-cache; "
                "the bf16 sparse path is not supported under DCP."
            )

        self.workspace_slots: list[tuple[str, tuple[tuple[int, ...], torch.dtype]]] = [
            ("q_concat", (q_concat_shape, torch.bfloat16))
        ]
        if kv_cache_dtype in QUANTIZED_DS_MLA_CACHE_FORMATS:
            # Reserve workspace during initialization
            assert vllm_config is not None and vllm_config.model_config is not None
            prefill_workspace_size = get_prefill_workspace_size(
                vllm_config.model_config.max_model_len
            )
            parallel_config = vllm_config.parallel_config
            dcp_size = parallel_config.decode_context_parallel_size
            gathers_kv = (
                parallel_config.prefill_context_parallel_size > 1 and dcp_size > 1
            )
            shard_rows = (
                prefill_workspace_size // dcp_size
                if gathers_kv
                else prefill_workspace_size
            )
            self.prefill_workspace_shape = (shard_rows, head_size)
            self.workspace_slots.append(
                ("prefill_bf16", (self.prefill_workspace_shape, torch.bfloat16))
            )
            if gathers_kv:
                self.workspace_slots.append(
                    (
                        "gathered_kv",
                        ((prefill_workspace_size, head_size), torch.bfloat16),
                    )
                )
        self.prefill_bf16_workspace: torch.Tensor | None = None
        self.gathered_kv_workspace: torch.Tensor | None = None
        self._refresh_workspaces()

    def _refresh_workspaces(self) -> None:
        names, specs = zip(*self.workspace_slots)
        buffers = dict(zip(names, current_workspace_manager().get_simultaneous(*specs)))
        self.q_concat_buffer = buffers["q_concat"]
        self.prefill_bf16_workspace = buffers.get("prefill_bf16")
        self.gathered_kv_workspace = buffers.get("gathered_kv")

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        actual_num_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill). req_id_per_token covers the whole batch; slice it
        # to the MQA tokens (q may exclude prefill tokens routed to dense MHA).
        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        topk_indices, topk_length = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[: topk_indices.shape[0]],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        return self._bf16_flash_mla_kernel(
            q,
            kv_rows,
            topk_indices,
            topk_length,
            actual_num_heads,
        )

    def _prefill_over_gathered_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        prefill_meta: GatheredPrefillMetadata,
        attn_out: torch.Tensor,
    ) -> None:
        """All-gather the KV and attend each prefill row over the whole context."""
        for chunk in prefill_meta.chunks:
            assert self.prefill_bf16_workspace is not None
            shard = self.prefill_bf16_workspace[: chunk.shard_rows]
            ops.cp_gather_and_upconvert_fp8_kv_cache(
                kv_c_and_k_pe_cache,
                shard,
                chunk.block_table,
                chunk.workspace_starts,
                len(chunk.block_table),
            )
            assert self.gathered_kv_workspace is not None
            gathered_kv = self.gathered_kv_workspace[
                : self.dcp_world_size * chunk.shard_rows
            ]
            dist.all_gather_into_tensor(
                gathered_kv, shard, group=get_dcp_group().device_group
            )

            tokens = chunk.tokens_slice
            chunk_indices, chunk_topk_length = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[tokens],
                attn_metadata.block_table,
                topk_indices[tokens],
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                HAS_PREFILL_WORKSPACE=True,
                prefill_workspace_request_ids=prefill_meta.region_ids[tokens],
                prefill_workspace_starts=prefill_meta.workspace_starts,
                prefill_workspace_rank_stride=chunk.shard_rows,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                return_valid_counts=True,
            )
            attn_out[tokens], _ = self._bf16_flash_mla_kernel(
                q[tokens],
                gathered_kv,
                chunk_indices,
                chunk_topk_length,
            )

    def _forward_fp8_kv_pcp_dcp(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PCP + DCP: gather the KV for prefill, keep the shard for decode."""
        assert self.dcp_world_size > 1, (
            "PCP+DCP gathered prefill reached a layer with dcp_world_size=1 "
            "(a sliding-window layer?), whose KV cache is not DCP-sharded"
        )
        prefill_meta = attn_metadata.gathered_prefill
        num_rows = q.shape[0]
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert num_decode_tokens <= num_rows, (
            f"{num_decode_tokens} decode tokens in a batch of {num_rows} rows"
        )
        has_prefill = num_rows > num_decode_tokens and prefill_meta is not None

        decode_out: torch.Tensor | None = None
        decode_lse: torch.Tensor | None = None
        if num_decode_tokens > 0:
            decode_out, decode_lse = self._forward_fp8_kv_mixed_batch(
                q[:num_decode_tokens],
                kv_c_and_k_pe_cache,
                topk_indices[:num_decode_tokens],
                attn_metadata,
            )
            assert decode_lse is not None, (
                "PCP+DCP needs the decode LSE for the cross-rank merge, but "
                "need_to_return_lse_for_decode is False"
            )
            if not has_prefill and num_decode_tokens == num_rows:
                # Decode-only: the kernel already produced the whole answer, so
                # skip allocating the combined buffer and copying into it.
                return decode_out, decode_lse

        attn_out = q.new_empty((num_rows, q.shape[1], self.kv_lora_rank))
        lse = q.new_empty((num_decode_tokens, q.shape[1]), dtype=torch.float32)
        if decode_out is not None:
            assert decode_lse is not None
            attn_out[:num_decode_tokens] = decode_out
            lse.copy_(decode_lse)

        if has_prefill:
            assert prefill_meta is not None
            self._prefill_over_gathered_context(
                q,
                kv_c_and_k_pe_cache,
                topk_indices,
                attn_metadata,
                prefill_meta,
                attn_out,
            )

        return attn_out, lse

    def _forward_fp8_kv_separate_prefill_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
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

        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill).
        # For FP8 cache: prefill uses workspace mapping (upconverted to BF16)
        # For BF16 cache: always use global cache slots (no workspace)
        # prefill_workspace_starts has been adjusted in-place per chunk so
        # prefill indices automatically come out chunk-local
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
                if self.kv_cache_dtype == "fp8_ds_mla":
                    ops.cp_gather_and_upconvert_fp8_kv_cache(
                        kv_c_and_k_pe_cache,
                        chunk_workspace,
                        chunk.block_table,
                        chunk.workspace_starts,
                        len(chunk.block_table),
                    )
                else:
                    ops.cp_gather_and_upconvert_nvfp4_kv_cache(
                        kv_c_and_k_pe_cache.view(torch.uint8),
                        chunk_workspace,
                        chunk.block_table,
                        chunk.workspace_starts,
                        len(chunk.block_table),
                    )

                chunk_q = q[chunk.tokens_slice]
                chunk_topk_indices_workspace = topk_indices[chunk.tokens_slice]
                chunk_topk_length = topk_length[chunk.tokens_slice]

                attn_out[chunk.tokens_slice], _ = self._bf16_flash_mla_kernel(
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Mixed batch FP8 forward path that treats all tokens as one batch.

        This is equivalent to main branch's approach and avoids the BF16
        prefill kernel which has head padding overhead when num_heads is small.
        Used when use_mixed_batch is True.

        The lse is only returned when DCP needs it, otherwise None.
        """
        req_id_per_token = attn_metadata.req_id_per_token[: q.shape[0]]

        if self.dcp_world_size > 1:
            # The indexer emits global token ids; keep this rank's shard and
            # convert to local slots. compact_valid_to_front=False keeps the
            # scattered -1s, which the fp8 kernel masks natively and the
            # empty-row neutralization below relies on. req_id is sliced to
            # topk_indices rows (the converter grids from req_id).
            topk_indices = triton_filter_and_convert_dcp_index(
                req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=attn_metadata.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                compact_valid_to_front=False,
            )
        else:
            # Convert per-request indices to global slots (decode) or workspace
            # offsets (prefill).
            topk_indices = triton_convert_req_index_to_global_index(
                req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            )

        assert attn_metadata.fp8_extra_metadata is not None
        assert isinstance(
            attn_metadata.fp8_extra_metadata,
            FlashMLASparseMetadata.FP8KernelMetadata,
        )
        fp8_metadata = attn_metadata.fp8_extra_metadata

        _attn_out, _lse = self._fp8_flash_mla_kernel(
            q=q.unsqueeze(0),  # unsqueeze to add batch_dim: (T, H, D) -> (1, T, H, D)
            kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
            topk_indices=topk_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
            kernel_metadata=fp8_metadata,
        )
        # Output is (1, T, H, D_v), squeeze back to (T, H, D_v)
        out = _attn_out.squeeze(0)

        if not self.need_to_return_lse_for_decode:
            return out, None

        # Kernel LSE is (1, H, T); the DCP merge consumes (T, H).
        lse = _lse.squeeze(0).transpose(0, 1)
        _neutralize_rows_without_local_kv(out, lse, topk_indices)
        # The head-padding slice above can leave `out` non-contiguous, and the
        # merge feeds it to reduce_scatter.
        return out.contiguous(), lse

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
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
            softmax_scale=self.softmax_scale,
        )

        # Slice output and lse back to actual head count if we padded
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]
            lse = lse[:, :actual_num_heads, :]

        return out, lse

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
        actual_num_heads: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # NOTE(Chen): kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell. Pad from q's head count, not
        # self.num_heads: under DCP the heads are all-gathered before this.
        if actual_num_heads is None:
            actual_num_heads = q.shape[1]
        padded_num_heads = (
            (actual_num_heads + self.prefill_padding - 1)
            // self.prefill_padding
            * self.prefill_padding
        )
        if q.shape[1] < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for BF16 sparse prefill kernel"
            )
            q_padded = q.new_empty((q.shape[0], padded_num_heads, q.shape[2]))
            q_padded[:, :actual_num_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output, _, lse = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            topk_length=topk_length,
        )

        output = output[:, :actual_num_heads, :]
        lse = lse[:, :actual_num_heads]
        return output, lse

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        self._refresh_workspaces()

        # Concatenate q if it's a tuple (ql_nope, q_pe)
        actual_num_heads = self.num_heads
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)
        else:
            actual_num_heads = q.shape[1]

        num_actual_toks = q.shape[0]

        # Get topk indices
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        use_fp8_cache = self.kv_cache_dtype in QUANTIZED_DS_MLA_CACHE_FORMATS

        lse: torch.Tensor | None = None

        if not use_fp8_cache:
            attn_out, bf16_lse = self._forward_bf16_kv(
                q,
                kv_c_and_k_pe_cache,
                topk_indices,
                attn_metadata,
                actual_num_heads,
            )
            if self.need_to_return_lse_for_decode:
                lse = bf16_lse
        elif attn_metadata.pcp_dcp_kv_gather:
            attn_out, lse = self._forward_fp8_kv_pcp_dcp(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        elif attn_metadata.fp8_use_mixed_batch:
            attn_out, lse = self._forward_fp8_kv_mixed_batch(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        else:
            attn_out = self._forward_fp8_kv_separate_prefill_decode(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )

        return attn_out, lse
