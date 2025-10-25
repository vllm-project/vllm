# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionLayer,
    MultipleOf,
)
from vllm.attention.backends.utils import get_mla_dims
from vllm.attention.ops.flashmla import (
    flash_mla_sparse_prefill,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.common import MLACommonBaseImpl
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
    split_prefill_chunks,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)
"""
NOTE: FlashMLA Sparse uses an fp8 cache with the following format

In the "FP8 with scale" format, each token's KV cache is 656 Bytes,
structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512
    `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values.
    The first `float32` is the scale for the first 128 `float8_e4m3` values,
    the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.
"""


class FlashMLASparseBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "fp8_ds_mla"]

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
            # custom storage fromat is 656 bytes
            #  see FlashMLA readme.md for details
            return (num_blocks, block_size, 656)
        else:
            return (num_blocks, block_size, head_size)


@dataclass
class FlashMLASparseMetadataBF16:
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


@dataclass
class FlashMLASparseMetadataFP8(FlashMLASparseMetadataBF16):
    num_prefills: int = 0
    num_decodes: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    @dataclass
    class DecodeMetadata:
        scheduler_metadata: torch.Tensor | None
        num_splits: torch.Tensor
        dummy_block_table: torch.Tensor
        cache_lens: torch.Tensor

    @dataclass
    class PrefillMetadata:
        # Sequence lengths (context + query) for prefill requests
        # Shape: [num_prefill_reqs]
        seq_lens: torch.Tensor

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
        class ChunkMetadata:
            """Metadata for a chunk of prefill requests.

            Prefill requests may be chunked to fit within the fixed workspace size.
            """

            seq_lens: torch.Tensor
            tokens_slice: slice
            block_table: torch.Tensor
            req_start_idx: int
            workspace_starts: torch.Tensor
            chunk_tot_seqlen: int

        chunks: list[ChunkMetadata]

    decode: DecodeMetadata | None = None
    prefill: PrefillMetadata | None = None


FlashMLASparseMetadata = FlashMLASparseMetadataBF16 | FlashMLASparseMetadataFP8


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    prefill_request_id_ptr,  # int32 [num_tokens], -1 for decode, >=0 for prefill
    workspace_starts_ptr,  # int32 [num_prefill_reqs+1] or nullptr
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    HAS_PREFILL: tl.constexpr,
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0
    is_prefill = False
    if HAS_PREFILL:
        prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
        is_prefill = prefill_req_id >= 0
    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block
    base = tl.load(bt_ptr, mask=valid_block & ~is_prefill, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    # Override with prefill output if prefill is enabled
    if HAS_PREFILL:
        workspace_start = tl.load(
            workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
        )
        prefill_out = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_out, out_val)
    out_val = tl.where(is_invalid_tok, -1, out_val)

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.

    When HAS_PREFILL_WORKSPACE is True, prefill tokens are mapped to workspace offsets
    instead of global cache slots. prefill_workspace_request_ids and
    prefill_workspace_starts must be provided.

    prefill_workspace_request_ids: int32 [num_tokens], -1 for decode else
        prefill request index (maps to prefill_workspace_starts)
    prefill_workspace_starts: int32 [num_prefills], 0-indexed workspace
        starts for each prefill request
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )

    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None
        assert prefill_workspace_starts is not None
        assert prefill_workspace_request_ids.dtype == torch.int32
        assert prefill_workspace_starts.dtype == torch.int32

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Prepare prefill pointers
    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None  # for mypy
        assert prefill_workspace_starts is not None  # for mypy
        assert prefill_workspace_request_ids.is_contiguous()
        assert prefill_workspace_starts.is_contiguous()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        prefill_workspace_request_ids,
        prefill_workspace_starts,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        HAS_PREFILL_WORKSPACE,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )
    return out


def get_prefill_workspace_size(max_model_len: int):
    # NOTE(Lucas): 5 is a magic number for controlling the prefill buffer size.
    # May be tuned later.
    # Memory usage: 5 * max_model_len * 576 * 2 bytes
    #   Example: DeepSeek-V3.2 with max_model_len=163840 ->
    #            5 * 163840 * 576 * 2 = ~900 MB
    # This fits nicely below the typical MoE workspace size of >2GB so this is "free"
    return max_model_len * 5


class FlashMLASparseMetadataBuilder(AttentionMetadataBuilder[FlashMLASparseMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        cache_config = vllm_config.cache_config
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.device = device

        # Treat requests with query length <= 1 as decodes to match the
        # DeepGEMM indexer constraint (fp8_paged_mqa_logits only supports next_n <= 2)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        props = torch.cuda.get_device_properties(device)
        sm_count = props.multi_processor_count

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)

        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.use_fp8_kv_cache = cache_config.cache_dtype == "fp8_ds_mla"
        self.topk_tokens_tensor = torch.tensor(
            [self.topk_tokens], device=device, dtype=torch.int32
        )
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=self.device
        )

        # Equation taken from FlashMLA/csrc/pybind.cpp
        h_q, h_k = self.num_heads, 1
        s_q = 1  # inversely proportional to s_q, so s_q = 1 is the largest
        max_num_sm_parts = int(
            max((sm_count // 2) / h_k // (cdiv(h_q // h_k, 2 * 64) * s_q), 1)
        )
        if current_platform.is_device_capability(100):
            max_num_sm_parts *= 2
        self.tile_scheduler_metadata_buffer = torch.empty(
            # TileSchedulerMetaDataSize = 8
            # see: FlashMLA/csrc/params.h
            (max_num_sm_parts, 8),
            dtype=torch.int32,
            device=device,
        )
        self.num_splits_buffer = torch.empty(
            # We pack all the tokens into one batch for sparse attention.
            # Otherwise, we can exceed the sm of `get_mla_metadata`.
            (2,),
            dtype=torch.int32,
            device=device,
        )
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def _build_fp8_extra_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        bf16_metadata: FlashMLASparseMetadataBF16,
    ):
        num_tokens = common_attn_metadata.num_actual_tokens

        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold or 1
            )
        )

        fp8_metadata = FlashMLASparseMetadataFP8(
            **asdict(bf16_metadata),
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
        )

        # Extract prefill sequence lengths (context + query, not just query)
        # Decode requests come first in the batch, prefill requests follow
        prefill_seq_lens = None
        prefill_request_id = None
        prefill_workspace_starts = None
        prefill_chunks = None

        # For pure decode batches, prefill_request_id will be None
        # For mixed batches, it will have -1 for decode and request_id for prefill
        if num_prefills > 0:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            seq_lens = common_attn_metadata.seq_lens
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

            prefill_seq_lens_cpu = seq_lens_cpu[num_decodes:]
            prefill_seq_lens = seq_lens[num_decodes:]

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

                chunk_seq_lens = prefill_seq_lens[chunk_start:chunk_end]
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
                    FlashMLASparseMetadataFP8.PrefillMetadata.ChunkMetadata(
                        seq_lens=chunk_seq_lens,
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

            fp8_metadata.prefill = FlashMLASparseMetadataFP8.PrefillMetadata(
                seq_lens=prefill_seq_lens,
                request_ids=prefill_request_id,
                workspace_starts=prefill_workspace_starts,
                chunks=prefill_chunks,
            )

        if num_decodes > 0:
            tile_scheduler_metadata, num_splits = get_mla_metadata(
                cache_seqlens=self.topk_tokens_tensor,
                num_q_tokens_per_head_k=num_tokens * self.num_heads,
                topk=self.topk_tokens,
                num_heads_q=self.num_heads,
                num_heads_k=1,
                is_fp8_kvcache=True,
            )

            num_sm_parts = tile_scheduler_metadata.size(0)
            # Copy to persistent buffer for full-CG support
            tile_scheduler_metadata_buffer = self.tile_scheduler_metadata_buffer[
                :num_sm_parts
            ]
            tile_scheduler_metadata_buffer.copy_(tile_scheduler_metadata)
            self.num_splits_buffer.copy_(num_splits)

            fp8_metadata.decode = FlashMLASparseMetadataFP8.DecodeMetadata(
                scheduler_metadata=tile_scheduler_metadata_buffer,
                num_splits=self.num_splits_buffer,
                # cache_lens and block_table are basically unused in sparse case
                # but the decode kernel will treat -1 and indices >= cache_lens
                # as invalid so we make sure cache_lens is large enough to not
                # accidentally mark indices invalid, we will use -1 exclusively
                # to mark invalid indices
                cache_lens=self.max_model_len_tensor,
                dummy_block_table=self.dummy_block_table,
            )

        return fp8_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashMLASparseMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]

        bf16_metadata = FlashMLASparseMetadataBF16(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )

        if self.use_fp8_kv_cache:
            return self._build_fp8_extra_metadata(common_attn_metadata, bf16_metadata)
        else:
            return bf16_metadata


class FlashMLASparseImpl(MLACommonBaseImpl[FlashMLASparseMetadata]):
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
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: Optional["Indexer"] = None,
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
        self.softmax_scale = scale
        assert indexer is not None
        self.topk_indices_buffer = indexer.topk_indices_buffer
        self.padding = 128 if current_platform.is_device_capability(100) else 64

        if kv_cache_dtype == "fp8_ds_mla":
            # Reserve workspace during initialization
            vllm_config = get_current_vllm_config()
            assert vllm_config is not None and vllm_config.model_config is not None
            prefill_workspace_size = get_prefill_workspace_size(
                vllm_config.model_config.max_model_len
            )
            self.prefill_workspace_shape = (prefill_workspace_size, head_size)
            (self.prefill_bf16_workspace,) = (
                current_workspace_manager().get_simultaneous(
                    (self.prefill_workspace_shape, torch.bfloat16)
                )
            )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # NOTE(Chen): kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell
        if self.num_heads % self.padding != 0:
            assert self.padding % self.num_heads == 0
            logger.warning_once(
                f"padding num_heads to {self.padding} \
                    due to sparse attn kernel requirement"
            )
            q_padded = q.new_empty((q.shape[0], self.padding, q.shape[2]))
            q_padded[:, : self.num_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = flash_mla_sparse_prefill(
            q, kv_c_and_k_pe_cache, topk_indices, self.softmax_scale
        )[0]
        output = output[:, : self.num_heads, :]
        return output

    def _forward_fp8_kv_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        metadata: FlashMLASparseMetadataFP8.DecodeMetadata,
    ) -> torch.Tensor:
        _attn_out, _ = flash_mla_with_kvcache(
            q=q.unsqueeze(0),  # unsqueeze to add batch_dim
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=metadata.cache_lens,
            tile_scheduler_metadata=metadata.scheduler_metadata,
            num_splits=metadata.num_splits,
            is_fp8_kvcache=True,
            indices=topk_indices.unsqueeze(0),  # unsqueeze to add batch_dim
            softmax_scale=self.softmax_scale,
        )

        return _attn_out

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for MLACommonImpl"
            )

        if attn_metadata is None:
            # Dummy run - no need to allocate buffers
            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs

        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        ql_nope = ql_nope.transpose(0, 1)

        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"

        prefill_request_ids = None
        prefill_workspace_starts = None
        has_prefill_workspace = False
        if (
            isinstance(attn_metadata, FlashMLASparseMetadataFP8)
            and attn_metadata.prefill is not None
        ):
            prefill_request_ids = attn_metadata.prefill.request_ids
            prefill_workspace_starts = attn_metadata.prefill.workspace_starts
            has_prefill_workspace = True

        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill).
        # For FP8 cache: prefill uses workspace mapping (upconverted to BF16)
        # For BF16 cache: always use global cache slots (no workspace)
        # prefill_workspace_starts has been adjusted in-place per chunk so
        # prefill indices automatically come out chunk-local
        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            HAS_PREFILL_WORKSPACE=has_prefill_workspace,
            prefill_workspace_request_ids=prefill_request_ids,
            prefill_workspace_starts=prefill_workspace_starts,
        )

        q = torch.cat([ql_nope, q_pe], dim=-1)

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        if not use_fp8_cache:
            assert isinstance(attn_metadata, FlashMLASparseMetadataBF16)
            attn_out = self._forward_bf16_kv(
                q, kv_cache, topk_indices_global, attn_metadata
            )
        else:
            assert isinstance(attn_metadata, FlashMLASparseMetadataFP8)
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            # Pure decode case: direct call without allocation
            if num_prefill_tokens == 0:
                assert attn_metadata.decode is not None
                attn_out = self._forward_fp8_kv_decode(
                    q, kv_cache, topk_indices_global, attn_metadata.decode
                )
            else:
                assert attn_metadata.prefill is not None
                num_decode_tokens = attn_metadata.num_decode_tokens

                # Mixed or pure prefill: allocate output tensor
                attn_out = q.new_empty(
                    (num_actual_toks, self.num_heads, self.kv_lora_rank),
                    dtype=q.dtype,
                    device=q.device,
                )

                # Fill decode portion if present
                if num_decode_tokens > 0:
                    assert attn_metadata.decode is not None
                    attn_out[:num_decode_tokens] = self._forward_fp8_kv_decode(
                        q[:num_decode_tokens],
                        kv_cache,
                        topk_indices_global[:num_decode_tokens],
                        attn_metadata.decode,
                    )

                for chunk in attn_metadata.prefill.chunks:
                    chunk_workspace = self.prefill_bf16_workspace[
                        : chunk.chunk_tot_seqlen
                    ]
                    ops.cp_gather_and_upconvert_fp8_kv_cache(
                        kv_cache,
                        chunk_workspace,
                        chunk.block_table,
                        chunk.seq_lens,
                        chunk.workspace_starts,
                        len(chunk.block_table),
                    )

                    chunk_q = q[chunk.tokens_slice]
                    chunk_topk_indices_workspace = topk_indices_global[
                        chunk.tokens_slice
                    ]

                    attn_out[chunk.tokens_slice] = self._forward_bf16_kv(
                        chunk_q,
                        chunk_workspace,
                        chunk_topk_indices_workspace,
                        attn_metadata,
                    )

        self._v_up_proj(attn_out, out=output[:num_actual_toks])
        return output
