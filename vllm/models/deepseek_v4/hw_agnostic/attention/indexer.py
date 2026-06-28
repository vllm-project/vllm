# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 indexer metadata builder for the hw_agnostic path.

Strips the upstream V3.2 builder's DeepGEMM scheduling apparatus, the
flatten path (the ``_prepare_uniform_decode_kernel`` + variable-decode
expansion), and the ``schedule_metadata`` field. The hw_agnostic
indexer dispatch goes through pure Triton + pure PyTorch kernels and
accepts any ``next_n``.
"""

from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.model_executor.hw_agnostic.v1.kv_cache_interface import (
    AttentionSpec,
    MLAAttentionSpec,
)
from vllm.models.deepseek_v4.hw_agnostic.attention._metadata_utils import (
    split_decodes_and_prefills,
)
from vllm.models.deepseek_v4.hw_agnostic.attention.sparse_mla import (
    _get_compressed_slot_mapping,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)


@dataclass
class DeepseekV4IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    skip_kv_gather: bool = False


@dataclass
class DeepseekV4IndexerPrefillMetadata:
    chunks: list[DeepseekV4IndexerPrefillChunkMetadata]


@dataclass
class DeepseekV4IndexerDecodeMetadata:
    block_table: torch.Tensor
    # seq_lens: per-token effective context lengths, shape (B, next_n).
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool


@dataclass
class DeepseekV4IndexerMetadata:
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepseekV4IndexerDecodeMetadata | None = None
    prefill: DeepseekV4IndexerPrefillMetadata | None = None


def get_max_prefill_buffer_size(vllm_config: VllmConfig) -> int:
    max_model_len = vllm_config.model_config.max_model_len
    # 40 = (576 * 2 // 132) * 5: indexer prefill (132 B) fits inside the
    # FlashMLA-sparse workspace (576 * 2 B, 5 * max_model_len entries).
    return max_model_len * 40


class DeepseekV4IndexerBackend(AttentionBackend):
    """Spec carrier for the indexer's k_cache.

    Hands the runner a metadata builder, a KV cache shape and a kernel
    block-size hint. Compute lives in ``SparseAttnIndexer``.
    """

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4IndexerMetadataBuilder"]:
        return DeepseekV4IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@triton.jit
def _build_prefill_chunk_metadata_kernel(
    # Inputs
    query_start_loc_ptr,
    uncompressed_seq_lens_ptr,
    cu_compressed_seq_lens_ptr,
    # Outputs
    token_to_seq_ptr,
    cu_compressed_seq_len_ks_ptr,
    cu_compressed_seq_len_ke_ptr,
    query_slice_start,
    query_slice_stop,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_start = tl.load(cu_compressed_seq_lens_ptr + batch_idx)
    seq_end = tl.load(cu_compressed_seq_lens_ptr + batch_idx + 1)
    compressed_seq_len = seq_end - seq_start

    uncompressed_seq_len = tl.load(uncompressed_seq_lens_ptr + batch_idx)
    start_pos = uncompressed_seq_len - query_len

    for i in range(0, query_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        abs_pos = query_start + offset
        mask = (
            (offset < query_len)
            & (abs_pos >= query_slice_start)
            & (abs_pos < query_slice_stop)
        )
        out_pos = abs_pos - query_slice_start

        # Compute cu_seq_len_ks
        tl.store(cu_compressed_seq_len_ks_ptr + out_pos, seq_start, mask=mask)

        # Compute cu_seq_len_ke
        seq_len_per_token = (start_pos + 1 + offset) // COMPRESS_RATIO
        tl.store(
            cu_compressed_seq_len_ke_ptr + out_pos,
            seq_start + seq_len_per_token,
            mask=mask,
        )

    # Compute token_to_seq
    for i in range(0, compressed_seq_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < compressed_seq_len
        tl.store(token_to_seq_ptr + seq_start + offset, batch_idx, mask=mask)


def split_indexer_prefill_chunks(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """
    Split prefill requests into chunks for the sparse indexer, respecting:
    - N constraint: total_seq_lens <= workspace_size (existing O(N) workspace)
    - Logits constraint: M * N * 4 <= max_logits_bytes

    When a single request-level chunk still exceeds the logits budget,
    sub-chunks on the query dimension (M) to bound peak memory.

    Returns list of (req_slice, query_slice) tuples.
    """
    chunks: list[tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max_logits_bytes // 4
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else chunk_m
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


def build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
) -> DeepseekV4IndexerPrefillChunkMetadata | None:
    total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
    if total_seq_lens == 0:
        return None

    num_reqs = end_idx - start_idx
    device = block_table.device
    token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)

    cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    # Assigning to slice avoids cpu sync.
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])

    query_start_loc = (
        query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]
    )

    total_query_len = int(
        (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item()
    )
    if query_slice is not None:
        qs_start = query_slice.start
        qs_stop = query_slice.stop
    else:
        qs_start = 0
        qs_stop = total_query_len
    output_query_len = qs_stop - qs_start

    cu_seq_len_ks = torch.empty(output_query_len, dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty(output_query_len, dtype=torch.int32, device=device)

    _build_prefill_chunk_metadata_kernel[(num_reqs,)](
        query_start_loc,
        uncompressed_seq_lens[start_idx:end_idx],
        cu_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        qs_start,
        qs_stop,
        BLOCK_SIZE=1024,
        COMPRESS_RATIO=compress_ratio,
    )

    token_start = query_start_loc_cpu[start_idx].item()
    if query_slice is not None:
        token_end = token_start + qs_stop
        token_start = token_start + qs_start
        skip_kv_gather = skip_kv_gather or qs_start > 0
    else:
        token_end = query_start_loc_cpu[end_idx].item()

    return DeepseekV4IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seq_len_ks,
        cu_seqlen_ke=cu_seq_len_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
    )


class DeepseekV4IndexerMetadataBuilder(AttentionMetadataBuilder):
    """Builds indexer metadata. Decode tensors are always (B, next_n)."""

    reorder_batch_threshold: int = 1

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        next_n = self.num_speculative_tokens + 1
        self.reorder_batch_threshold += self.num_speculative_tokens

        self.offsets_buffer = torch.arange(
            next_n, device=self.device, dtype=torch.int32
        )
        self.decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )

        self.compress_ratio = 1
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio

        if self.compress_ratio > 1:
            self.compressed_slot_mapping_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            self.expanded_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )

        self._max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )

    def _prepare_decode_tensors(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        decode_lens: torch.Tensor,
        num_decodes: int,
        next_n: int,
        max_decode_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        """Native-path decode-tensor preparation.

        Returns (seq_lens, block_table, decode_lens, batch_size, requires_padding).
        ``seq_lens`` is 2D ``(B, next_n)`` for spec-decode (next_n > 1) and
        1D ``(B,)`` for plain decode.

        ``requires_padding`` is hardcoded to ``False`` here:
        ``split_decodes_and_prefills(require_uniform=True)`` partitions any
        non-uniform query into the prefill side, so the surviving decode
        batch is uniform by construction (``min_decode_len ==
        max_decode_len``). The upstream code keeps this dynamic only because
        the flatten path could absorb non-uniform decodes; we do not have
        that path.
        """
        if next_n > 1:
            assert self.decode_seq_lens_buffer.dim() == 1
            # (B, max_decode_len): token j attends to L - max_decode_len + j + 1
            # KV tokens.
            seq_lens_buffer = self.decode_seq_lens_buffer[
                : num_decodes * max_decode_len
            ].view(num_decodes, max_decode_len)
            seq_lens_buffer[:] = (
                seq_lens.unsqueeze(1)
                - max_decode_len
                + 1
                + self.offsets_buffer[:max_decode_len]
            )
            seq_lens = seq_lens_buffer
        return seq_lens, block_table, decode_lens, num_decodes, False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        slot_mapping = common_attn_metadata.slot_mapping
        block_table = common_attn_metadata.block_table_tensor

        # Always require_uniform=True: the flatten path that handled
        # non-uniform decodes via per-token expansion is gone.
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        compressed_slot_mapping = slot_mapping
        compressed_seq_lens = seq_lens
        if self.compress_ratio > 1:
            compressed_slot_mapping = _get_compressed_slot_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                block_table,
                self.kv_cache_spec.storage_block_size,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
            compressed_seq_lens = seq_lens // self.compress_ratio

        prefill_metadata = None
        if num_prefills > 0:
            # This CPU value is an upper bound for async-spec extend rows. It
            # is safe for chunking/allocation because CUDA metadata below is
            # built from exact device seq_lens and gather ignores the tail.
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            compressed_seq_lens_cpu = (
                seq_lens_cpu // self.compress_ratio
                if self.compress_ratio > 1
                else seq_lens_cpu
            )
            prefill_query_lens_cpu = torch.diff(
                query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
            )
            max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            chunk_specs = split_indexer_prefill_chunks(
                compressed_seq_lens_cpu[num_decodes:],
                prefill_query_lens_cpu,
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes,
            )

            chunks = []
            for req_slice, query_slice in chunk_specs:
                metadata = build_prefill_chunk_metadata(
                    req_slice.start,
                    req_slice.stop,
                    query_start_loc,
                    query_start_loc_cpu,
                    seq_lens,
                    compressed_seq_lens,
                    compressed_seq_lens_cpu,
                    common_attn_metadata.block_table_tensor,
                    self.compress_ratio,
                    query_slice=query_slice,
                    skip_kv_gather=query_slice.start > 0,
                )
                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    chunks.append(metadata)
            prefill_metadata = DeepseekV4IndexerPrefillMetadata(chunks)

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            # Decodes are uniform by construction (``require_uniform=True``
            # above) so all per-request decode lengths are equal; read the
            # value off the CPU mirror to avoid an H2D sync.
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )
            max_decode_len = int(decode_lens_cpu.max().item())
            next_n = 1 + self.num_speculative_tokens

            seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            seq_lens, block_table, decode_lens, batch_size, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=seq_lens,
                    block_table=block_table,
                    decode_lens=decode_lens,
                    num_decodes=num_decodes,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                )
            )

            # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores
            # compressed tokens. Convert uncompressed seq_lens to compressed.
            if self.compress_ratio > 1:
                # ``seq_lens`` aliases ``decode_seq_lens_buffer`` only on the
                # native MTP path (next_n > 1). Otherwise it aliases
                # ``common_attn_metadata.seq_lens`` and must not be mutated.
                if next_n > 1:
                    seq_lens //= self.compress_ratio
                else:
                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens // self.compress_ratio
                    )
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

            # Non-MTP: unsqueeze to (B, 1) so consumers see the (B, next_n)
            # layout consistently with the MTP path.
            if seq_lens.dim() == 1:
                seq_lens = seq_lens.unsqueeze(-1)

            decode_metadata = DeepseekV4IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
            )

        return DeepseekV4IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=compressed_slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )
