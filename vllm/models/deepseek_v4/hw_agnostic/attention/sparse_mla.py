# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.model_executor.hw_agnostic.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.model_executor.hw_agnostic.v1.kv_cache_interface import AttentionSpec
from vllm.models.deepseek_v4.hw_agnostic.attention._metadata_utils import (
    split_decodes_and_prefills,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

# Align C128A topk row width to 128 to satisfy OOT sparse kernels that key
# divisibility off h_q ∈ {64, 128}.
_C128A_TOPK_ALIGNMENT = 128


@triton.jit
def _compressed_slot_mapping_kernel(
    slot_mapping_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len

    for i in range(0, query_len, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < query_len

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
        is_valid = (pos + 1) % COMPRESS_RATIO == 0
        pos_after_compress = pos // COMPRESS_RATIO

        block_ids = pos_after_compress // block_size
        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + block_ids,
            mask=mask & is_valid,
        )
        slot_ids = block_numbers * block_size + pos_after_compress % block_size
        slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)


def _get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        # Pre-fill with -1 so padded/invalid positions don't yield bogus block ids.
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    num_reqs = block_table.shape[0]
    _compressed_slot_mapping_kernel[(num_reqs,)](
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_size,
        compress_ratio,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
    )
    return slot_mapping


class DeepseekV4HWAgnosticBackend(AttentionBackend):
    """Spec carrier for the DSv4 sparse-MLA cache.

    Hands the runner a metadata builder, a KV cache shape and a kernel
    block-size hint. Compute lives in ``DeepseekV4MLAAttention``.
    """

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_HW_AGNOSTIC"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4HWAgnosticMetadataBuilder"]:
        return DeepseekV4HWAgnosticMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            return (num_blocks, block_size, 584)
        return (num_blocks, block_size, head_size)


@dataclass
class DeepseekV4HWAgnosticMetadata(AttentionMetadata):
    block_table: torch.Tensor
    block_size: int
    # Compressed-cache slot ids (one per token, -1 for non-boundary positions).
    # Consumed by ``compress_norm_rope_store_triton`` to address the KV write.
    slot_mapping: torch.Tensor

    # Pre-computed C128A metadata (compress_ratio == 128 only).
    # Decode: global slot ids + valid-entry counts (fused from positions).
    c128a_global_decode_topk_indices: torch.Tensor | None = None
    c128a_decode_topk_lens: torch.Tensor | None = None
    # Prefill: local topk indices (used by combine_topk_swa_indices).
    c128a_prefill_topk_indices: torch.Tensor | None = None


class DeepseekV4HWAgnosticMetadataBuilder(
    AttentionMetadataBuilder[DeepseekV4HWAgnosticMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        assert hasattr(self.kv_cache_spec, "compress_ratio")
        self.compress_ratio = self.kv_cache_spec.compress_ratio

        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        # Compressed-cache slot ids (one per token; -1 for non-boundary
        # positions). Consumed by the compressor's KV-write kernel.
        if self.compress_ratio > 1:
            self.compressed_slot_mapping_buffer = torch.empty(
                max_num_batched_tokens, dtype=torch.int64, device=device
            )

        # Pre-allocate C128A topk buffers for CUDA-graph address stability.
        if self.compress_ratio == 128:
            c128a_max_compressed = cdiv(
                self.model_config.max_model_len, self.compress_ratio
            )
            c128a_max_compressed = (
                cdiv(c128a_max_compressed, _C128A_TOPK_ALIGNMENT)
                * _C128A_TOPK_ALIGNMENT
            )
            # Passed to the C128A kernel as max_compressed_tokens. The kernel
            # default (8192) iterates past row width and spills writes into
            # adjacent rows when buffer stride is smaller.
            self.c128a_max_compressed = c128a_max_compressed
            self.req_id_per_token_buffer = torch.empty(
                (max_num_batched_tokens,), dtype=torch.int32, device=device
            )
            self.c128a_global_decode_buffer = torch.empty(
                (max_num_batched_tokens, c128a_max_compressed),
                dtype=torch.int32,
                device=device,
            )
            self.c128a_decode_lens_buffer = torch.empty(
                max_num_batched_tokens, dtype=torch.int32, device=device
            )
            self.c128a_prefill_buffer = torch.empty(
                (max_num_batched_tokens, c128a_max_compressed),
                dtype=torch.int32,
                device=device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4HWAgnosticMetadata:
        cm = common_attn_metadata

        slot_mapping = cm.slot_mapping
        if self.compress_ratio > 1:
            slot_mapping = _get_compressed_slot_mapping(
                cm.num_actual_tokens,
                cm.query_start_loc,
                cm.seq_lens,
                cm.block_table_tensor.clamp(min=0),
                int(self.kv_cache_spec.storage_block_size),
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )

        c128a_fields: dict[str, torch.Tensor | None] = {}
        if self.compress_ratio == 128:
            num_tokens = cm.num_actual_tokens
            starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
            seg_lengths = np.diff(starts)
            req_id_per_token_np = np.repeat(
                np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
            )
            self.req_id_per_token_buffer.fill_(0)
            self.req_id_per_token_buffer[: req_id_per_token_np.shape[0]].copy_(
                torch.from_numpy(req_id_per_token_np), non_blocking=True
            )
            req_id_per_token = self.req_id_per_token_buffer[:num_tokens]
            c128a_fields = self._build_c128a_metadata(cm, req_id_per_token)

        return DeepseekV4HWAgnosticMetadata(
            block_table=cm.block_table_tensor,
            block_size=self.kv_cache_spec.block_size,
            slot_mapping=slot_mapping,
            c128a_global_decode_topk_indices=c128a_fields.get(
                "c128a_global_decode_topk_indices"
            ),
            c128a_decode_topk_lens=c128a_fields.get("c128a_decode_topk_lens"),
            c128a_prefill_topk_indices=c128a_fields.get("c128a_prefill_topk_indices"),
        )

    def _build_c128a_metadata(
        self,
        cm: CommonAttentionMetadata,
        req_id_per_token: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        (num_decodes, _, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                cm,
                decode_threshold=self.reorder_batch_threshold or 1,
            )
        )

        num_total = num_decode_tokens + num_prefill_tokens
        if num_total == 0:
            return {}

        assert cm.positions is not None, (
            "positions is required for C128A metadata build"
        )
        block_size = self.kv_cache_spec.block_size // self.compress_ratio
        global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
            cm.positions[:num_total],
            self.compress_ratio,
            num_decode_tokens,
            req_id_per_token,
            cm.block_table_tensor[:num_decodes],
            block_size,
            cm.slot_mapping,
            self.c128a_global_decode_buffer,
            self.c128a_decode_lens_buffer,
            self.c128a_prefill_buffer,
            max_compressed_tokens=self.c128a_max_compressed,
        )

        result: dict[str, torch.Tensor | None] = {}
        if num_decode_tokens > 0:
            result["c128a_global_decode_topk_indices"] = global_decode.view(
                num_decode_tokens, 1, -1
            )
            result["c128a_decode_topk_lens"] = decode_lens
        if num_prefill_tokens > 0:
            result["c128a_prefill_topk_indices"] = prefill_local
        return result


def build_c128a_topk_metadata(
    positions: torch.Tensor,
    compress_ratio: int,
    num_decode_tokens: int,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    global_decode_buffer: torch.Tensor,
    decode_lens_buffer: torch.Tensor,
    prefill_buffer: torch.Tensor,
    max_compressed_tokens: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build C128A topk metadata for both decode and prefill tokens."""
    num_tokens = positions.shape[0]
    num_prefill_tokens = num_tokens - num_decode_tokens

    global_decode = global_decode_buffer[:num_decode_tokens]
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer[:num_prefill_tokens]

    if num_tokens == 0:
        return global_decode, decode_lens, prefill_local

    _build_c128a_topk_metadata_kernel[(num_tokens,)](
        global_decode_buffer,
        global_decode_buffer.stride(0),
        decode_lens_buffer,
        prefill_buffer,
        prefill_buffer.stride(0),
        positions,
        compress_ratio,
        max_compressed_tokens,
        num_decode_tokens,
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        slot_mapping,
        BLOCK_SIZE=1024,
    )
    return global_decode, decode_lens, prefill_local


@triton.jit
def _build_c128a_topk_metadata_kernel(
    # Decode outputs
    global_decode_ptr,
    global_decode_stride,
    decode_lens_ptr,
    # Prefill output
    prefill_local_ptr,
    prefill_local_stride,
    # Inputs
    positions_ptr,
    compress_ratio,
    max_compressed_tokens,
    num_decode_tokens,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    slot_mapping_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    position = tl.load(positions_ptr + token_idx)
    num_compressed = (position + 1) // compress_ratio
    num_compressed = tl.minimum(num_compressed, max_compressed_tokens)
    is_decode = token_idx < num_decode_tokens

    if is_decode:
        is_valid_token = tl.load(slot_mapping_ptr + token_idx) >= 0
        req_idx = tl.load(token_to_req_indices_ptr + token_idx)
        count = tl.zeros((), dtype=tl.int32)
        for i in range(0, max_compressed_tokens, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < max_compressed_tokens
            is_valid = offset < num_compressed

            block_indices = offset // block_size
            block_numbers = tl.load(
                block_table_ptr + req_idx * block_table_stride + block_indices,
                mask=mask & is_valid,
            )
            block_offsets = offset % block_size
            slot_ids = block_numbers * block_size + block_offsets
            slot_ids = tl.where(is_valid, slot_ids, -1)
            tl.store(
                global_decode_ptr + token_idx * global_decode_stride + offset,
                slot_ids,
                mask=mask,
            )
            count += tl.sum(is_valid.to(tl.int32), axis=0)

        tl.store(
            decode_lens_ptr + token_idx,
            tl.where(is_valid_token, count, 0),
        )
    else:
        pfx_idx = token_idx - num_decode_tokens
        for i in range(0, max_compressed_tokens, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < max_compressed_tokens
            tl.store(
                prefill_local_ptr + pfx_idx * prefill_local_stride + offset,
                tl.where(offset < num_compressed, offset, -1),
                mask=mask,
            )
