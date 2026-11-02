# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request ring of the open CSA region's raw indexer rows."""

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheSpec

# One bf16 ``[index_k | index_z]`` row: 2 * 256 * 2 bytes.
_ROW_ELEMENTS = 512
_ROW_BYTES = _ROW_ELEMENTS * 2


@dataclass
class Step5CSATailRingMetadata:
    # [num_tokens] first ring row of the token's request block, or -1.
    base: torch.Tensor
    # [num_tokens] ring row this step stores the token in, or -1 when no later
    # step can replay it.
    slot_mapping: torch.Tensor


@triton.jit
def _ring_metadata_kernel(
    base_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    token_to_req_ptr,
    positions_ptr,
    seq_lens_ptr,
    query_start_loc_ptr,
    num_reqs,
    num_actual_tokens,
    num_tokens,
    CAPACITY: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Graph padding may count as actual tokens; only scheduled rows own a slot.
    num_scheduled = tl.load(query_start_loc_ptr + num_reqs)
    valid = (offsets < num_actual_tokens) & (offsets < num_scheduled)
    req = tl.load(token_to_req_ptr + offsets, mask=valid, other=0).to(tl.int64)
    block = tl.load(block_table_ptr + req * block_table_stride, mask=valid, other=0)
    # Dummy and capture batches map every request to the shared null block 0,
    # which other KV cache groups read as padding; never write it.
    valid = valid & (block > 0)
    pos = tl.load(positions_ptr + offsets, mask=valid, other=0)
    seq_len = tl.load(seq_lens_ptr + req, mask=valid, other=0)
    base = block.to(tl.int64) * ROWS_PER_BLOCK
    keep = valid & (pos >= seq_len - CAPACITY)
    out = offsets < num_tokens
    tl.store(base_ptr + offsets, tl.where(valid, base, -1), mask=out)
    tl.store(
        slot_mapping_ptr + offsets,
        tl.where(keep, base + pos % CAPACITY, -1),
        mask=out,
    )


class Step5CSATailRingMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spec = cast(CircularBufferSpec, self.kv_cache_spec)
        self.capacity = spec.page_size_bytes // _ROW_BYTES
        context = self.vllm_config.compilation_config.static_forward_context
        self.layers = [cast("Step5CSATailRing", context[n]) for n in self.layer_names]
        tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req = torch.zeros(tokens, dtype=torch.int32, device=self.device)
        self.base = torch.empty(tokens, dtype=torch.int64, device=self.device)
        self.slot_mapping = torch.empty(tokens, dtype=torch.int64, device=self.device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Step5CSATailRingMetadata:
        num_tokens = common_attn_metadata.slot_mapping.numel()
        positions = common_attn_metadata.positions
        assert positions is not None
        block_table = common_attn_metadata.block_table_tensor
        _ring_metadata_kernel[(triton.cdiv(num_tokens, 256),)](
            self.base,
            self.slot_mapping,
            block_table,
            block_table.stride(0),
            common_attn_metadata.token_to_req_indices(self.token_to_req),
            positions,
            common_attn_metadata.seq_lens,
            common_attn_metadata.query_start_loc,
            common_attn_metadata.num_reqs,
            common_attn_metadata.num_actual_tokens,
            num_tokens,
            CAPACITY=self.capacity,
            ROWS_PER_BLOCK=self.layers[0].rows_per_block,
            BLOCK=256,
        )
        return Step5CSATailRingMetadata(
            base=self.base[:num_tokens], slot_mapping=self.slot_mapping[:num_tokens]
        )


class Step5CSATailRingBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "STEP5_CSA_TAIL_RING"

    @staticmethod
    def get_builder_cls() -> type[Step5CSATailRingMetadataBuilder]:
        return Step5CSATailRingMetadataBuilder


class Step5CSATailRing(nn.Module, AttentionLayerBase):
    """Raw ``[index_k | index_z]`` rows of each request's open CSA region.

    A region summary is a fixed-order online softmax over its eight rows, so
    the rows are the whole state: the row that closes a region replays the
    other seven from here, and a rejected draft is undone by the next step
    overwriting its row.

    One verify step stores up to ``k + 1`` rows while the next step may still
    replay up to seven earlier rows of the same region, so the ring needs
    ``7 + k`` rows. It is sized to one main KV page so that it shares the
    page size of the other KV cache groups.
    """

    def __init__(
        self,
        *,
        prefix: str,
        main_layer_name: str,
        num_speculative_tokens: int,
        static_forward_context: dict[str, Any],
    ):
        super().__init__()
        if prefix in static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_forward_context[prefix] = self
        self._static_forward_context = static_forward_context
        self.layer_name = prefix
        self.main_layer_name = main_layer_name
        self.required_rows = 7 + num_speculative_tokens
        self.kv_cache = torch.tensor([])
        self.rows = self.kv_cache
        self.capacity = 0
        self.rows_per_block = 0

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        main = self._static_forward_context[self.main_layer_name]
        main_spec = main.get_kv_cache_spec(vllm_config)
        page_bytes = main_spec.page_size_bytes
        if page_bytes % _ROW_BYTES or page_bytes // _ROW_BYTES < self.required_rows:
            raise ValueError(
                f"Step5 CSA tail ring needs {self.required_rows} rows of "
                f"{_ROW_BYTES} bytes in one {page_bytes}-byte KV page; "
                "increase --block-size."
            )
        self.capacity = page_bytes // _ROW_BYTES
        # Declare the main block size so the ring does not coarsen the hybrid
        # prefix-cache alignment; the page holds ``capacity`` 1 KB rows.
        block_size = main_spec.block_size
        return CircularBufferSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=page_bytes // (block_size * torch.bfloat16.itemsize),
            head_size_v=0,
            dtype=torch.bfloat16,
            # Rows are addressed across blocks, which may be strided.
            block_stride_alignment=_ROW_BYTES,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Step5CSATailRingBackend

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        assert kv_cache.stride(0) % _ROW_ELEMENTS == 0
        self.kv_cache = kv_cache
        self.capacity = kv_cache[0].numel() // _ROW_ELEMENTS
        self.rows_per_block = kv_cache.stride(0) // _ROW_ELEMENTS
        # Block b's ring is rows [b * rows_per_block, + capacity).
        self.rows = kv_cache.as_strided(
            (
                (kv_cache.shape[0] - 1) * self.rows_per_block + self.capacity,
                _ROW_ELEMENTS,
            ),
            (_ROW_ELEMENTS, 1),
        )

    def metadata(self) -> Step5CSATailRingMetadata:
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)
        return cast(Step5CSATailRingMetadata, attn_metadata[self.layer_name])
