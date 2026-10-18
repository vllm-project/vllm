# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
    fused_save_compress_norm,
    rope_quant_insert,
    save_ring_rows,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheSpec


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder


@dataclass
class CompressorMetadata:
    # [num_tokens] ring slot of every token: block * capacity + pos % capacity.
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor  # [num_reqs + 1]
    token_to_req_indices: torch.Tensor  # [num_tokens]
    # PCP only: every rank gathers the last row of each rank's local requests.
    gather_tokens: torch.Tensor | None = None  # [rows] this rank's rows
    # [pcp * rows] ring slot of each request's newest row if it is open, else -1.
    ring_write_slots: torch.Tensor | None = None
    # [num_reqs] gathered row preceding each local chunk, or -1 to use the ring.
    prev_row_indices: torch.Tensor | None = None


@triton.jit(do_not_specialize=["block_table_stride", "num_actual_tokens", "num_tokens"])
def _ring_slot_mapping_kernel(
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    token_to_req_ptr,
    positions_ptr,
    num_actual_tokens,
    num_tokens,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < num_actual_tokens
    req = tl.load(token_to_req_ptr + offsets, mask=valid, other=0).to(tl.int64)
    block = tl.load(block_table_ptr + req * block_table_stride, mask=valid, other=0)
    pos = tl.load(positions_ptr + offsets, mask=valid, other=0)
    slot = block.to(tl.int64) * CAPACITY + pos % CAPACITY
    # Block 0 is the null block. A request without a ring block (dummy or padding
    # rows, which the runners fill with the null block) must not write.
    tl.store(
        slot_mapping_ptr + offsets,
        tl.where(valid & (block != 0), slot, -1),
        mask=offsets < num_tokens,
    )


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, CircularBufferSpec)
        self.capacity = self.kv_cache_spec.block_size
        self.pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        max_num_batched_tokens = (
            self.vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.token_to_req_indices = torch.zeros(
            max_num_batched_tokens, dtype=torch.int32, device=self.device
        )
        self.slot_mapping_buffer = torch.empty(
            max_num_batched_tokens, dtype=torch.int64, device=self.device
        )
        # The ring keeps only the open row under PCP, too few for spec decoding.
        assert self.pcp_size == 1 or self.vllm_config.speculative_config is None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        num_tokens = common_attn_metadata.slot_mapping.numel() // self.pcp_size
        positions = common_attn_metadata.positions
        assert positions is not None
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        slot_mapping = self.slot_mapping_buffer[:num_tokens]
        block_table = common_attn_metadata.block_table_tensor
        _ring_slot_mapping_kernel[(triton.cdiv(num_tokens, 256),)](
            slot_mapping,
            block_table,
            block_table.stride(0),
            token_to_req_indices,
            positions,
            common_attn_metadata.num_actual_tokens,
            num_tokens,
            CAPACITY=self.capacity,
            BLOCK=256,
        )
        metadata = CompressorMetadata(
            slot_mapping=slot_mapping,
            query_start_loc=common_attn_metadata.query_start_loc,
            token_to_req_indices=token_to_req_indices,
        )
        if self.pcp_size > 1:
            self._link_pcp_chunks(metadata, common_attn_metadata.num_reqs, positions)
        return metadata

    def _link_pcp_chunks(
        self, metadata: CompressorMetadata, num_reqs: int, positions: torch.Tensor
    ) -> None:
        """Link ratio-2 groups split across PCP chunks and pick the ring writes.

        Every rank gathers ``(ring block, position)`` of each local request's
        last token; the ring block names a request on every rank. Sorted, they
        give each local chunk's predecessor row and each request's newest row.
        """
        num_tokens = metadata.slot_mapping.numel()
        # Equal on every rank, so the gathers line up.
        num_rows = min(num_tokens, 2 * self.vllm_config.scheduler_config.max_num_seqs)
        num_reqs = min(num_reqs, num_rows)
        query_start_loc = metadata.query_start_loc.long()
        starts = query_start_loc[:num_reqs].clamp(max=num_tokens - 1)
        last = (query_start_loc[1 : num_reqs + 1] - 1).clamp(min=0)
        nonempty = query_start_loc[1 : num_reqs + 1] > query_start_loc[:num_reqs]

        def key(slots: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
            return torch.where(slots >= 0, (slots // self.capacity) << 32 | pos, -1)

        local_keys = torch.full((num_rows,), -1, dtype=torch.int64, device=self.device)
        local_keys[:num_reqs] = key(
            torch.where(nonempty, metadata.slot_mapping[last], -1), positions[last]
        )
        sorted_keys, order = get_pcp_group().all_gather(local_keys, dim=0).sort()

        blocks, sorted_positions = sorted_keys >> 32, sorted_keys & 0xFFFFFFFF
        newest = torch.ones_like(sorted_keys, dtype=torch.bool)
        newest[:-1] = blocks[:-1] != blocks[1:]
        is_open = (sorted_keys >= 0) & newest & (sorted_positions % 2 == 0)
        ring_write_slots = torch.empty_like(sorted_keys)
        ring_write_slots[order] = torch.where(
            is_open, blocks * self.capacity + sorted_positions % self.capacity, -1
        )

        first_positions = positions[starts]
        wanted = key(metadata.slot_mapping[starts], first_positions - 1)
        found = torch.searchsorted(sorted_keys, wanted).clamp_(max=order.numel() - 1)
        has_prev = (
            nonempty
            & (wanted >= 0)
            & (first_positions % 2 == 1)
            & (sorted_keys[found] == wanted)
        )

        gather_tokens = torch.zeros_like(local_keys)
        gather_tokens[:num_reqs] = last
        metadata.gather_tokens = gather_tokens
        metadata.ring_write_slots = ring_write_slots
        metadata.prev_row_indices = torch.where(has_prev, order[found], -1)


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    """Per-request ring of the open group's [kv, score] rows (ratio > 1)."""

    def __init__(self, state_dim: int, dtype: torch.dtype, prefix: str):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        # Drafts + bonus token + the previous row, as a power of two >= 8.
        rows_per_step = get_current_vllm_config().num_speculative_tokens + 2
        self.block_size = max(8, 1 << (rows_per_step - 1).bit_length())

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return CircularBufferSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            head_size_v=0,
            dtype=self.dtype,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    """DeepSeek V4.1 KV/score compressor.

    Pools ``compress_ratio`` consecutive tokens into one KV latent with a
    learned softmax gate (ratio 1 has no gate and no pooling). Owns the
    linear, norm and state cache. State saving, compression and RMSNorm share
    one Triton kernel. The emitted latent feeds independent main-cache and
    indexer K kernels, which the attention layer can schedule concurrently.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
    ):
        super().__init__()
        if compress_ratio not in (1, 2):
            raise NotImplementedError(
                "DeepSeek V4.1 compressor supports compress_ratio 1 (full-length "
                f"compressed cache) and 2; got {compress_ratio}. The ratio-4/128 "
                "CuTe-DSL kernels are v4.0-specific and not wired here."
            )
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        # Ratio 1 pools single tokens, so the checkpoint carries no gate.
        self.has_gate = compress_ratio > 1

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        assert self.head_dim == 512 and self.rope_head_dim == 64
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        wkv_wgate_sizes = (
            [self.head_dim, self.head_dim] if self.has_gate else [self.head_dim]
        )
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            wkv_wgate_sizes,
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = (
            CompressorStateCache(
                state_dim=2 * self.head_dim,  # kv_state + score_state
                dtype=torch.float32,
                prefix=f"{prefix}.state_cache",
            )
            if compress_ratio > 1
            else None
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

    def forward(
        self,
        # [num_tokens, (2 if has_gate else 1) * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
    ) -> torch.Tensor | None:
        """Save states and return the BF16 latent for cache insertion and indexing.

        Only valid group-boundary rows are written. Under PCP each rank pools
        its own rows; the last row of every rank's chunks is gathered, both to
        close groups split across chunks and to keep every rank's ring equal.
        """
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None

        prev_rows = prev_row_indices = ring_write_slots = None
        if self.state_cache is None:
            state_cache = query_start_loc = token_to_req_indices = None
            slot_mapping = cast(Any, attn_metadata[self.k_cache_prefix]).slot_mapping
            if self.pcp_size > 1:
                slot_mapping = slot_mapping.chunk(self.pcp_size)[self.pcp_rank]
        else:
            state_metadata = cast(
                CompressorMetadata, attn_metadata[self.state_cache.prefix]
            )
            state_cache = self.state_cache.kv_cache
            slot_mapping = state_metadata.slot_mapping
            query_start_loc = state_metadata.query_start_loc
            token_to_req_indices = state_metadata.token_to_req_indices
            if self.pcp_size > 1:
                assert state_metadata.gather_tokens is not None
                prev_rows = get_pcp_group().all_gather(
                    kv_score[state_metadata.gather_tokens], dim=0
                )
                prev_row_indices = state_metadata.prev_row_indices
                ring_write_slots = state_metadata.ring_write_slots

        latent = torch.empty(
            kv_score.shape[0],
            self.head_dim,
            dtype=torch.bfloat16,
            device=kv_score.device,
        )
        fused_save_compress_norm(
            kv_score,
            positions,
            state_cache,
            slot_mapping,
            query_start_loc,
            token_to_req_indices,
            self.norm.weight,
            self.rms_norm_eps,
            self.compress_ratio,
            latent,
            prev_rows=prev_rows,
            prev_row_indices=prev_row_indices,
        )
        if prev_rows is not None:
            assert state_cache is not None and ring_write_slots is not None
            save_ring_rows(prev_rows, ring_write_slots, state_cache)
        return latent

    def insert_cache(
        self,
        latent: torch.Tensor | None,
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        """Publish compressed main-cache rows after the latent becomes ready."""
        if latent is None:
            return
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache
        # Plain-row per-tensor fp8 caches (FlashInfer) carry the layer's scale;
        # fp8_ds_mla and bf16 rows need none.
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if kv_cache.dtype == torch.float8_e4m3fn
            else None
        )

        rope_quant_insert(
            latent,
            positions,
            rotary_emb.cos_sin_cache,
            kv_cache,
            k_cache_metadata.slot_mapping,
            self.compress_ratio,
            fp8_scale=fp8_scale,
        )
