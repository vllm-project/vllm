# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
    fused_save_compress_norm,
    rope_quant_insert,
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

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
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


@triton.jit
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
    tl.store(
        slot_mapping_ptr + offsets, tl.where(valid, slot, -1), mask=offsets < num_tokens
    )


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, CircularBufferSpec)
        self.capacity = self.kv_cache_spec.block_size
        max_num_batched_tokens = (
            self.vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.token_to_req_indices = torch.zeros(
            max_num_batched_tokens, dtype=torch.int32, device=self.device
        )
        self.slot_mapping_buffer = torch.empty(
            max_num_batched_tokens, dtype=torch.int64, device=self.device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        num_tokens = common_attn_metadata.slot_mapping.numel()
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
        return CompressorMetadata(
            slot_mapping=slot_mapping,
            query_start_loc=common_attn_metadata.query_start_loc,
            token_to_req_indices=token_to_req_indices,
        )


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

        Only valid group-boundary rows are written.
        """
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None

        if self.state_cache is None:
            state_cache = query_start_loc = token_to_req_indices = None
            slot_mapping = cast(Any, attn_metadata[self.k_cache_prefix]).slot_mapping
        else:
            state_metadata = cast(
                CompressorMetadata, attn_metadata[self.state_cache.prefix]
            )
            state_cache = self.state_cache.kv_cache
            slot_mapping = state_metadata.slot_mapping
            query_start_loc = state_metadata.query_start_loc
            token_to_req_indices = state_metadata.token_to_req_indices

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
        )
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
