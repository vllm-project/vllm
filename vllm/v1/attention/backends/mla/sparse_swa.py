# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import ClassVar, cast

import torch

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta, get_mla_metadata
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

# DeepseekV4 decode layer types, keyed by compress_ratio. Each type has a distinct
# (topk, extra_topk, extra_page_block_size) config, so they cannot share a
# FlashMLA tile-scheduler plan. Within a type, all ~60 DeepseekV4 layers share one
# plan per step because b / s_q / h_q / page_block_sizes / topks are identical.
_LAYER_TYPE_SWAONLY = "swaonly"
_LAYER_TYPE_C4A = "c4a"
_LAYER_TYPE_C128A = "c128a"


def _layer_type_for(compress_ratio: int) -> str:
    if compress_ratio <= 1:
        return _LAYER_TYPE_SWAONLY
    if compress_ratio == 4:
        return _LAYER_TYPE_C4A
    if compress_ratio == 128:
        return _LAYER_TYPE_C128A
    raise ValueError(
        f"Unsupported DeepseekV4 compress_ratio={compress_ratio}; "
        "expected 1, 4, or 128."
    )


class DeepseekV4SWACache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.window_size = window_size
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Block size is constrained by tensor sharing between SWA and C4A KV blocks.
        # Since both block types share the same physical tensor, they must use the
        # same page size. The C4A KV block shape [256//4, head_dim] = [64, head_dim]
        # determines the SWA block size of 64 tokens per block.
        # TODO(yifan): make SWA block size automatically determined and configurable.
        self.block_size = 64
        assert self.dtype == torch.uint8

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
            model_version="deepseek_v4",
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekSparseSWABackend


class DeepseekSparseSWABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_SPARSE_SWA"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 256

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @staticmethod
    def get_builder_cls() -> type["DeepseekSparseSWAMetadataBuilder"]:
        if current_platform.is_rocm():
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse_dsv4 import (
                DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
            )

            return DeepseekV4ROCMAiterSparseSWAMetadataBuilder
        return DeepseekSparseSWAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 SWA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class DeepseekSparseSWAMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    seq_lens: torch.Tensor | None = None  # [num_seqs]
    query_start_loc: torch.Tensor | None = None  # [num_seqs + 1]
    query_start_loc_cpu: torch.Tensor | None = None  # [num_seqs + 1]

    is_valid_token: torch.Tensor | None = None  # [num_tokens]
    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    swa_indices: torch.Tensor | None = None  # [num_tokens, 1, window_size]
    swa_lens: torch.Tensor | None = None  # [num_tokens]

    # Number of decode/prefill requests/tokens (batch is reordered: decodes first)
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    # Pre-computed prefill metadata shared across all DeepseekV4 attention layers
    # when using the bf16 prefill fallback path.
    prefill_seq_lens: torch.Tensor | None = None
    prefill_gather_lens: torch.Tensor | None = None

    # Per-layer-type FlashMLA tile-scheduler metadata. All tokens in a scheduler
    # step share one FlashMLASchedMeta per DeepseekV4 layer type.
    tile_sched_swaonly: "FlashMLASchedMeta | None" = None
    tile_sched_c4a: "FlashMLASchedMeta | None" = None
    tile_sched_c128a: "FlashMLASchedMeta | None" = None


class DeepseekSparseSWAMetadataBuilder(AttentionMetadataBuilder):
    """Builds metadata for DeepseekV4 SWA cache.

    This keeps the decode/prefill split counts for compatibility with other
    DeepseekV4 metadata builders, and produces token metadata consumed by
    FlashMLA's direct KV-cache path.

    Supports:
    - Mixed decode/prefill batches
    - MTP (Multi-Token Prediction) where decode has query_len > 1
    - Chunked prefill tokens treated as independent decode-kernel queries
    """

    # Base threshold: query_len <= 1 is decode
    reorder_batch_threshold: int = 1
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.head_size = mla_spec.head_size  # Already considered quantization.
        self.compress_ratio = mla_spec.compress_ratio
        self.block_size = mla_spec.block_size

        # Handle MTP: adjust decode_threshold like the indexer does
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        # With MTP, decode can have query_len up to 1 + num_speculative_tokens.
        # Must match the threshold used by the indexer and flashmla_sparse so
        # that all backends agree on the decode/prefill split.
        self.decode_threshold = (
            self.reorder_batch_threshold + self.num_speculative_tokens
        )

        hf_config = self.vllm_config.model_config.hf_config
        assert hasattr(hf_config, "sliding_window")
        self.window_size = hf_config.sliding_window

        # Detect which DeepseekV4 layer types this model uses so we only build a
        # FlashMLA tile-scheduler plan for types that will actually be called.
        # Models without compress_ratios (pure SWA) fall back to swaonly.
        compress_ratios = getattr(hf_config, "compress_ratios", None) or [1]
        self._layer_types: set[str] = set()
        for ratio in compress_ratios:
            self._layer_types.add(_layer_type_for(int(ratio)))
        attention_config = self.vllm_config.attention_config
        self.use_flashmla_direct_kvcache_prefill = (
            attention_config.use_deepseek_v4_flashmla_direct_kvcache_prefill
        )

        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_indices = torch.zeros(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.swa_indices = torch.zeros(
            max_tokens,
            1,
            self.window_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.swa_lens = torch.zeros(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.is_valid_token = torch.zeros(
            max_tokens,
            dtype=torch.bool,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekSparseSWAMetadata:
        """Build SWA metadata for mixed decode/prefill batches.

        The batch is assumed to be reordered with decodes first by the scheduler.
        """
        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # Split into decode and prefill portions using configurable threshold
        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.decode_threshold
            )
        )

        # NOTE: Ensure all metadata tensors maintain fixed memory addresses
        # for CUDA graph compatibility. Padded full-cudagraph decode rows have
        # zero query length, so the repeated request-id list can be shorter
        # than the padded token count. Keep the exposed slice padded and use a
        # dummy request id for invalid rows; validity is carried separately by
        # is_valid_token.
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        num_tokens = num_decode_tokens + num_prefill_tokens
        token_to_req_indices = self.token_to_req_indices[:num_tokens]
        token_to_req_indices.zero_()
        token_to_req_indices[: x.shape[0]].copy_(x, non_blocking=True)

        is_valid_token = self.is_valid_token[:num_tokens]
        is_valid_token.copy_(slot_mapping >= 0)

        num_swa_tokens = (
            num_tokens
            if self.use_flashmla_direct_kvcache_prefill
            else num_decode_tokens
        )

        if num_swa_tokens > 0:
            self.swa_lens[num_swa_tokens:] = 0
            _compute_swa_indices_and_lens_kernel[(num_swa_tokens,)](
                self.swa_indices,
                self.swa_indices.stride(0),
                self.swa_lens,
                self.window_size,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                block_table.stride(0),
                self.block_size,
                TRITON_BLOCK_SIZE=1024,
            )

        deepseek_v4_fields = {}
        if not self.use_flashmla_direct_kvcache_prefill:
            # Pre-compute DeepseekV4 prefill metadata shared across all
            # attention layers for the bf16 prefill fallback.
            deepseek_v4_fields = self._build_deepseek_v4_metadata(
                num_decodes,
                num_prefills,
                seq_lens,
                query_start_loc,
            )

        # Per-layer-type tile-scheduler plan holders. The direct KV-cache
        # prefill path uses one FlashMLA decode-kernel call for all tokens, so
        # the scheduler is built for the same token count as swa_indices.
        tile_sched = self.build_tile_scheduler(num_swa_tokens)

        return DeepseekSparseSWAMetadata(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_valid_token=is_valid_token,
            token_to_req_indices=token_to_req_indices,
            swa_indices=self.swa_indices[:num_swa_tokens],
            swa_lens=self.swa_lens[:num_swa_tokens],
            block_size=self.block_size,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            tile_sched_swaonly=tile_sched[_LAYER_TYPE_SWAONLY],
            tile_sched_c4a=tile_sched[_LAYER_TYPE_C4A],
            tile_sched_c128a=tile_sched[_LAYER_TYPE_C128A],
            **deepseek_v4_fields,
        )

    def build_tile_scheduler(
        self,
        num_tokens: int,
    ) -> dict[str, FlashMLASchedMeta | None]:
        """Allocate one empty ``FlashMLASchedMeta`` per present DeepseekV4 layer type.

        Returned instances have ``tile_scheduler_metadata`` / ``num_splits``
        set to ``None``; the FlashMLA C++ path will allocate them and run the
        tile-scheduler planner on the first ``flash_mla_with_kvcache`` call of
        each type. Subsequent same-type calls reuse the plan because the
        tensors (and ``have_initialized``) are populated on the struct.

        Returns all-``None`` when there are no tokens for that phase, so callers
        see a clean sentinel.
        """
        out: dict[str, FlashMLASchedMeta | None] = {
            _LAYER_TYPE_SWAONLY: None,
            _LAYER_TYPE_C4A: None,
            _LAYER_TYPE_C128A: None,
        }
        if num_tokens == 0 or current_platform.is_rocm():
            return out
        for layer_type in self._layer_types:
            # get_mla_metadata() is the official FlashMLA entry point that
            # returns a fresh empty FlashMLASchedMeta; using it keeps this
            # call site aligned with the rest of the vLLM FlashMLA backends
            # that already go through the same stub.
            sched_meta = get_mla_metadata()[0]
            out[layer_type] = sched_meta
        return out

    def _build_deepseek_v4_metadata(
        self,
        num_decodes: int,
        num_prefills: int,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Pre-compute metadata for the DeepSeek V4 bf16 prefill fallback."""
        result: dict[str, torch.Tensor | None] = {}

        if num_prefills > 0:
            pfx_gather_lens = torch.empty(
                num_prefills, dtype=torch.int32, device=seq_lens.device
            )
            _compute_prefill_metadata_kernel[(1,)](
                pfx_gather_lens,
                seq_lens,
                query_start_loc,
                num_prefills,
                num_decodes,
                self.window_size,
                BLOCK_SIZE=triton.next_power_of_2(num_prefills),
            )

            result["prefill_seq_lens"] = seq_lens[num_decodes:]
            result["prefill_gather_lens"] = pfx_gather_lens

        return result


@triton.jit
def _compute_prefill_metadata_kernel(
    # Outputs
    prefill_gather_lens_ptr,
    # Inputs
    seq_lens_ptr,
    query_start_loc_ptr,
    num_prefills,
    num_decodes,
    window_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute prefill gather_lens in a single pass."""
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < num_prefills

    seq_len = tl.load(seq_lens_ptr + num_decodes + offset, mask=mask)
    qsl_start = tl.load(query_start_loc_ptr + num_decodes + offset, mask=mask)
    qsl_end = tl.load(query_start_loc_ptr + num_decodes + offset + 1, mask=mask)

    query_len = qsl_end - qsl_start
    prefix_len = seq_len - query_len
    gather_len = query_len + tl.minimum(prefix_len, window_size - 1)

    tl.store(prefill_gather_lens_ptr + offset, gather_len, mask=mask)


@triton.jit
def _compute_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + token_idx, 0)
        for i in range(0, window_size, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(
                swa_indices_ptr + token_idx * swa_indices_stride + offset,
                -1,
                mask=offset < window_size,
            )
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    start_pos = tl.maximum(pos - window_size + 1, 0)
    end_pos = pos + 1

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + token_idx, swa_len)

    for i in range(0, window_size, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + token_idx * swa_indices_stride + offset,
            slot_ids,
            mask=offset < window_size,
        )
