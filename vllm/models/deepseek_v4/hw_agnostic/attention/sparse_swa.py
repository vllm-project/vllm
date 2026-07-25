# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import ClassVar, cast

import torch

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.hw_agnostic.layers.attention_layer_base import (
    AttentionLayerBase,
)
from vllm.model_executor.hw_agnostic.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.model_executor.hw_agnostic.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.models.deepseek_v4.hw_agnostic.attention._metadata_utils import (
    split_decodes_and_prefills,
)
from vllm.triton_utils import tl, triton


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

        # Block size matched to C4A page size (shared physical tensor).
        self.block_size = 64
        assert self.dtype in (torch.uint8, torch.bfloat16, torch.float8_e4m3fn)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla pages carry 584B of payload per token (448B fp8 NoPE
        # + 128B bf16 RoPE + 8B UE8M0 scale); padded to 576B alignment.
        # The Triton dequant kernel reads pages at the same stride.
        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else None,
            model_version="deepseek_v4",
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekSparseSWABackend


class DeepseekSparseSWABackend(AttentionBackend):
    """Spec carrier for the DSv4 sliding-window cache."""

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_SPARSE_SWA"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 256

    @staticmethod
    def get_builder_cls() -> type["DeepseekSparseSWAMetadataBuilder"]:
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
            # 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            return (num_blocks, block_size, 584)
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
    seq_lens: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    query_start_loc_cpu: torch.Tensor | None = None

    is_valid_token: torch.Tensor | None = None
    token_to_req_indices: torch.Tensor | None = None
    decode_swa_indices: torch.Tensor | None = None
    decode_swa_lens: torch.Tensor | None = None

    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    prefill_seq_lens: torch.Tensor | None = None
    prefill_gather_lens: torch.Tensor | None = None


class DeepseekSparseSWAMetadataBuilder(AttentionMetadataBuilder):
    """Builds metadata for DeepseekV4 SWA cache."""

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

        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        # decode_threshold = 1 + num_speculative_tokens; matches the indexer
        # builder so both builders agree on the decode/prefill split.
        self.decode_threshold = (
            self.reorder_batch_threshold + self.num_speculative_tokens
        )

        hf_config = self.vllm_config.model_config.hf_config
        assert hasattr(hf_config, "sliding_window")
        self.window_size = hf_config.sliding_window

        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_indices = torch.zeros(
            max_tokens, dtype=torch.int32, device=self.device
        )
        self.decode_swa_indices = torch.zeros(
            max_tokens, 1, self.window_size, dtype=torch.int32, device=self.device
        )
        self.decode_swa_lens = torch.zeros(
            max_tokens, dtype=torch.int32, device=self.device
        )
        self.is_valid_token = torch.zeros(
            max_tokens, dtype=torch.bool, device=self.device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekSparseSWAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.decode_threshold
            )
        )

        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)

        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        is_valid_token.copy_(slot_mapping >= 0)

        if num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            _compute_swa_indices_and_lens_kernel[(num_decode_tokens,)](
                self.decode_swa_indices,
                self.decode_swa_indices.stride(0),
                self.decode_swa_lens,
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

        deepseek_v4_fields = self._build_deepseek_v4_metadata(
            num_decodes,
            num_prefills,
            seq_lens,
            query_start_loc,
        )

        return DeepseekSparseSWAMetadata(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_valid_token=is_valid_token,
            token_to_req_indices=token_to_req_indices,
            decode_swa_indices=self.decode_swa_indices[:num_decode_tokens],
            decode_swa_lens=self.decode_swa_lens[:num_decode_tokens],
            block_size=self.block_size,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            **deepseek_v4_fields,
        )

    def _build_deepseek_v4_metadata(
        self,
        num_decodes: int,
        num_prefills: int,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
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
    prefill_gather_lens_ptr,
    seq_lens_ptr,
    query_start_loc_ptr,
    num_prefills,
    num_decodes,
    window_size,
    BLOCK_SIZE: tl.constexpr,
):
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
