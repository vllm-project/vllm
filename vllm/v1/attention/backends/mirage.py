# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import cdiv, is_pin_memory_available
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    get_kv_cache_layout,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

logger = init_logger(__name__)

class MirageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    # TODO: (Jianan) Make sure these are correct.
    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_size() -> list[int | MultipleOf]:
        return [16, 32, 64, 4096]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            attn_type = cls.__name__.removesuffix("Backend")
            raise ValueError(
                f"Head size {head_size} is not supported by {attn_type}. "
                f"Supported head sizes are: {supported_head_sizes}. "
                "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use "
                "FlexAttention backend which supports all head sizes."
            )

    @staticmethod
    def get_name() -> str:
        return "MPK_ATTENTION"

    @staticmethod
    def get_impl_cls() -> type["MirageAttentionImpl"]:
        return MirageAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["MirageAttentionMetadata"]:
        return MirageAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["MirageAttentionMetadataBuilder"]:
        return MirageAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

@dataclass
class MirageAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.

    # The data type of the query
    q_data_type: torch.dtype

    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    # Meta tensors
    qo_indptr_gpu: torch.Tensor | None = None
    paged_kv_indptr_gpu: torch.Tensor | None = None
    paged_kv_indices_gpu: torch.Tensor | None = None
    paged_kv_last_page_len_gpu: torch.Tensor | None = None


class MirageAttentionMetadataBuilder(AttentionMetadataBuilder[MirageAttentionMetadata]):
    cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config

        if vllm_is_batch_invariant():
            self.decode_fixed_split_size = 2048
            self.prefill_fixed_split_size = 4096
            self.disable_split_kv = True
        else:
            self.decode_fixed_split_size = -1
            self.prefill_fixed_split_size = -1
            self.disable_split_kv = False

        self.compilation_config = vllm_config.compilation_config
        max_num_pages_per_req = cdiv(
            self.model_config.max_model_len, self.kv_cache_spec.block_size
        )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        self.num_qo_heads = self.model_config.get_num_attention_heads(
            self.vllm_config.parallel_config
        )
        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        self.page_size = self.kv_cache_spec.block_size

        # Preparing persistent buffers (device-side)
        self.paged_kv_indptr = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=self.device
        )
        self.paged_kv_indices = torch.zeros(
            max_num_pages,  # max num pages possible
            dtype=torch.int32,
            device=self.device,
        )
        self.paged_kv_last_page_len = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=self.device
        )
        # host-side buffer
        pin_memory = is_pin_memory_available()
        self.paged_kv_indptr_cpu = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device="cpu", pin_memory=pin_memory
        )
        self.paged_kv_indptr_np = self.paged_kv_indptr_cpu.numpy()
        self.paged_kv_indices_cpu = torch.zeros(
            max_num_pages, dtype=torch.int32, device="cpu", pin_memory=pin_memory
        )
        self.paged_kv_last_page_len_cpu = torch.zeros(
            max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=pin_memory
        )
        self.paged_kv_last_page_len_np = self.paged_kv_last_page_len_cpu.numpy()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MirageAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        page_size = self.page_size
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        seq_lens_np = seq_lens_cpu.numpy()
        block_table_tensor = common_attn_metadata.block_table_tensor

        num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size

        # write self.paged_kv_indptr_cpu inplace (0-index is always 0)
        np.cumsum(
            num_blocks_np,
            dtype=np.int32,
            out=self.paged_kv_indptr_np[1 : num_reqs + 1],
        )

        paged_kv_indptr = self.paged_kv_indptr[: num_reqs + 1]
        paged_kv_indptr.copy_(
            self.paged_kv_indptr_cpu[: num_reqs + 1], non_blocking=True
        )

        # write self.paged_kv_indices inplace
        num_actual_pages = self.paged_kv_indptr_np[num_reqs]
        paged_kv_indices = self.paged_kv_indices[:num_actual_pages]
        _copy_page_indices_kernel[(num_reqs,)](
            paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )

        # write self.paged_kv_last_page_len_cpu inplace
        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len_np[:num_reqs] = np.where(
            paged_kv_last_page_len_np == 0,
            page_size,
            paged_kv_last_page_len_np,
        )
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]
        paged_kv_last_page_len.copy_(
            self.paged_kv_last_page_len_cpu[:num_reqs], non_blocking=True
        )

        # uses_spec_reorder = self.reorder_batch_threshold > 1
        
        assert self.q_data_type == torch.bfloat16, "MirageAttentionBackend currently only supports bfloat16"

        attn_metadata = MirageAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            q_data_type=self.q_data_type,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            qo_indptr_gpu=common_attn_metadata.query_start_loc_gpu,
            paged_kv_indptr_gpu=self.paged_kv_indptr,
            paged_kv_indices_gpu=self.paged_kv_indices,
            paged_kv_last_page_len_gpu=self.paged_kv_last_page_len,
        )

        return attn_metadata


class MirageAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.sinks = sinks
        pass

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MirageAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass that do nothing.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: KV cache tensor with different possible shapes:
                - NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]
                - HND: [num_blocks, 2, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        return output


@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )
