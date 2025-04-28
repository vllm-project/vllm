# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm._ipex_ops import ipex_ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata, FlashAttentionMetadataBuilder)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@dataclass
class IPEXAttentionMetadata(FlashAttentionMetadata):
    seq_start_loc: torch.Tensor = torch.tensor([0], dtype=torch.int64)

    def __init__(self,
                 flash_attn_metadata: FlashAttentionMetadata,
                 seq_start_loc: torch.Tensor = None,
                 **kwargs) -> None:
        super().__init__(**flash_attn_metadata.__dict__, **kwargs)
        if seq_start_loc is not None:
            self.seq_start_loc = seq_start_loc
        else:
            self.seq_start_loc = torch.tensor([0],
                                              dtype=torch.int64,
                                              device=self.block_table.device)


class IPEXAttentionMetadataBuilder(FlashAttentionMetadataBuilder):

    def __init__(self, runner: "GPUModelRunner", kv_cache_spec: AttentionSpec,
                 block_table: BlockTable):
        super().__init__(runner, kv_cache_spec, block_table)
        self.aot_schedule = (get_flash_attn_version() == 3)

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata):
        attn_metadata = super().build(num_reqs, num_actual_tokens,
                                      max_query_len, common_prefix_len,
                                      common_attn_metadata)
        seq_start_loc_cpu = self.runner.seq_start_loc_cpu[:num_reqs + 1]
        seq_start_loc = seq_start_loc_cpu.to(self.runner.device,
                                             non_blocking=True)
        return IPEXAttentionMetadata(attn_metadata,
                                     seq_start_loc=seq_start_loc)


class IPEXAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "IPEX_V1"

    @staticmethod
    def get_impl_cls() -> type["IPEXAttentionImpl"]:
        return IPEXAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return IPEXAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["IPEXAttentionMetadataBuilder"]:
        return IPEXAttentionMetadataBuilder


class IPEXAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = IPEXAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "IpexAttnBackendImpl")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: IPEXAttentionBackend,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with IPEXAttention.
        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output

        # NOTE(woosuk): IPEXAttention does not support FP8 KV cache.
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0, (
            "key/v_scale is not supported in IPEXAttention.")

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_heads = self.num_heads
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads
        query = query.view(-1, num_heads, head_size)
        key = key.view(-1, num_kv_heads, head_size)
        value = value.view(-1, num_kv_heads, head_size)

        # Reshape the input keys and values and store them in the cache.
        key_cache, value_cache = kv_cache.unbind(0)

        ipex_ops.reshape_and_cache_flash(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        ipex_ops.chunked_prefill(
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            output[:num_actual_tokens],
            attn_metadata.query_start_loc,
            attn_metadata.seq_start_loc,
            None,
            attn_metadata.block_table,
            self.alibi_slopes,
            attn_metadata.max_query_len,
            attn_metadata.max_seq_len,
            0.0,
            self.scale,
            False,
            True,
            False,
            None,
        )
        return output