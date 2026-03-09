# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MPS (Apple Metal) attention backend using pure PyTorch operations.

Uses F.scaled_dot_product_attention for both prefill and decode,
with paged KV cache via tensor indexing (no C++ extensions needed).
"""

import logging
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

logger = init_logger(__name__)


class MPSAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.float32,
    ]

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.float32]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 112, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "MPS_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["MPSAttentionBackendImpl"]:
        return MPSAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["MPSAttentionMetadataBuilder"]:
        return MPSAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return 2, num_blocks, num_kv_heads, block_size, head_size

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class MPSAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_reqs: int = 0
    causal: bool = True
    # CPU copies to avoid GPU→CPU sync in the per-sequence loop.
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None


class MPSAttentionMetadataBuilder(AttentionMetadataBuilder[MPSAttentionMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self._init_reorder_batch_threshold(None, False)

        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.num_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.head_dim = kv_cache_spec.head_size
        self.dtype = vllm_config.model_config.dtype
        self.block_size = vllm_config.cache_config.block_size
        self.is_cross_attention = isinstance(kv_cache_spec, CrossAttentionSpec)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MPSAttentionMetadata:
        causal = False if self.is_cross_attention else common_attn_metadata.causal
        num_reqs = common_attn_metadata.num_reqs
        return MPSAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_reqs=num_reqs,
            causal=causal,
            # CPU copies avoid GPU→CPU sync in the attention hot path.
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu[
                : num_reqs + 1
            ],
            seq_lens_cpu=common_attn_metadata.seq_lens[:num_reqs].to(
                "cpu", non_blocking=True
            ),
        )


class MPSAttentionBackendImpl(AttentionImpl):
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
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.attn_type = attn_type

        if alibi_slopes is not None:
            logger.warning_once("MPS attention does not support ALiBi slopes.")
        self.alibi_slopes = None

        if logits_soft_cap is not None and logits_soft_cap > 0:
            logger.warning_once("MPS attention does not support logits soft cap.")
        self.logits_soft_cap = None

        if sliding_window is not None:
            logger.warning_once("MPS attention does not support sliding window.")
        self.sliding_window = None

        if sinks is not None:
            logger.warning_once("MPS attention does not support attention sinks.")
        self.sinks = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MPSAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with paged KV cache on MPS.

        Args:
            query: [num_tokens, num_heads, head_size]
            key: [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            kv_cache: [2, num_blocks, num_kv_heads, block_size, head_size]
            attn_metadata: MPS attention metadata
        Returns:
            [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not yet supported "
                "for MPSAttentionBackendImpl"
            )

        # Warmup pass
        if attn_metadata is None:
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Encoder attention: no KV cache
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._run_sdpa_forward(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
            )

        # Decoder / cross-attention: use paged KV cache
        key_cache, value_cache = kv_cache.unbind(0)

        # Write new K,V into cache
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            if logger.isEnabledFor(logging.DEBUG):
                sm = attn_metadata.slot_mapping
                torch.mps.synchronize()
                sm_cpu = sm[: key.shape[0]].cpu()
                logger.debug(
                    "_reshape_and_cache: key=%s kc=%s sm=%s "
                    "sm_dtype=%s sm_dev=%s sm_vals=%s",
                    key.shape,
                    key_cache.shape,
                    sm.shape,
                    sm.dtype,
                    sm.device,
                    sm_cpu.tolist(),
                )
            _reshape_and_cache(
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                key_cache,
                value_cache,
                attn_metadata.slot_mapping[:num_actual_tokens],
            )

        # Run attention per-sequence with paged KV gather
        block_table = attn_metadata.block_table
        block_size = key_cache.shape[
            2
        ]  # [num_blocks, num_kv_heads, block_size, head_size]
        num_seqs = attn_metadata.num_reqs

        # Use pre-computed CPU copies to avoid GPU→CPU sync per layer.
        query_start_loc_cpu = attn_metadata.query_start_loc_cpu
        seq_lens_cpu = attn_metadata.seq_lens_cpu
        if query_start_loc_cpu is None:
            query_start_loc_cpu = attn_metadata.query_start_loc[: num_seqs + 1].cpu()
        if seq_lens_cpu is None:
            seq_lens_cpu = attn_metadata.seq_lens[:num_seqs].cpu()

        for i in range(num_seqs):
            q_start = int(query_start_loc_cpu[i])
            q_end = int(query_start_loc_cpu[i + 1])
            q_len = q_end - q_start

            if q_len == 0:
                continue

            seq_len = int(seq_lens_cpu[i])
            num_blocks_needed = (seq_len + block_size - 1) // block_size
            blocks = block_table[i, :num_blocks_needed]

            # Gather K,V from paged cache
            # key_cache[blocks]:
            #   [num_blocks_needed, num_kv_heads, block_size, head_size]
            # Transpose to [num_kv_heads, num_blocks_needed, block_size, head_size]
            # then reshape to merge blocks×block_size into the sequence dim.
            k_paged = (
                key_cache[blocks]
                .transpose(0, 1)
                .reshape(self.num_kv_heads, -1, self.head_size)[:, :seq_len, :]
            )
            v_paged = (
                value_cache[blocks]
                .transpose(0, 1)
                .reshape(self.num_kv_heads, -1, self.head_size)[:, :seq_len, :]
            )

            # query: [q_len, num_heads, head_size]
            #     -> [1, num_heads, q_len, head_size]
            q = query[q_start:q_end].transpose(0, 1).unsqueeze(0)
            # k,v: [num_kv_heads, seq_len, head_size]
            #   -> [1, num_kv_heads, seq_len, head_size]
            k = k_paged.unsqueeze(0)
            v = v_paged.unsqueeze(0)

            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=(attn_metadata.causal and q_len > 1),
                scale=self.scale,
                enable_gqa=(self.num_heads != self.num_kv_heads),
            )

            # [1, num_heads, q_len, head_size] -> [q_len, num_heads, head_size]
            output[q_start:q_end] = attn_out.squeeze(0).transpose(0, 1)

        return output

    def _run_sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MPSAttentionMetadata,
    ) -> torch.Tensor:
        """Run SDPA for encoder/encoder-only attention (no KV cache)."""
        num_seqs = attn_metadata.num_reqs
        query_start_loc_cpu = attn_metadata.query_start_loc_cpu
        if query_start_loc_cpu is None:
            query_start_loc_cpu = attn_metadata.query_start_loc[: num_seqs + 1].cpu()

        for i in range(num_seqs):
            start = int(query_start_loc_cpu[i])
            end = int(query_start_loc_cpu[i + 1])

            q = query[start:end].transpose(0, 1).unsqueeze(0)
            k = key[start:end].transpose(0, 1).unsqueeze(0)
            v = value[start:end].transpose(0, 1).unsqueeze(0)

            sub_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
                enable_gqa=(self.num_heads != self.num_kv_heads),
            )

            output[start:end] = sub_out.squeeze(0).transpose(0, 1)

        return output


def _reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter K,V into the paged cache using indexing.

    key: [num_tokens, num_kv_heads, head_size]
    key_cache: [num_blocks, num_kv_heads, block_size, head_size]
    slot_mapping: [num_tokens] — flat slot indices
    """
    num_tokens = key.shape[0]
    if num_tokens == 0:
        return

    block_size = key_cache.shape[2]
    slot_mapping_flat = slot_mapping[:num_tokens]
    block_idx = slot_mapping_flat // block_size
    block_off = slot_mapping_flat % block_size

    key_cache[block_idx, :, block_off, :] = key[:num_tokens]
    value_cache[block_idx, :, block_off, :] = value[:num_tokens]
