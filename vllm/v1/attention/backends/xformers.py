# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with XFormersAttention."""

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

try:
    from xformers import ops as xops
    from xformers.ops.fmha.attn_bias import (
        AttentionBias,
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    )

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

from vllm import _custom_ops as ops

logger = init_logger(__name__)


class XFormersAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
            136,
            144,
            152,
            160,
            168,
            176,
            184,
            192,
            200,
            208,
            216,
            224,
            232,
            240,
            248,
            256,
        ]

    @staticmethod
    def get_name() -> str:
        return "XFORMERS"

    @staticmethod
    def get_impl_cls() -> type["XFormersAttentionImpl"]:
        return XFormersAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["XFormersAttentionMetadataBuilder"]:
        return XFormersAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class XFormersAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # Biases for different attention types.
    attn_bias: Optional["AttentionBias"] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["XFormersAttentionMetadata"] = None
    _cached_decode_metadata: Optional["XFormersAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["XFormersAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        q_start_loc = self.query_start_loc[self.num_decodes :]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[self.num_decodes :]
        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = XFormersAttentionMetadata(
            num_actual_tokens=self.num_prefill_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc - q_start_loc[0],
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_table=self.block_table[self.num_decodes :],
            slot_mapping=self.slot_mapping[self.num_decode_tokens :],
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["XFormersAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata

        q_start_loc = self.query_start_loc
        q_seqlens = torch.diff(q_start_loc)
        decode_kv_seqlens = self.seq_lens[: self.num_decodes]
        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = XFormersAttentionMetadata(
            num_actual_tokens=self.num_decode_tokens,
            max_query_len=int(q_seqlens[: self.num_decodes].max().item()),
            query_start_loc=q_start_loc[: self.num_decodes + 1],
            max_seq_len=int(decode_kv_seqlens.max().item()),
            seq_lens=decode_kv_seqlens,
            block_table=self.block_table[: self.num_decodes],
            slot_mapping=self.slot_mapping[: self.num_decode_tokens],
            attn_bias=self.attn_bias,
        )
        return self._cached_decode_metadata


class XFormersAttentionMetadataBuilder(
    AttentionMetadataBuilder[XFormersAttentionMetadata]
):
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        assert XFORMERS_AVAILABLE
        self.block_size = kv_cache_spec.block_size
        self._num_decodes = 0
        self._num_decode_tokens = 0

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XFormersAttentionMetadata:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        q_start_loc = common_attn_metadata.query_start_loc
        q_seqlens = torch.diff(q_start_loc)
        max_query_len = common_attn_metadata.max_query_len
        kv_seqlens = common_attn_metadata.seq_lens
        max_seq_len = common_attn_metadata.max_seq_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        bias = None
        if num_decodes > 0:
            # Construct the decoder bias.
            decode_q_seqlens = q_seqlens[:num_decodes]
            decode_kv_seqlens = kv_seqlens[:num_decodes]
            bias = PagedBlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=decode_q_seqlens.tolist(),
                kv_seqlen=decode_kv_seqlens.tolist(),
                page_size=self.block_size,
                block_tables=block_table[:num_decodes],
                device=block_table.device,
            )

        return XFormersAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            max_query_len=max_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=kv_seqlens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            attn_bias=bias,
        )


class XFormersAttentionImpl(AttentionImpl):
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
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in V0.")
        if alibi_slopes is not None:
            raise NotImplementedError("XFormers does not support alibi slopes yet.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        if logits_soft_cap is None:
            # Setting logits_soft_cap to 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "XFormersAttentionImpl."
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XFormersAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with XFormers.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for XFormersAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # Cache the input KVs.
        key_cache, value_cache = kv_cache.unbind(0)
        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        if prefill_meta := attn_metadata.prefill_metadata:
            descale_shape = (prefill_meta.query_start_loc.shape[0] - 1, key.shape[1])
            unified_attention(
                q=query[num_decode_tokens:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                seqused_k=prefill_meta.seq_lens,
                max_seqlen_k=prefill_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=prefill_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )

        if decode_meta := attn_metadata.decode_metadata:
            # Query for decode. KV is not needed because it is already cached.
            decode_query = query[:num_decode_tokens]
            # Reshape query to [1, B_T, G, H, D].
            q = decode_query.view(
                1, -1, self.num_kv_heads, self.num_queries_per_kv, self.head_size
            )
            # Reshape the k and v caches to [1, Bkv_T, G, H, D]
            cache_k = key_cache.view(
                1, -1, self.num_kv_heads, 1, self.head_size
            ).expand(
                1,
                -1,
                self.num_kv_heads,
                self.num_queries_per_kv,
                self.head_size,
            )
            cache_v = value_cache.view(
                1, -1, self.num_kv_heads, 1, self.head_size
            ).expand(
                1,
                -1,
                self.num_kv_heads,
                self.num_queries_per_kv,
                self.head_size,
            )

            attn_bias = decode_meta.attn_bias
            output[:num_decode_tokens] = xops.memory_efficient_attention_forward(
                q,
                cache_k,
                cache_v,
                attn_bias=attn_bias,
                p=0.0,
                scale=self.scale,
            ).view(decode_query.shape)

        # Reshape the output tensor.
        return output
