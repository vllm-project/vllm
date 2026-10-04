# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton attention backend with different K/V head dimensions (DiffKV).

The KV cache layout is identical to ``FlashAttentionDiffKVBackend``: K and V
are packed along the last dim in the logical shape
``[num_blocks, num_kv_heads, block_size, head_size_qk + head_size_v]``.
"""

from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    can_use_split_kv,
    unified_attention_diffkv,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@dataclass
class TritonAttentionDiffKVMetadata(TritonAttentionMetadata):
    partitions: tuple[TritonAttentionMetadata, ...] = ()


class TritonAttentionDiffKVMetadataBuilder(TritonAttentionMetadataBuilder):
    """Override the parent's softmax buffer last-dim to head_size_v.

    The parent allocates ``softmax_segm_output`` with last-dim sized to
    ``next_power_of_2(head_size)`` (== Q/K head size).  For DiffKV the
    accumulator and per-segment partial outputs are V-shaped, so we
    re-allocate with ``next_power_of_2(head_size_v)`` instead.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        head_size_v = TritonAttentionDiffKVBackend.head_size_v
        head_size_v_padded = next_power_of_2(head_size_v)
        self.softmax_segm_output = torch.empty(
            (
                self.softmax_segm_max.shape[0],
                self.num_heads_q,
                self.num_par_softmax_segments,
                head_size_v_padded,
            ),
            dtype=torch.float32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionDiffKVMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata = TritonAttentionDiffKVMetadata(**vars(base))
        capacity = min(
            self.seq_threshold_3D,
            self.softmax_segm_output.shape[0],
            self.softmax_segm_max.shape[0],
            self.softmax_segm_expsum.shape[0],
        )
        assert self.reorder_batch_threshold is not None
        if (
            base.num_actual_tokens <= capacity
            or base.max_query_len <= self.reorder_batch_threshold
        ):
            return metadata
        num_decodes, _, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )
        if not (0 < num_decode_tokens <= capacity and num_prefill_tokens > 0):
            return metadata
        decode_max_query_len = int(
            (
                common_attn_metadata.query_start_loc_cpu[1 : num_decodes + 1]
                - common_attn_metadata.query_start_loc_cpu[:num_decodes]
            ).max()
        )
        if not can_use_split_kv(
            decode_max_query_len,
            num_decodes,
            self.seq_threshold_3D,
            self.num_par_softmax_segments,
            self.softmax_segm_output,
            self.softmax_segm_max,
            self.softmax_segm_expsum,
        ):
            return metadata
        decode = replace(
            base,
            num_actual_tokens=num_decode_tokens,
            max_query_len=decode_max_query_len,
            query_start_loc=base.query_start_loc[: num_decodes + 1],
            seq_lens=base.seq_lens[:num_decodes],
            block_table=base.block_table[:num_decodes],
            slot_mapping=base.slot_mapping[:num_decode_tokens],
        )
        prefill = replace(
            base,
            num_actual_tokens=num_prefill_tokens,
            query_start_loc=base.query_start_loc[num_decodes:] - num_decode_tokens,
            seq_lens=base.seq_lens[num_decodes:],
            block_table=base.block_table[num_decodes:],
            slot_mapping=base.slot_mapping[num_decode_tokens:],
        )
        metadata.partitions = (decode, prefill)
        return metadata


class TritonAttentionDiffKVBackend(TritonAttentionBackend):
    # V head dim — set per layer via ``set_head_size_v`` before instantiation.
    head_size_v: int = 128

    # No FP8 / int8 KV cache for the DiffKV path yet; require fp16/bf16/fp32.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        cls.head_size_v = head_size_v

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN_DIFFKV"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionDiffKVImpl"]:
        return TritonAttentionDiffKVImpl

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionDiffKVMetadataBuilder"]:
        return TritonAttentionDiffKVMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # DiffKV K head sizes (e.g. 192 for MiMo-V2.5) need to be allowed.
        return head_size >= 32

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        # DiffKV only implements decoder self-attention.  Unlike the parent
        # TritonAttentionBackend (which advertises all types), encoder
        # attention is not supported, so gate it here at backend selection.
        return attn_type == AttentionType.DECODER


class TritonAttentionDiffKVImpl(TritonAttentionImpl):
    """Triton attention impl for the DiffKV packed KV cache layout."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not yet support quantized "
                f"KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."
            )
        if self._is_per_token_head_quant:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support per-token-head "
                "quantization."
            )
        if self.chunk_lookback > -1:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support chunked "
                "attention with lookback."
            )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Cache is logical (B, H, N, C); the diffkv reshape kernel expects
        # (B, N, H, C).
        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache.transpose(1, 2),
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        # The fused rope+cache path assumes the standard 2-tensor layout.
        return False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Shapes:
            query:    [num_tokens, num_heads, head_size_qk]
            key:      [num_tokens, num_kv_heads, head_size_qk]
            value:    [num_tokens, num_kv_heads, head_size_v]
            kv_cache: [num_blocks, num_kv_heads, block_size,
                       head_size_qk + head_size_v]
            output:   [num_tokens, num_heads, head_size_v]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for "
                "TritonAttentionDiffKVImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False, (
            "Cascade attention not supported for TritonAttentionDiffKVImpl"
        )

        head_size_qk = self.head_size
        head_size_v = TritonAttentionDiffKVBackend.head_size_v

        # Triton DiffKV kernels consume (B, N, H, D) cache views.
        kv_cache = kv_cache.transpose(1, 2)
        key_cache = kv_cache[..., :head_size_qk]
        value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]

        partitions = getattr(attn_metadata, "partitions", ()) or (attn_metadata,)
        token_start = 0
        for partition in partitions:
            token_end = token_start + partition.num_actual_tokens
            unified_attention_diffkv(
                q=query[token_start:token_end],
                k=key_cache,
                v=value_cache,
                out=output[token_start:token_end],
                cu_seqlens_q=partition.query_start_loc,
                seqused_k=partition.seq_lens,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                use_alibi_sqrt=self.use_alibi_sqrt,
                window_size=self.sliding_window,
                block_table=partition.block_table,
                softcap=self.logits_soft_cap,
                sinks=self.sinks,
                max_seqlen_q=partition.max_query_len,
                seq_threshold_3D=partition.seq_threshold_3D,
                num_par_softmax_segments=partition.num_par_softmax_segments,
                softmax_segm_output=partition.softmax_segm_output,
                softmax_segm_max=partition.softmax_segm_max,
                softmax_segm_expsum=partition.softmax_segm_expsum,
            )
            token_start = token_end
        return output
