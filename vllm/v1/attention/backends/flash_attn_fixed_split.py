# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4 attention with a fixed split-KV schedule, for batch invariance.

Every split covers the same keys regardless of batch composition, sequence
length or prefill/decode phase, so a token's output does not depend on what
else is scheduled; a training-side replay can run the same schedule.
"""

import inspect

import torch

from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

# Every split covers SEQLEN_K_PER_SPLIT keys, for any batch or sequence length.
MAX_SEQ_LEN = 262144
SEQLEN_K_PER_SPLIT = 8192
NUM_SPLITS = 32


def fixed_split_unsupported_reason(
    head_size: int,
    kv_cache_dtype: str,
    sliding_window: int | None,
    max_model_len: int,
) -> str | None:
    """Why the fixed split-KV schedule cannot serve this attention layer."""
    if not current_platform.is_device_capability_family(100):
        return "requires an SM10x GPU"
    params = inspect.signature(_flash_attn_fwd).parameters
    if not {"seqlen_k_per_split", "disable_scheduler_metadata"} <= params.keys():
        return "requires a vllm-flash-attn build with fixed split-KV scheduling"
    if head_size != 128:
        return f"requires head_size 128, got {head_size}"
    if kv_cache_dtype not in ("fp8", "fp8_e4m3", "auto", "bfloat16"):
        return f"requires an FP8 E4M3 or BF16 KV cache, got {kv_cache_dtype!r}"
    if sliding_window is not None:
        return "does not support sliding-window attention"
    if max_model_len > MAX_SEQ_LEN:
        return f"supports max_model_len <= {MAX_SEQ_LEN}, got {max_model_len}"
    return None


class FlashAttnFixedSplitImpl(FlashAttentionImpl):
    supports_quant_query_input = True
    supports_dcp = False

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
        sinks: torch.Tensor | None = None,
    ) -> None:
        config = get_current_vllm_config()
        reason = fixed_split_unsupported_reason(
            head_size,
            kv_cache_dtype,
            sliding_window,
            config.model_config.max_model_len,
        )
        if reason is None and (
            alibi_slopes is not None
            or logits_soft_cap is not None
            or sinks is not None
            or attn_type != AttentionType.DECODER
            or kv_sharing_target_layer_name is not None
            or config.parallel_config.decode_context_parallel_size != 1
        ):
            reason = "requires plain causal decoder attention without DCP"
        if reason is not None:
            raise ValueError(f"Fixed split-KV FA4 attention {reason}")
        # FlashAttentionImpl.__init__ is skipped: it picks a vllm_flash_attn
        # version and rejects FP8 KV caches, while this impl calls the CuTe FA4
        # kernel directly. Set the attributes other code reads.
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.alibi_slopes = None
        self.sliding_window = (-1, -1)
        self.logits_soft_cap = 0
        self.sinks = None
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = None
        self.attn_type = attn_type
        self.batch_invariant_enabled = True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("Fixed split-KV FA4 does not fuse output quantization")
        if attn_metadata is None:
            return output.fill_(0)
        if (
            attn_metadata.causal is not True
            or attn_metadata.max_query_len > MAX_SEQ_LEN
            or attn_metadata.max_seq_len > MAX_SEQ_LEN
        ):
            raise ValueError("Unsupported fixed split-KV FA4 metadata")

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        fp8 = self.kv_cache_dtype.startswith("fp8")
        if fp8:
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())
        num_tokens = attn_metadata.num_actual_tokens
        scale_shape = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)
        descale = (
            dict(
                q_descale=layer._q_scale.expand(scale_shape),
                k_descale=layer._k_scale.expand(scale_shape),
                v_descale=layer._v_scale.expand(scale_shape),
            )
            if fp8
            else {}
        )
        _flash_attn_fwd(
            query[:num_tokens],
            key_cache,
            value_cache,
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_q=MAX_SEQ_LEN,
            max_seqlen_k=MAX_SEQ_LEN,
            page_table=attn_metadata.block_table,
            softmax_scale=self.scale,
            causal=True,
            **descale,
            tile_mn=(128, 128),
            pack_gqa=True,
            num_splits=NUM_SPLITS,
            seqlen_k_per_split=SEQLEN_K_PER_SPLIT,
            disable_scheduler_metadata=True,
            out=output[:num_tokens],
        )
        return output


class FlashAttnFixedSplitBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[FlashAttnFixedSplitImpl]:
        return FlashAttnFixedSplitImpl
