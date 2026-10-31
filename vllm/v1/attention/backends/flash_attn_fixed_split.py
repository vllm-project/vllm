# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4 attention with a fixed split-KV schedule, for batch invariance.

Every split covers the same keys regardless of batch composition, sequence
length or prefill/decode phase, so a token's output does not depend on what
else is scheduled; a training-side replay can run the same schedule.
"""

import inspect
from typing import ClassVar, cast

import torch

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
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
    dtype: torch.dtype,
    kv_cache_dtype: str,
    sliding_window: int | None,
    max_model_len: int,
    *,
    has_sink: bool = False,
    attn_type: str = AttentionType.DECODER,
    use_dcp: bool = False,
) -> str | None:
    """Why the fixed split-KV schedule cannot serve this attention layer:
    the backend's declared limits (``validate_configuration``) plus the
    schedule's sequence-length bound."""
    capability = current_platform.get_device_capability()
    if capability is None:
        reasons = ["compute capability not supported"]
    else:
        reasons = FlashAttnFixedSplitBackend.validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=cast(CacheDType, kv_cache_dtype),
            block_size=None,
            use_mla=False,
            has_sink=has_sink,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=capability,
            attn_type=attn_type,
            has_sliding_window=sliding_window is not None,
            use_dcp=use_dcp,
        )
    if max_model_len > MAX_SEQ_LEN:
        reasons.append(f"max_model_len {max_model_len} > {MAX_SEQ_LEN}")
    return "; ".join(reasons) or None


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
            config.model_config.dtype,
            kv_cache_dtype,
            sliding_window,
            config.model_config.max_model_len,
            has_sink=sinks is not None,
            attn_type=attn_type,
            use_dcp=config.parallel_config.decode_context_parallel_size != 1,
        )
        if reason is None and (
            alibi_slopes is not None
            or logits_soft_cap is not None
            or kv_sharing_target_layer_name is not None
        ):
            reason = "ALiBi, logits soft cap and KV sharing not supported"
        if reason is not None:
            raise ValueError(
                f"Fixed split-KV FA4 attention (head_size={head_size}, "
                f"kv_cache_dtype={kv_cache_dtype!r}, "
                f"sliding_window={sliding_window}, attn_type={attn_type}) "
                f"is not supported: {reason}"
            )
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
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_impl_cls() -> type[FlashAttnFixedSplitImpl]:
        return FlashAttnFixedSplitImpl

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == 128

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_rswa(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        params = inspect.signature(_flash_attn_fwd).parameters
        if not {"seqlen_k_per_split", "disable_scheduler_metadata"} <= params.keys():
            return "requires a vllm-flash-attn build with fixed split-KV scheduling"
        return None
