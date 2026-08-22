# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential FlashAttention backend for Phi-4 Flash."""

from __future__ import annotations

from typing import ClassVar

import torch
from einops import rearrange

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )

from .flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
    cascade_attention,
)

logger = init_logger(__name__)


class DifferentialFlashAttentionBackend(FlashAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_name() -> str:
        return "DIFFERENTIAL_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type[DifferentialFlashAttentionImpl]:
        return DifferentialFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return FlashAttentionMetadataBuilder

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
        if num_kv_heads % 2 != 0:
            raise ValueError("num_kv_heads must be divisible by 2.")
        return (2, 2, num_blocks, block_size, num_kv_heads // 2, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (2, 3, 0, 1, 4, 5)
        return (0, 1, 2, 3, 4, 5)


class DifferentialFlashAttentionImpl(FlashAttentionImpl):
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
        differential_flash_attention_config: dict | None = None,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
        )
        if differential_flash_attention_config is None:
            differential_flash_attention_config = {}
        self.differential_flash_attention_config = differential_flash_attention_config
        self.lambda_full: torch.Tensor | None = None
        self.lambda_init = differential_flash_attention_config["lambda_init"]
        self.subln = differential_flash_attention_config["subln"]

    def _get_lambda_full(self, q: torch.Tensor) -> torch.Tensor:
        if self.lambda_full is None:
            lambda_q1 = self.differential_flash_attention_config["lambda_q1"]
            lambda_k1 = self.differential_flash_attention_config["lambda_k1"]
            lambda_q2 = self.differential_flash_attention_config["lambda_q2"]
            lambda_k2 = self.differential_flash_attention_config["lambda_k2"]
            lambda_1 = torch.exp(torch.sum(lambda_q1 * lambda_k1, dim=-1).float())
            lambda_2 = torch.exp(torch.sum(lambda_q2 * lambda_k2, dim=-1).float())
            self.lambda_full = (lambda_1 - lambda_2 + self.lambda_init).type_as(q)
        return self.lambda_full

    def _split_heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = rearrange(x, "... (h two) d -> ... h two d", two=2)
        return x[..., 0, :].contiguous(), x[..., 1, :].contiguous()

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        if kv_cache.numel() == 0:
            empty = torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)
            return (empty, empty), (empty, empty)
        kv_cache_1 = kv_cache[0]
        kv_cache_2 = kv_cache[1]
        key_cache_1, value_cache_1 = kv_cache_1.unbind(0)
        key_cache_2, value_cache_2 = kv_cache_2.unbind(0)
        return (key_cache_1, value_cache_1), (key_cache_2, value_cache_2)

    def _forward_single(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            raise NotImplementedError(
                "Differential FlashAttention only supports decoder attention."
            )
        if self.dcp_world_size > 1:
            raise NotImplementedError(
                "Differential FlashAttention does not support DCP."
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        if not attn_metadata.use_cascade:
            sliding_window_size = (
                list(self.sliding_window) if self.sliding_window is not None else None
            )
            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                key_cache.shape[-2],
            )
            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=attn_metadata.block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=layer._q_scale.expand(descale_shape),
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )
            return output

        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        (key_cache_1, value_cache_1), (key_cache_2, value_cache_2) = (
            self._split_kv_cache(kv_cache)
        )
        key_1, key_2 = self._split_heads(key)
        value_1, value_2 = self._split_heads(value)
        reshape_and_cache_flash(
            key_1,
            value_1,
            key_cache_1,
            value_cache_1,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
        reshape_and_cache_flash(
            key_2,
            value_2,
            key_cache_2,
            value_cache_2,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

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
            raise NotImplementedError(
                "fused output quantization is not yet supported for "
                "DifferentialFlashAttentionImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)

        lambda_full = self._get_lambda_full(query)
        query_1, query_2 = self._split_heads(query)
        (key_cache_1, value_cache_1), (key_cache_2, value_cache_2) = (
            self._split_kv_cache(kv_cache)
        )

        attn11 = torch.empty_like(query_1)
        attn12 = torch.empty_like(query_1)
        attn21 = torch.empty_like(query_2)
        attn22 = torch.empty_like(query_2)

        self._forward_single(
            layer, query_1, key_cache_1, value_cache_1, attn_metadata, attn11
        )
        self._forward_single(
            layer, query_1, key_cache_1, value_cache_2, attn_metadata, attn12
        )
        self._forward_single(
            layer, query_2, key_cache_2, value_cache_1, attn_metadata, attn21
        )
        self._forward_single(
            layer, query_2, key_cache_2, value_cache_2, attn_metadata, attn22
        )

        attn1 = torch.cat([attn11, attn12], dim=-1)
        attn2 = torch.cat([attn21, attn22], dim=-1)
        attn = self.subln(attn1 - lambda_full * attn2)
        attn = attn * (1 - self.lambda_init)
        attn_output = rearrange(attn, "... h (two d) -> ... (h two) d", two=2)
        output[: attn_metadata.num_actual_tokens] = attn_output[
            : attn_metadata.num_actual_tokens
        ]
        return output
