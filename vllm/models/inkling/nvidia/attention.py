# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from ..configs import InklingModelConfig
from .layernorm import InklingRMSNorm
from .ops.fa4_rel_attention import (
    bucket_max_seqlen_q,
    inkling_fa4_num_splits,
    inkling_fa4_rel_attention,
)
from .ops.fa4_warmup import InklingFA4WarmupConfig, register_fa4_warmup
from .ops.qkvr_prep import fused_qkvr_prep
from .sconv_swa_attn import _K, _V, InklingConvState, InklingSconvMetadata
from .short_conv import InklingShortConv


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    effective_n = (positions + 1).to(torch.float32)
    return 1.0 + alpha * torch.log(torch.clamp(effective_n / float(n_floor), min=1.0))


class RelLogitsProj(nn.Module):
    """Project the per-head relative branch ``r`` to per-distance logits."""

    def __init__(self, d_rel: int, rel_extent: int) -> None:
        super().__init__()
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent), requires_grad=False)

    def forward(self, r_out: torch.Tensor) -> torch.Tensor:
        # r_out: (T, num_heads, d_rel) -> (T, num_heads, rel_extent)
        return torch.einsum("thd,de->the", r_out, self.proj)


class InklingAttention(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        config: InklingModelConfig,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rel_extent: int,
        local_extent: int,
        is_local: bool,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        conv_owner: InklingConvState,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.d_rel = config.d_rel
        self.log_scaling_n_floor = config.log_scaling_n_floor
        self.log_scaling_alpha = config.log_scaling_alpha
        # q/k are per-head RMS-normed (unit norm), so Inkling scales by 1/head_dim.
        self.scaling = 1.0 / head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.num_total_heads = num_heads
        self.num_total_kv_heads = num_kv_heads
        assert self.num_total_heads % tp_size == 0
        self.num_heads = self.num_total_heads // tp_size
        if self.num_total_kv_heads >= tp_size:
            assert self.num_total_kv_heads % tp_size == 0
        else:
            assert tp_size % self.num_total_kv_heads == 0
        self.num_kv_heads = max(1, self.num_total_kv_heads // tp_size)
        # When tp_size > num_kv_heads the K/V projections are padded up to
        # tp_size heads so each rank gets at least one (GQA replication).
        kv_total_for_sizing = max(self.num_total_kv_heads, tp_size)

        self.qkvr = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[
                head_dim * self.num_total_heads,
                head_dim * kv_total_for_sizing,
                head_dim * kv_total_for_sizing,
                self.d_rel * self.num_total_heads,
            ],
            bias=config.q_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            input_size=head_dim * self.num_total_heads,
            output_size=config.hidden_size,
            bias=config.o_bias,
            quant_config=quant_config,
            # reduce_results=False: the partial output is all-reduced below
            # (one-shot custom AR) so the attention-output sconv can run on the
            # full hidden width fused with the residual add + rmsnorm.
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_extent = local_extent if is_local else rel_extent
        self.local_extent = local_extent if is_local else None
        self.rel_logits_proj = RelLogitsProj(self.d_rel, self.rel_extent)
        self.q_norm = InklingRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = InklingRMSNorm(head_dim, eps=config.rms_norm_eps)

        # Short convolution on the K/V streams (per-head-width, TP sharded),
        # applied after the qkvr projection and before q/k norm.
        kv_conv_dim = self.num_kv_heads * head_dim
        self.conv_owner = conv_owner
        self.k_sconv = InklingShortConv(
            kv_conv_dim, config.sconv_kernel_size, owner=conv_owner, stream_idx=_K
        )
        self.v_sconv = InklingShortConv(
            kv_conv_dim, config.sconv_kernel_size, owner=conv_owner, stream_idx=_V
        )

        # FA4 left/right window; right=0 keeps it causal. local_extent-1 mirrors
        # the source (sliding_window_size - 1).
        self.window_size: tuple[int, int] = (
            (local_extent - 1, 0) if is_local else (-1, -1)
        )
        # Static per-layer-type KV length bound for the split heuristic: local
        # layers never see more than the sliding window.
        vllm_config = get_current_vllm_config()
        self._max_kv_len = (
            local_extent if is_local else vllm_config.model_config.max_model_len
        )

        # ---- KV-cache wiring (reuse FlashAttentionBackend for metadata) ----
        cache_config = vllm_config.cache_config
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        self.register_buffer("k_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("v_scale", torch.ones((), dtype=torch.float32))

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache

        register_fa4_warmup(
            InklingFA4WarmupConfig(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                rel_extent=self.rel_extent,
                window_size=self.window_size,
                is_local=self.is_local,
                max_kv_len=self._max_kv_len,
                dtype=vllm_config.model_config.dtype,
                kv_dtype=self.kv_cache_torch_dtype,
                block_size=vllm_config.cache_config.block_size,
                max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
                max_num_batched_tokens=(
                    vllm_config.scheduler_config.max_num_batched_tokens
                ),
            )
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return FlashAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.block_size
        if self.is_local:
            assert self.local_extent is not None
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                dtype=self.kv_cache_torch_dtype,
                sliding_window=self.local_extent,
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
        )

    def _split_kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = self.kv_cache.transpose(1, 2).split(
            self.head_dim, dim=-1
        )
        return (
            canonicalize_singleton_dim_strides(key_cache),
            canonicalize_singleton_dim_strides(value_cache),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        log_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        qkvr, _ = self.qkvr(hidden_states)

        attn_metadata = get_forward_context().attn_metadata
        attn_output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=qkvr.dtype,
            device=qkvr.device,
        )
        if not isinstance(attn_metadata, dict):
            attn_output.zero_()
        else:
            conv_meta = attn_metadata[self.conv_owner.prefix]
            md = attn_metadata[self.prefix]
            assert isinstance(conv_meta, InklingSconvMetadata)
            fa_md = cast(FlashAttentionMetadata, md)
            assert self.kv_cache.numel() > 0
            assert self.conv_owner.kv_cache.numel() > 0
            # One launch: K/V sconv (conv-cache insert + conv + residual),
            # Q/K per-head rmsnorm, and the attention KV-cache write. K/V are
            # consumed via the KV cache; only normed q is materialized.
            key_cache, value_cache = self._split_kv_cache()
            off_k, _ = self.conv_owner.stream_ranges[_K]
            off_v, _ = self.conv_owner.stream_ranges[_V]
            q, rel_logits = fused_qkvr_prep(
                qkvr,
                self.k_sconv.weight.squeeze(1),
                self.v_sconv.weight.squeeze(1),
                self.q_norm.weight,
                self.k_norm.weight,
                self.rel_logits_proj.proj,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.d_rel,
                self.conv_owner.kv_cache,
                key_cache,
                value_cache,
                positions,
                conv_meta.block_table,
                conv_meta.seq_idx,
                conv_meta.slot_mapping,
                conv_meta.query_start,
                fa_md.slot_mapping,
                off_k,
                off_v,
                self.conv_owner.block_size,
                log_scaling if not self.is_local else None,
            )
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            self._attention(q, rel_logits, attn_output)

        flat = attn_output.view(num_tokens, -1)
        output, _ = self.wo_ud(flat)
        return output

    @eager_break_during_capture
    def _attention(
        self,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)
        md = cast(FlashAttentionMetadata, attn_metadata[self.prefix])

        nt = md.num_actual_tokens
        key_cache, value_cache = self._split_kv_cache()
        max_seqlen_q = bucket_max_seqlen_q(md.max_query_len)
        num_splits = inkling_fa4_num_splits(
            is_local=self.is_local,
            batch_size=md.seq_lens.shape[0],
            max_query_len=max_seqlen_q,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            max_kv_len=self._max_kv_len,
        )
        inkling_fa4_rel_attention(
            q[:nt],
            key_cache,
            value_cache,
            block_table=md.block_table,
            cache_seqlens=md.seq_lens,
            cu_seqlens_q=md.query_start_loc,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=self.scaling,
            causal=True,
            window_size=self.window_size,
            rel_extent=self.rel_extent,
            rel_logits=rel_logits[:nt],
            num_splits=num_splits,
            out=output[:nt],
        )
