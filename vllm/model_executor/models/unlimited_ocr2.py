# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unlimited-OCR-2 DASH with full decode and breakable prefill CUDA graphs."""

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.unlimited_ocr2 import (
    DashRouter,
    dash_attention,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionMetadata,
)
from vllm.v1.attention.ops.unlimited_ocr2 import cache_dash_prefill, paged_dash_decode

TOKEN_ROLES_KEY = "unlimited_ocr2_token_roles"


class UnlimitedOCR2Attention(nn.Module):
    def __init__(
        self,
        vllm_config,
        config,
        hidden_size,
        num_heads,
        cache_config=None,
        quant_config=None,
        reduce_results=True,
        prefix="",
        **kwargs,
    ):
        # Keep main KV and one shared router/metadata head per TP rank.
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp = get_tensor_model_parallel_world_size()
        self.num_heads = num_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dash_backend = config.unlimited_ocr2_attention_backend
        self.sp = config.sparse_attention_config
        self.layer_idx = int(prefix.split(".")[-2])
        index_dim = int(self.sp["sparse_index_dim"])
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
        )
        self.dash_router = DashRouter(hidden_size, index_dim, config.rms_norm_eps)
        window = vllm_config.model_config.rswa_window
        attention_cls = RSWAAttention if window is not None else Attention
        attention_kwargs = {"rswa_window": window} if window is not None else {}
        self.attn = attention_cls(
            self.num_kv_heads + 1,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads + 1,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
            attn_backend=TritonAttentionBackend,
            **attention_kwargs,
        )

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view_as(k)
        iq, ik = self.dash_router(hidden_states)
        roles = get_forward_context().additional_kwargs.get(TOKEN_ROLES_KEY)
        result = torch.empty_like(q)
        self._attention(q, k, v, iq, ik, roles, result)
        return self.o_proj(result.flatten(1))[0]

    @eager_break_during_capture
    def _attention(self, q, k, v, iq, ik, roles, result):
        context = get_forward_context()
        all_metadata = context.attn_metadata
        if all_metadata is None:
            result.zero_()
            return
        assert isinstance(all_metadata, dict)
        metadata = all_metadata[self.attn.layer_name]
        assert isinstance(metadata, TritonAttentionMetadata)
        if context.is_dummy_run and metadata.max_query_len > 1:
            result.zero_()
            return
        assert roles is not None
        assert isinstance(context.slot_mapping, dict)
        slots = context.slot_mapping[self.attn.layer_name]
        cache = self.attn.kv_cache
        if metadata.max_query_len == 1:
            result.copy_(
                paged_dash_decode(
                    q,
                    k,
                    v,
                    iq,
                    ik,
                    roles,
                    cache,
                    metadata.block_table,
                    metadata.seq_lens,
                    slots,
                    self.dash_router.head_cls,
                    self.sp,
                    self.layer_idx,
                    self.max_model_len,
                    metadata.rswa_prefix_lens,
                    backend=self.dash_backend,
                    cached_summaries=True,
                )
            )
        else:
            starts = metadata.query_start_loc.tolist()
            lengths = metadata.seq_lens.tolist()
            result.zero_()
            for request, length in enumerate(lengths):
                start, end = starts[request : request + 2]
                if start == end:
                    continue
                if end - start == 1:
                    result[start:end] = paged_dash_decode(
                        q[start:end],
                        k[start:end],
                        v[start:end],
                        iq[start:end],
                        ik[start:end],
                        roles[start:end],
                        cache,
                        metadata.block_table[request : request + 1],
                        metadata.seq_lens[request : request + 1],
                        slots[start:end],
                        self.dash_router.head_cls,
                        self.sp,
                        self.layer_idx,
                        self.max_model_len,
                        metadata.rswa_prefix_lens[request : request + 1]
                        if metadata.rswa_prefix_lens is not None
                        else None,
                        backend=self.dash_backend,
                        cached_summaries=True,
                    )
                else:
                    assert length == end - start, "DASH requires complete prefills"
                    result[start:end] = dash_attention(
                        q[start:end],
                        k[start:end],
                        v[start:end],
                        iq[start:end],
                        ik[start:end],
                        roles[start:end],
                        self.dash_router.head_cls,
                        self.sp,
                        self.layer_idx,
                        backend=self.dash_backend,
                    )
                    cache_dash_prefill(
                        k[start:end],
                        v[start:end],
                        ik[start:end],
                        roles[start:end],
                        slots[start:end].long(),
                        cache,
                        self.sp,
                        self.dash_router.head_cls,
                    )
