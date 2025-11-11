# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.utils.flashinfer import has_flashinfer
from vllm.forward_context import get_forward_context
from dataclasses import dataclass

import vllm.envs as envs
import torch

from vllm.attention.layer import MLAAttention
from vllm.config import CacheConfig
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization import QuantizationConfig


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None


@CustomOp.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(CustomOp):
    """MLA layer registered as CustomOp to allow OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently MLA ignores the enable/disable mechanism of CustomOp
    because there is only one in-tree implementation in forward_native.
    TODO: implement this with a new PluggableLayer mechanism.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.is_sparse = mla_modules.is_sparse
        self.fuse_rope_fp8 = (
            envs.VLLM_FLASHINFER_FUSE_MLA_ROPE_FP8 and 
            has_flashinfer() and
            cache_config is not None and
            cache_config.cache_dtype == "fp8"
        )


        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            fuse_rope_fp8=self.fuse_rope_fp8
        )

        self.prefix = prefix

    def forward_native(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)


        attn_metadata = get_forward_context().attn_metadata
        # print(f"{attn_metadata=}")
        if isinstance(attn_metadata, dict):
            # print(f"{attn_metadata=}")
            attn_metadata = attn_metadata[f"{self.prefix}.attn"]
        if self.rotary_emb is not None:
            decode_only = attn_metadata is not None and attn_metadata.num_prefills == 0
            if decode_only:
                print("decode only")
            if self.fuse_rope_fp8 and decode_only:
                q_nope = q[..., : self.qk_nope_head_dim]
                print(f"{q_nope.shape=}, {kv_c_normed.shape=}")
                # TODO: separate q_rope and decode_ql_nope and return as tuple
                # TODO: calculate decode_ql_nope here (fix shape potentially)
                decode_ql_nope = torch.bmm(q[..., :self.qk_nope_head_dim], self.kv_b_proj.weight)
                q_rope, k_pe, decode_ql_nope, kv_c_normed  = self.rotary_emb(
                    positions, q[..., self.qk_nope_head_dim :], k_pe, None, decode_ql_nope, kv_c_normed
                )
                q = torch.cat(q_rope, decode_ql_nope, dim=-1)
                print(f"{q.dtype=}, {k_pe.dtype=} {kv_c_normed.dtype=}")
            else:
                q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                    positions, q[..., self.qk_nope_head_dim :], k_pe
                )
        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(hidden_states, q_c, positions, self.rotary_emb)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        return self.o_proj(attn_out)[0]

    def forward_cuda(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)
