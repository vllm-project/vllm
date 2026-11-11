# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm import ir
from vllm.config import CacheConfig
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.index_group import (
        SparseMLAIndexGroupBuilder,
    )


@triton.jit
def _sigmoid_gate_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    x_stride,
    gate_stride,
    out_stride,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One row per program. bf16(x * bf16(sigmoid(gate))) in fp32, the same
    # roundings as the eager ``x * gate.sigmoid()``.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    g = tl.load(gate_ptr + row * gate_stride + cols, mask=mask).to(tl.float32)
    x = tl.load(x_ptr + row * x_stride + cols, mask=mask).to(tl.float32)
    s = (1.0 / (1.0 + tl.exp(-g))).to(gate_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + row * out_stride + cols, x * s, mask=mask)


def mla_output_gate(
    attn_out: torch.Tensor, gate: torch.Tensor, fused: bool = False
) -> torch.Tensor:
    """attn_out * sigmoid(gate); ``fused`` runs it as one Triton launch instead
    of separate sigmoid and mul kernels."""
    if not fused:
        return attn_out * gate.sigmoid()
    assert attn_out.ndim == 2 and attn_out.shape == gate.shape
    assert attn_out.stride(-1) == 1 and gate.stride(-1) == 1
    num_tokens, n = attn_out.shape
    out = torch.empty_like(attn_out)
    if num_tokens > 0:
        _sigmoid_gate_kernel[(num_tokens,)](
            attn_out,
            gate,
            out,
            attn_out.stride(0),
            gate.stride(0),
            out.stride(0),
            N=n,
            BLOCK_N=triton.next_power_of_2(n),
            num_warps=4,
        )
    return out


def _bounded_q_kv_rmsnorm(
    q_c: torch.Tensor,
    kv_c: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # An op so the token-count dispatch runs at runtime, not at trace time.
    if q_c.shape[0] <= max_tokens:
        return fused_q_kv_rmsnorm(q_c, kv_c, q_weight, kv_weight, eps)
    return ir.ops.rms_norm(q_c, q_weight, eps), ir.ops.rms_norm(kv_c, kv_weight, eps)


def _bounded_q_kv_rmsnorm_fake(
    q_c: torch.Tensor,
    kv_c: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(q_c.shape, dtype=q_c.dtype, device=q_c.device), torch.empty(
        kv_c.shape, dtype=kv_c.dtype, device=kv_c.device
    )


direct_register_custom_op(
    op_name="mla_bounded_q_kv_rmsnorm",
    op_func=_bounded_q_kv_rmsnorm,
    fake_impl=_bounded_q_kv_rmsnorm_fake,
)


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module | None
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None
    g_proj: torch.nn.Module | None = None
    index_group_builder: "SparseMLAIndexGroupBuilder | None" = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

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
        skip_topk: bool = False,
        non_causal_multi_token_decode: bool = False,
        allow_short_prefill_indexer_scoring_skip: bool = False,
        fuse_qkv_rmsnorm: bool = False,
        fuse_output_gate: bool = False,
        fuse_qkv_rmsnorm_max_tokens: int | None = None,
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
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.g_proj = mla_modules.g_proj

        # Whether to skip top-k token selection computation in this layer.
        # When True, the indexer will not be called, and the layer will reuse
        # the topk_tokens buffer written by a previous layer in the same pass.
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = skip_topk
        # When True, fuse the q_a and kv_a RMSNorms into a single kernel launch
        # (MLA layers with q-LoRA). Opt-in; default False preserves other models.
        self.fuse_qkv_rmsnorm = fuse_qkv_rmsnorm
        # With fuse_qkv_rmsnorm, batches above this many tokens keep the two
        # separate RMSNorms (None: always fused).
        self.fuse_qkv_rmsnorm_max_tokens = fuse_qkv_rmsnorm_max_tokens
        # When True, apply the g_proj output gate with one fused Triton
        # sigmoid-multiply instead of separate sigmoid and mul launches.
        # Opt-in; default False preserves other models.
        self.fuse_output_gate = fuse_output_gate
        # qrep is active when the query projection is a DCP-group-sharded layer
        # that materializes the full group head set locally.
        q_proj_layer = self.q_b_proj if self.q_lora_rank is not None else self.q_proj
        self.dcp_q_replicate = getattr(q_proj_layer, "qrep_active", False)
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
            dcp_q_replicate=self.dcp_q_replicate,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=mla_modules.topk_indices_buffer,
            index_group_builder=mla_modules.index_group_builder,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )
        indexer_op = getattr(self.indexer, "indexer_op", None)
        if indexer_op is not None and hasattr(
            indexer_op, "dense_mha_metadata_layer_name"
        ):
            enable_short_prefill_scoring_skip = (
                allow_short_prefill_indexer_scoring_skip
                and not self.skip_topk
                and not getattr(indexer_op, "use_pcp", False)
                and current_platform.is_cuda()
            )
            # The indexer and main MLA use independent decode thresholds and
            # may classify the same short extend differently. Bind the main
            # MLA layer name so the eager indexer op can check whether the
            # batch's top-k indices will be consumed.
            # PCP is excluded because indexer cache/scoring ownership differs
            # across ranks and the no-consumer invariant has not been
            # established there.
            indexer_op.dense_mha_metadata_layer_name = (
                self.mla_attn.layer_name if enable_short_prefill_scoring_skip else ""
            )
        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        skip_o_proj: bool = False,
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
            q_proj_layer = self.q_b_proj
            if not self.fuse_qkv_rmsnorm:
                q_c = self.q_a_layernorm(q_c)
            q_proj_input = q_c
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q_proj_layer = self.q_proj
            q_proj_input = hidden_states

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        if self.fuse_qkv_rmsnorm and q_c is not None:
            # Fuse q_a + kv_a RMSNorm into one kernel launch (q_c is still the
            # pre-norm projection output here).
            assert self.q_a_layernorm is not None
            if self.fuse_qkv_rmsnorm_max_tokens is None:
                q_proj_input, kv_c_normed = fused_q_kv_rmsnorm(
                    q_c,
                    kv_c,
                    self.q_a_layernorm.weight.data,
                    self.kv_a_layernorm.weight.data,
                    self.q_a_layernorm.variance_epsilon,
                )
            else:
                q_proj_input, kv_c_normed = torch.ops.vllm.mla_bounded_q_kv_rmsnorm(
                    q_c,
                    kv_c,
                    self.q_a_layernorm.weight.data,
                    self.kv_a_layernorm.weight.data,
                    self.q_a_layernorm.variance_epsilon,
                    self.fuse_qkv_rmsnorm_max_tokens,
                )
            q_c = q_proj_input
        else:
            kv_c_normed = self.kv_a_layernorm(kv_c)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        q = q_proj_layer(q_proj_input)[0]
        heads = self.num_heads
        if self.dcp_q_replicate:
            heads *= q_proj_layer.group_size
        q = q.view(-1, heads, self.qk_head_dim)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
        if self.is_sparse:
            self.mla_attn.impl.record_logical_topk_ready()  # type: ignore[attr-defined]

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        q_dcp_replicated = None
        if self.dcp_q_replicate:
            q_dcp_replicated, q = q, q_proj_layer._local_view(q)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            q_dcp_replicated=q_dcp_replicated,
        )

        if self.g_proj is not None:
            attn_out = mla_output_gate(
                attn_out, self.g_proj(hidden_states)[0], self.fuse_output_gate
            )

        if skip_o_proj:
            return attn_out
        return self.o_proj(attn_out)[0]
