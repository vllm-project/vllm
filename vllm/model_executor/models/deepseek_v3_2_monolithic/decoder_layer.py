# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Monolithic decoder layer for DeepSeek V3.2 on SM100 (Blackwell).
Direct kernel calls, no module wrappers for norms.
Gate weight inlined, FusedMoE kept for quantized expert kernels.
"""

from __future__ import annotations

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

from .allreduce_rms import AllReduceRMSParams, allreduce_add_rms_norm
from .attention import MonolithicMLAAttention
from .ops import fused_norm_rope, fused_q, rms_norm
from .sparse_indexer import sparse_attn_indexer

_side_stream: torch.cuda.Stream | None = None


def _get_side_stream() -> torch.cuda.Stream:
    """Lazily created CUDA stream shared by all decoder layers."""
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
    return _side_stream


class MonolithicDecoderLayer(nn.Module):
    """
    Single decoder layer: norm -> attn -> norm -> MoE/MLP.
    Norms are raw weight + direct kernel call.
    Gate inlined as raw weight, experts kept as FusedMoE for quantization.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        topk_indices_buffer: torch.Tensor,
        topk_page_indices_buffer: torch.Tensor,
        prefix: str = "",
        fi_params: AllReduceRMSParams | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.tp_size = get_tensor_model_parallel_world_size()
        self._fi_params = fi_params

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.indexer_workspace_size = get_max_prefill_buffer_size(vllm_config)
        self.max_model_len = vllm_config.model_config.max_model_len

        # LayerNorm weights (raw)
        dtype = torch.get_default_dtype()
        self.input_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=dtype)
        )
        self.post_attention_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=dtype)
        )

        # Fused QKV A-projection lives inside self_attn namespace
        # for weight loading compatibility with original checkpoint paths
        from vllm.model_executor.models.deepseek_v2 import (
            DeepSeekV2FusedQkvAProjLinear,
        )

        self.self_attn = nn.Module()
        self.self_attn.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            config.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.fused_qkv_a_proj",
        )

        # MLA Attention — disable AllReduce in o_proj when using fused path
        self.attn = MonolithicMLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            cache_config=cache_config,
            quant_config=quant_config,
            topk_indices_buffer=topk_indices_buffer,
            topk_page_indices_buffer=topk_page_indices_buffer,
            prefix=f"{prefix}.self_attn",
        )
        self.attn.o_proj.reduce_results = False

        # MoE or Dense MLP
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        self.is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        from vllm.model_executor.models.deepseek_v2 import (
            DeepseekV2MLP,
            DeepseekV2MoE,
        )

        if self.is_moe:
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.mlp.skip_final_allreduce = True
            self.mlp.skip_scale_and_add = True
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.mlp",
            )

    def fuse_indexer_weights(self) -> None:
        """Fuse Step 1 and Step 3 BF16 linears used by the monolithic path.

        Call after model weights are loaded.
        """
        attn = self.attn
        qkv_a = self.self_attn.fused_qkv_a_proj.weight.data  # [2112, 7168]
        wk = attn.indexer_wk.weight.data  # [128, 7168]
        wp = attn.indexer_weights_proj.weight.data  # [64, 7168]
        if not (qkv_a.dtype == wk.dtype == wp.dtype):
            raise ValueError(
                "Cannot fuse Step 1 weights: expected matching dtypes for "
                "fused_qkv_a_proj, indexer_wk, and indexer_weights_proj."
            )
        self._fused_step1_hidden_w = nn.Parameter(
            torch.cat([qkv_a, wk, wp], dim=0),  # [2304, 7168]
            requires_grad=False,
        )
        self._step1_split_sizes = [
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            wk.shape[0],
            wp.shape[0],
        ]

        wq_b = attn.indexer_wq_b.weight.data
        q_b = attn.q_b_proj.weight.data
        if wq_b.dtype != q_b.dtype:
            raise ValueError(
                "Cannot fuse Step 3 weights: expected matching dtypes for "
                "indexer_wq_b and q_b_proj."
            )
        self._fused_step3_q_w = nn.Parameter(
            torch.cat([wq_b, q_b], dim=0),
            requires_grad=False,
        )
        self._step3_index_q_dim = wq_b.shape[0]

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.forward_context import get_forward_context

        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata
        mla = self.attn.mla_attn
        mla_attn_metadata = None
        slot_mapping = None
        if isinstance(attn_metadata, dict):
            idx_meta = attn_metadata[self.attn.indexer_k_cache.prefix]
            mla_attn_metadata = attn_metadata[mla.layer_name]
            # Indexer and MLA caches share the same block_size and track
            # the same requests, so their slot_mappings are identical.
            slot_mapping = idx_meta.slot_mapping

        if slot_mapping is not None:
            indexer_k_cache = self.attn.indexer_k_cache.kv_cache
            mla_kv_cache = mla.kv_cache
            mla_k_scale = mla._k_scale
        else:
            indexer_k_cache = None
            mla_kv_cache = None
            mla_k_scale = None

        # Input norm + residual
        # When fused_allreduce_rms is enabled, hidden_states arriving from
        # the previous layer is the *unreduced* MLP/MoE output. We fuse
        # AllReduce + residual-add + RMSNorm into a single kernel.
        if residual is None:
            # First layer: hidden_states is from embed_tokens (already
            # fully materialised), no allreduce needed.
            residual = hidden_states
            hidden_states = rms_norm(
                hidden_states, self.input_layernorm_weight, self.rms_norm_eps
            )
        else:
            hidden_states, residual = allreduce_add_rms_norm(
                hidden_states,
                residual,
                self.input_layernorm_weight,
                self.rms_norm_eps,
                self._fi_params,
            )

        if not hasattr(self, "_fused_step1_hidden_w") or not hasattr(
            self, "_fused_step3_q_w"
        ):
            raise RuntimeError(
                "Monolithic decoder fused weights are not initialized. "
                "Call fuse_indexer_weights() after weight loading."
            )

        # Step 1. hidden_states -> q_c, kv_c, k_pe, index_k, index_weights
        step1_out = torch.mm(hidden_states, self._fused_step1_hidden_w.T)
        q_c, kv_c, k_pe, index_k, index_weights = step1_out.split(
            self._step1_split_sizes,
            dim=-1,
        )

        # Step 2. Q RMS norm
        #         + KV RMS norm + KV RoPE + MLA cache write
        #         + Index K layer norm + RoPE + FP8 quant + cache write
        #         + Init topk indices
        q_c = fused_norm_rope(
            positions,
            # Q RMS norm
            q_c,
            self.attn.q_a_layernorm_weight,
            self.rms_norm_eps,
            # KV RMS norm
            kv_c,
            self.attn.kv_a_layernorm_weight,
            self.attn.rms_norm_eps,
            # KV RoPE
            k_pe,
            self.attn.rotary_emb.cos_sin_cache,
            # Index K layer norm + RoPE
            index_k,
            self.attn.indexer_k_norm.weight,
            self.attn.indexer_k_norm.bias,
            self.attn.rms_norm_eps,
            self.attn.indexer_rope_emb.cos_sin_cache,
            # Top k indices
            self.attn.topk_indices_buffer,
            # Fused cache writes (single slot_mapping for both caches)
            slot_mapping=slot_mapping,
            indexer_k_cache=indexer_k_cache,
            mla_kv_cache=mla_kv_cache,
            mla_kv_cache_dtype=self.attn.mla_attn.kv_cache_dtype,
            mla_k_scale=mla_k_scale,
        )

        # Step 3. q_c -> index_q, q
        step3_out = torch.mm(q_c, self._fused_step3_q_w.T)
        index_q, q = step3_out.split(
            [self._step3_index_q_dim, step3_out.shape[-1] - self._step3_index_q_dim],
            dim=-1,
        )
        index_q = index_q.view(-1, self.attn.index_n_heads, self.attn.index_head_dim)
        q = q.view(-1, self.attn.num_local_heads, self.attn.qk_head_dim)

        # Step 4. Second fused stage:
        #   Q RoPE + Index Q RoPE + Index Q FP8 + index-weight scaling
        #   + W_UK_T absorption + MQA FP8 query packing.
        q_nope, q_pe = q.split([mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(0, 1)
        ql_nope = torch.bmm(q_nope, mla.W_UK_T)
        ql_nope = ql_nope.transpose(0, 1)

        assert mla.kv_cache_dtype.startswith("fp8")
        assert mla.impl.supports_quant_query_input

        index_q_fp8, index_weights, mqa_q = fused_q(
            positions,
            q_pe,
            self.attn.rotary_emb.cos_sin_cache,
            index_q,
            self.attn.indexer_rope_emb.cos_sin_cache,
            ql_nope,
            mla._q_scale,
            index_weights,
            self.attn.indexer_softmax_scale,
            self.attn.index_n_heads**-0.5,
        )

        # Step 5. Sparse indexer.
        # The FP8 quant + cache write for index_k is already done in
        # fused_norm_rope (step 2) when slot_mapping is available.
        sparse_attn_indexer(
            self.attn.indexer_k_cache.prefix,
            self.attn.indexer_k_cache.kv_cache,
            index_q_fp8,
            index_weights,
            self.attn.topk_tokens,
            self.attn.index_head_dim,
            self.max_model_len,
            self.indexer_workspace_size,
            self.attn.topk_indices_buffer,
            self.attn.topk_page_indices_buffer,
        )

        # Step 6. MLA sparse decode attention (inlined).
        # The KV cache update was already done in fused_norm_rope (step 2).
        output_shape = (hidden_states.shape[0], mla.num_heads * mla.v_head_dim)
        output_dtype = mla.W_UV.dtype
        if mla_attn_metadata is None or slot_mapping is None:
            attn_out = torch.zeros(
                output_shape,
                dtype=output_dtype,
                device=hidden_states.device,
            )
        else:
            num_actual_toks = mla_attn_metadata.num_actual_tokens
            mqa_q = mqa_q[:num_actual_toks]
            kv_cache = mla.kv_cache
            if (
                mla.kv_cache_dtype.startswith("fp8")
                and mla.kv_cache_dtype != "fp8_ds_mla"
            ):
                kv_cache = kv_cache.view(torch.float8_e4m3fn)

            attn_out, _ = mla.impl.forward_mqa(
                mqa_q,
                kv_cache,
                mla_attn_metadata,
                mla,
            )

            output = torch.empty(
                output_shape,
                dtype=output_dtype,
                device=kv_cache.device,
            )
            x = attn_out.view(-1, mla.num_heads, mla.kv_lora_rank).transpose(0, 1)
            out = output[:num_actual_toks].view(-1, mla.num_heads, mla.v_head_dim)
            out = out.transpose(0, 1)
            torch.bmm(x, mla.W_UV, out=out)
            attn_out = output

        # Step 7. Output projection (AllReduce disabled when fused).
        hidden_states, _ = self.attn.o_proj(attn_out)

        # Post-attn norm + residual
        # Fuse the o_proj AllReduce with post-attention RMSNorm.
        hidden_states, residual = allreduce_add_rms_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm_weight,
            self.rms_norm_eps,
            self._fi_params,
        )

        # MLP / MoE
        # When fused_allreduce_rms is enabled, the MLP/MoE AllReduce is
        # deferred — it will be fused with the next layer's input norm.
        if self.is_moe:
            import flashinfer

            moe = self.mlp
            experts = moe.experts
            runner = experts.runner
            quant_method = experts.quant_method

            # Specialize to the current DeepSeek V3.2-NVFP4 path:
            # no EP/PCP, no routed-input transform, monolithic TRTLLM MoE.
            assert not moe.is_sequence_parallel
            assert not experts.use_ep
            assert experts.moe_config.pcp_size == 1
            assert not runner.use_dp_chunking
            assert runner.routed_input_transform is None
            assert quant_method.is_monolithic
            assert moe.shared_experts is not None
            assert moe.gate is not None
            assert experts.routing_method_type.name == "DeepSeekV3"

            hidden_states = hidden_states.view(-1, self.hidden_size)

            use_shared_experts_stream = (
                runner.has_separate_shared_experts
                and runner.shared_experts_stream is not None
                and hidden_states.shape[0]
                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            )
            runner.use_shared_experts_stream = use_shared_experts_stream

            if use_shared_experts_stream:
                shared_experts_stream = runner.shared_experts_stream
                assert shared_experts_stream is not None
                assert experts.moe_config.disable_inplace
                hidden_states.record_stream(shared_experts_stream)
                shared_experts_stream.wait_stream(torch.cuda.current_stream())
                shared_output = None
            else:
                shared_output = moe.shared_experts(hidden_states)

            router_logits, _ = moe.gate(hidden_states)
            quant_config = experts.quant_method.moe_quant_config
            assert quant_config is not None
            input_sf = (
                quant_config.a1_gscale
                if quant_config.use_nvfp4_w4a4
                else quant_config.a1_scale
            )
            assert input_sf is not None
            a1q, a1q_scale = moe_kernel_quantize_input(
                hidden_states,
                input_sf,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
                is_fp4_scale_swizzled=quant_config.is_nvfp4_scale_swizzled,
            )
            assert a1q_scale is not None
            assert quant_config.w1_scale is not None
            assert quant_config.w2_scale is not None
            assert quant_config.g1_alphas is not None
            assert quant_config.g2_alphas is not None

            routed_output = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
                routing_logits=router_logits.to(torch.float32),
                routing_bias=experts.e_score_correction_bias,
                hidden_states=a1q,
                hidden_states_scale=a1q_scale.view(torch.float8_e4m3fn).reshape(
                    *a1q.shape[:-1], -1
                ),
                gemm1_weights=experts.w13_weight,
                gemm1_weights_scale=quant_config.w1_scale.view(torch.float8_e4m3fn),
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=experts.w2_weight,
                gemm2_weights_scale=quant_config.w2_scale.view(torch.float8_e4m3fn),
                gemm2_bias=None,
                output1_scale_scalar=experts.g1_scale_c,
                output1_scale_gate_scalar=quant_config.g1_alphas,
                output2_scale_scalar=quant_config.g2_alphas,
                num_experts=experts.global_num_experts,
                top_k=experts.top_k,
                n_group=(experts.num_expert_group or 0),
                topk_group=(experts.topk_group or 0),
                intermediate_size=experts.moe_config.intermediate_size_per_partition,
                local_expert_offset=0,
                local_num_experts=experts.local_num_experts,
                routed_scaling_factor=experts.routed_scaling_factor,
                routing_method_type=experts.routing_method_type,
                do_finalize=True,
                activation_type=activation_to_flashinfer_int(experts.activation),
            )[0]

            if use_shared_experts_stream:
                shared_experts_stream = runner.shared_experts_stream
                assert shared_experts_stream is not None
                with torch.cuda.stream(shared_experts_stream):
                    shared_output = moe.shared_experts(hidden_states)
                torch.cuda.current_stream().wait_stream(shared_experts_stream)

            assert shared_output is not None

            hidden_states = scale_and_add(
                routed_output,
                self.routed_scaling_factor,
                shared_output,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@torch.compile
def scale_and_add(x: torch.Tensor, scale: float, y: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x * scale
    z = x + y.to(torch.float32)
    return z.to(orig_dtype)
