# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD-specific MLA wrapper for Kimi-K3."""

from typing import cast

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.models.kimi_k3.amd.ops.fused_mla_qk_prep import (
    fused_mla_qk_prep,
    make_identity_rope_cache,
    supports_fused_qk_prep,
)
from vllm.platforms import current_platform


class KimiK3MultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    """Kimi-K3 MLA wrapper with eager AITER q/kv RMSNorm fusion and a fused
    decode Q-prep path.

    Both fusions are kept here rather than in the shared MLA layer so only AMD
    Kimi-K3 changes, mirroring the q/kv RMSNorm fusion this class already owns.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_eager_qk_rmsnorm_fusion = bool(rocm_aiter_ops.is_enabled())
        self._fused_qk_prep = bool(
            rocm_aiter_ops.is_mla_enabled()
            and supports_fused_qk_prep(
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.num_heads,
                self.mla_attn.kv_cache_dtype,
            )
        )
        self._identity_rope: tuple[torch.Tensor, torch.Tensor] | None = None

    def _normalize_q_kv(
        self,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_layernorm = cast(RMSNorm, self.q_a_layernorm)
        kv_layernorm = cast(RMSNorm, self.kv_a_layernorm)

        if self._use_eager_qk_rmsnorm_fusion and not torch.compiler.is_compiling():
            return torch.ops.vllm.fused_mla_dual_rms_norm(
                q_c,
                q_layernorm.weight,
                kv_c,
                kv_layernorm.weight,
                q_layernorm.variance_epsilon,
                kv_layernorm.variance_epsilon,
            )

        return q_layernorm(q_c), kv_layernorm(kv_c)

    def _try_fused_decode(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        output_shape: tuple[int, int],
    ) -> torch.Tensor | None:
        """Decode-only MLA via one fused AITER launch, or ``None`` to fall back.

        Replaces the fp8 KV-cache write plus the ``[ql_nope | q_pe]`` concat and
        that query's static fp8 quant with a single
        ``fused_qk_rope_concat_and_cache_mla``. On an untouched upstream build
        the concat and quant arrive already fused by inductor into one
        ``triton_poi_fused__to_copy_cat_clamp_mul_reciprocal_view`` kernel, so
        this collapses two launches into one.

        Returning ``None`` leaves the stock ``MLAAttention`` path untouched,
        which is what every unsupported shape, batch mix and backend option
        does.
        """
        attn = self.mla_attn
        attn_metadata, layer, kv_cache, slot_mapping = get_attention_context(
            attn.layer_name
        )

        # Profile / dummy runs, and any batch that is not purely decode: the
        # fused call covers only the MQA slice, so a mixed batch's prefill rows
        # would never reach the KV cache.
        if attn_metadata is None or kv_cache.numel() == 0 or slot_mapping is None:
            return None
        num_decode = getattr(attn_metadata, "num_decode_tokens", None)
        if num_decode is None or attn_metadata.num_actual_tokens != num_decode:
            return None

        # The fused kernel writes q_out for slot >= 0 only, so padded rows would
        # be garbage after a DCP query all-gather. PCP and HiSparse additionally
        # rewrite the KV inputs/slots before the standalone write, which this
        # path bypasses.
        impl = attn.impl
        if impl.dcp_world_size > 1 or attn.use_pcp or attn.hisparse_cache is not None:
            return None

        # Head padding would change q_out's head count.
        if attn.q_pad_num_heads is not None:
            return None

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # (B, N, P) -> (N, B, P)
        q_nope_t = q_nope.transpose(0, 1)

        # Mirror MLAAttention.forward_impl's W_UK dispatch, *including its
        # precedence* -- fp4 is checked before fp8 there, and both flags can be
        # set at once. ql_nope must come out bit-identical to what the unfused
        # path feeds its concat+quant, because this fusion only replaces what
        # happens after the bmm. Both variants return bf16, which is what the
        # AITER kernel consumes.
        if attn.is_aiter_triton_fp4_bmm_enabled:
            from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

            ql_nope = batched_gemm_a16wfp4(
                q_nope_t,
                attn.W_K,
                attn.W_K_scale,
                transpose_bm=True,
                prequant=True,
                y_scale=layer._q_scale,
            )
        elif (
            not attn.is_aiter_triton_fp8_bmm_enabled
            and not attn.is_amx_bmm_enabled
            and attn.W_UK_T is not None
        ):
            B, N = q_nope.shape[0], q_nope.shape[1]
            L = attn.W_UK_T.shape[-1]
            ql_nope = q_nope_t.new_empty((B, N, L))
            torch.bmm(q_nope_t, attn.W_UK_T, out=ql_nope.transpose(0, 1))
        else:
            # fp8-bmm and AMX variants are not reproduced here.
            return None

        if self._identity_rope is None:
            self._identity_rope = make_identity_rope_cache(
                self.qk_rope_head_dim, kv_c_normed.dtype, kv_c_normed.device
            )
        cos_cache, sin_cache = self._identity_rope

        # An fp8 KV cache is allocated as uint8 and re-viewed as fp8 before use;
        # the AITER kernel rejects the raw uint8 dtype outright.
        fp8_dtype = current_platform.fp8_dtype()
        if kv_cache.dtype != fp8_dtype:
            kv_cache = kv_cache.view(fp8_dtype)

        q_out = fused_mla_qk_prep(
            ql_nope,
            q_pe,
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            layer._k_scale,
            layer._q_scale,
            positions,
            cos_cache,
            sin_cache,
            head_size=attn.kv_lora_rank + self.qk_rope_head_dim,
            q_out_dtype=current_platform.fp8_dtype(),
        )

        attn_out, _ = impl.forward_mqa(q_out, kv_cache, attn_metadata, attn)
        output = q.new_empty(output_shape)
        attn._v_up_proj(attn_out, out=output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None

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
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            q_proj_input, kv_c_normed = self._normalize_q_kv(q_c, kv_c)
            q_proj_layer = self.q_b_proj
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_c_normed = self.kv_a_layernorm(kv_c)
            q_proj_layer = self.q_proj
            q_proj_input = hidden_states

        # Add head dim of 1 to k_pe.
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

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        q_dcp_replicated = None
        if self.dcp_q_replicate:
            q_dcp_replicated, q = q, q_proj_layer._local_view(q)

        output_shape = (hidden_states.shape[0], self.num_heads * self.v_head_dim)

        attn_out = None
        if (
            self._fused_qk_prep
            and q_dcp_replicated is None
            and not torch.compiler.is_compiling()
        ):
            attn_out = self._try_fused_decode(
                q, kv_c_normed, k_pe, positions, output_shape
            )

        if attn_out is None:
            attn_out = self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=output_shape,
                q_dcp_replicated=q_dcp_replicated,
            )

        if self.g_proj is not None:
            attn_out = attn_out * self.g_proj(hidden_states)[0].sigmoid()

        return self.o_proj(attn_out)[0]
