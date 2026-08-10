# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clean Multi-head Latent Attention for Kimi-K3 (ROCm).

Duplicated from ``nvidia/mla.py``. Sole difference: ``_decode_concat_cache``
dequantizes the fp8 query for backends without ``supports_quant_query_input``
(the DSpark draft's TRITON_MLA) instead of asserting.

This is a self-contained MLA layer that owns the full attention path:

    hidden_states
      -> fused pre-attention ops (fused_qkv_a_proj / norms / q_b_proj)
      -> explicit prefill / decode split
           prefill: fused key-concat + cache-insert kernel -> run_prefill_new_tokens
                    (+ chunked-context merge); dispatched by cache dtype
                    (bf16 / plain fp8 / fp8_ds_mla)
           decode : W_UK absorb (BMM1) -> fused q-concat + cache-insert kernel
                    -> impl.forward_mqa -> W_UV up-proj (MQA)
      -> optional output gate
      -> o_proj

Unlike ``MultiHeadLatentAttentionWrapper`` (which delegates orchestration to
``MLAAttention.forward``), this class *is* the ``AttentionLayerBase``: it selects
the backend, builds the impl, registers itself in the forward context, owns the
KV cache, and absorbs ``kv_b_proj`` into ``W_UK_T`` / ``W_UV`` -- mirroring the
``DeepseekV4Attention`` structure.

K3 specifics: optional rotary embedding (disabled for the target model's NoPE
layers, enabled for DSpark) and an optional sigmoid output gate (``g_proj``).

Out of scope (extension points, not wired here): context parallelism (DCP/PCP),
sparse/indexer MLA, and the ROCm/aiter fp8/fp4 BMM fast paths.
"""

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    _init_kv_cache_quant,
    set_default_quant_scales,
    should_load_quant_weights,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.utils import replace_parameter
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.models.kimi_k3.amd.ops.fused_mla_key_concat_kv_cache import (
    fused_mla_decode_q_concat_kv_cache_insert,
    fused_mla_key_concat_ds_mla_insert,
    fused_mla_key_concat_kv_cache_insert,
    fused_mla_qkv_quant_kv_cache_fp8_insert,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import (
    is_quantized_kv_cache,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec, get_kv_quant_mode

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata

logger = init_logger(__name__)

# Below this many tokens, overlap the g_proj GEMM on the aux stream with the
# attention front-end (the GEMM is small and launch-bound, so the overlap
# hides it); at or above it, run the gate on the main stream.
_GATE_MULTI_STREAM_TOKEN_THRESHOLD = 512


@torch.compile(backend=current_platform.simple_compile_backend)
def _gate_sigmoid_mul(attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply the sigmoid output gate to a precomputed ``g_proj`` projection."""
    return attn_out * gate.sigmoid()


class MultiHeadLatentAttention(nn.Module, AttentionLayerBase):
    """Kimi-K3 Multi-head Latent Attention with optional RoPE and output gate."""

    def __init__(
        self,
        config: KimiLinearConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        use_output_gate: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
        use_rope: bool = False,
        non_causal_multi_token_decode: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.non_causal_multi_token_decode = non_causal_multi_token_decode
        # Latent "head" seen by the attention kernel / KV cache.
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.scale = self.qk_head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps
        self.layer_name = prefix

        self.rotary_emb: RotaryEmbedding | None = None
        if use_rope:
            rope_parameters = dict(config.rope_parameters)
            if rope_parameters["rope_type"] != "default":
                rope_parameters["rope_type"] = (
                    "deepseek_yarn"
                    if rope_parameters.get("apply_yarn_scaling", True)
                    else "deepseek_llama_scaling"
                )
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=rope_parameters,
                is_neox_style=False,
                dtype=torch.float32,
            )
            if rope_parameters["rope_type"] == "deepseek_yarn":
                mscale_all_dim = rope_parameters.get("mscale_all_dim", False)
                scaling_factor = rope_parameters["factor"]
                mscale = (
                    1.0
                    if scaling_factor <= 1
                    else 0.1 * float(mscale_all_dim) * math.log(scaling_factor) + 1.0
                )
                self.scale *= mscale * mscale
            # The fused epilogues read the cos/sin table directly in fp32 and run
            # the RoPE math in fp32, so there is no per-forward dtype cast (and no
            # precision loss). deepseek_yarn builds cos_sin_cache in fp32 already;
            # dtype=torch.float32 above forces it for the default rope too (the
            # DSpark draft, which has no yarn scaling).
            assert self.rotary_emb.cos_sin_cache.dtype == torch.float32, (
                "K3 fused MLA RoPE requires an fp32 cos/sin cache; got "
                f"{self.rotary_emb.cos_sin_cache.dtype}."
            )

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_heads = num_heads
        self.num_local_heads = num_heads // tp_size

        # ---- Pre-attention projections (fusable front-end) ----
        # Two query variants: a low-rank q-LoRA path (Kimi-K3) fused with the
        # kv-down proj, or an uncompressed q path (Kimi-Linear, ``q_lora_rank``
        # None) with a standalone ``q_proj`` and separate ``kv_a_proj_with_mqa``.
        if self.q_lora_rank is not None:
            # Fused q-down + kv-down projection. Replicated (disable_tp) because
            # the low-rank latents are shared across TP ranks; TP splitting
            # happens at q_b_proj / kv_b_proj. Checkpoint weights ``q_a_proj``
            # and ``kv_a_proj_with_mqa`` map onto shards 0 and 1 respectively.
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            # Uncompressed query: full-rank q_proj (TP-split over heads) plus a
            # replicated kv-down projection (shared latent across TP ranks).
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # ---- Post-attention projections ----
        self.use_output_gate = use_output_gate
        self.g_proj = (
            ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
            if use_output_gate
            else None
        )
        # Aux stream (created at the model level, DeepseekV4 convention) for
        # overlapping the g_proj GEMM with the attention front-end. None on
        # ROCm/non-cuda -> maybe_execute_in_parallel falls back to sequential.
        self.aux_stream = aux_stream
        self._gate_events = (
            [torch.cuda.Event(), torch.cuda.Event()]
            if self.g_proj is not None and current_platform.is_cuda_alike()
            else None
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # ---- Attention backend / impl / KV cache ----
        self.quant_config = quant_config
        if cache_config is not None:
            self.kv_cache_dtype = cache_config.cache_dtype
        else:
            self.kv_cache_dtype = "auto"

        dtype = torch.get_default_dtype()
        self.attn_backend = get_attn_backend(
            self.head_size,
            dtype,
            self.kv_cache_dtype,
            use_mla=True,
            use_sparse=False,
            num_heads=self.num_local_heads,
        )
        _init_kv_cache_quant(self, quant_config, prefix)
        # Unit (1.0) scale for the fused fp8 prefill path: q/k/v are cast
        # unscaled to match forward_mha (the prefill flash path does not
        # dequantize); only the cache uses _k_scale.
        self.register_buffer(
            "_one_scale", torch.ones(1, dtype=torch.float32), persistent=False
        )

        impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
        self.impl = impl_cls(  # type: ignore[assignment]
            num_heads=self.num_local_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=self.kv_cache_dtype,
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=self.kv_b_proj,
            indexer=None,
        )
        if getattr(self.impl, "dcp_world_size", -1) < 1:
            # FlashAttention requires the cp_world_size is positive and the cp_rank
            # is non negative; manually set here if not set by caller (-1 is unset)
            self.impl.dcp_world_size = 1
            self.impl.dcp_rank = 0
        self.q_pad_num_heads = getattr(self.impl, "q_pad_num_heads", None)

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        assert (
            parallel_config.decode_context_parallel_size <= 1
            and parallel_config.prefill_context_parallel_size <= 1
        ), "Kimi-K3 MultiHeadLatentAttention does not support context parallelism."
        self.prefill_backend = get_mla_prefill_backend(vllm_config)(
            num_heads=self.num_local_heads,
            scale=self.scale,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            vllm_config=vllm_config,
        )

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

    # ------------------------------------------------------------------
    # AttentionLayerBase interface
    # ------------------------------------------------------------------
    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        # TODO: Remove this mypy workaround once the K3 PR is fully merged.
        return MLAAttentionSpec(  # type: ignore[call-arg]
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=self.kv_cache_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            non_causal_multi_token_decode=self.non_causal_multi_token_decode,
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Absorb ``kv_b_proj`` into decode-time ``W_UK_T`` / ``W_UV`` bmm weights.

        ``kv_b_proj`` produces ``[k_nope; v]`` per head from the ``kv_lora_rank``
        latent. For the MQA decode path we pre-split it so that queries are
        projected into latent space by ``W_UK_T`` and the attention output is
        projected back to ``v`` by ``W_UV`` -- avoiding materializing full K/V.
        """
        kv_b_proj_weight = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=act_dtype
        ).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_local_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), f"{kv_b_proj_weight.shape=}"
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # (L, N, V) -> (N, L, V)
        replace_parameter(self, "W_UV", W_UV.transpose(0, 1), prefer_copy=True)
        # (L, N, P) -> (N, P, L)
        replace_parameter(self, "W_UK_T", W_UK.permute(1, 2, 0), prefer_copy=True)

        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

        # Precompute reciprocal scales once here (scales are final after load;
        # K3 has no runtime calculate_kv_scales path) so the fp8 fused kernels
        # in the decode/prefill hot path take a ready inverse instead of
        # launching a per-step reciprocal kernel.
        self.register_buffer(
            "_q_scale_inv", self._q_scale.reciprocal().reshape(1), persistent=False
        )
        self.register_buffer(
            "_k_scale_inv", self._k_scale.reciprocal().reshape(1), persistent=False
        )

    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor) -> None:
        """Project latent attention output back to ``v`` via ``W_UV`` (bmm)."""
        # (B, N, L) -> (N, B, L)
        x = x.view(-1, self.num_local_heads, self.kv_lora_rank).transpose(0, 1)
        out = out.view(-1, self.num_local_heads, self.v_head_dim)
        # (N, B, L) x (N, L, V) -> (N, B, V) written transposed into (B, N, V)
        torch.bmm(x, self.W_UV, out=out.transpose(0, 1))

    def _attn_read_kv_cache(self) -> torch.Tensor:
        """Latent cache as seen by the attention read kernels (decode / context).

        A plain per-tensor fp8 cache is stored as ``uint8``; view it as fp8 so
        the backend reads it as E4M3 rather than fp4/E2M1 -- the latter doubles
        the perceived head dim (``head_size * 2``) and fails the kernel's
        ``head_dim_k == head_dim_q`` check. Mirrors ``MLAAttention.forward``;
        the fp8_ds_mla layout keeps its native uint8 view.
        """
        cache = self.kv_cache
        if (
            is_quantized_kv_cache(self.kv_cache_dtype)
            and self.kv_cache_dtype != "fp8_ds_mla"
        ):
            return cache.view(current_platform.fp8_dtype())
        return cache

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _forward_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Attention front-end: fused qkv-a proj -> norms -> q_b -> attention.

        Returns the pre-gate attention output ``[num_tokens,
        num_local_heads * v_head_dim]``. On a profile/dummy run
        it returns a zeroed buffer.
        """
        if self.q_lora_rank is not None:
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_c, k_pe = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            q_c, kv_c_normed = fused_q_kv_rmsnorm(
                q_c,
                kv_c,
                self.q_a_layernorm.weight.data,
                self.kv_a_layernorm.weight.data,
                self.rms_norm_eps,
            )
            q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            # Uncompressed query: project directly (no q-LoRA, no q norm) and
            # normalize only the kv latent.
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_c_normed = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)

        attn_out = torch.empty(
            (hidden_states.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._attention(positions, q, kv_c_normed, k_pe, attn_out)
        return attn_out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Both branches produce (attn_out, gate); they differ only in whether
        # the g_proj GEMM is overlapped on the aux stream.
        g_proj = self.g_proj
        events = self._gate_events
        if (
            g_proj is not None
            and events is not None
            and self.aux_stream is not None
            and hidden_states.shape[0] < _GATE_MULTI_STREAM_TOKEN_THRESHOLD
        ):
            attn_out, gate = maybe_execute_in_parallel(
                lambda: self._forward_attn(positions, hidden_states),
                lambda: g_proj(hidden_states)[0],
                events[0],
                events[1],
                self.aux_stream,
            )
        else:
            attn_out = self._forward_attn(positions, hidden_states)
            gate = g_proj(hidden_states)[0] if g_proj is not None else None

        if gate is not None:
            attn_out = _gate_sigmoid_mul(attn_out, gate)

        # ``o_proj`` (RowParallelLinear + out-of-place all-reduce) returns a
        # fresh private tensor, so return it directly rather than copying into a
        # caller buffer -- the previous ``output[:] = ...`` convention forced an
        # extra [num_tokens, hidden] copy per layer.
        return self.o_proj(attn_out)[0]

    @eager_break_during_capture
    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_by_layer = forward_context.attn_metadata
        if attn_metadata_by_layer is None:
            attn_out.zero_()
            return
        assert isinstance(attn_metadata_by_layer, dict)
        attn_metadata = cast(
            "MLACommonMetadata", attn_metadata_by_layer[self.layer_name]
        )

        num_actual_toks = attn_metadata.num_actual_tokens
        slot_mapping_by_layer = forward_context.slot_mapping
        assert isinstance(slot_mapping_by_layer, dict)
        slot_mapping = slot_mapping_by_layer[self.layer_name]

        q = q[:num_actual_toks]
        kv_c_normed = kv_c_normed[:num_actual_toks]
        k_pe = k_pe[:num_actual_toks]
        positions = positions[:num_actual_toks]
        attn_out = attn_out[:num_actual_toks]

        cos_sin_cache = None
        rope_positions = None
        if self.rotary_emb is not None:
            # Pass the fp32 cos/sin table straight to the fused epilogue (it reads
            # fp32 and does the RoPE math in fp32) -- no per-forward dtype cast.
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            rope_positions = positions

        # Decode tokens are laid out first, prefill tokens after. The fused
        # prefill covers every supported config (bf16 / plain-fp8 /
        # fp8_ds_mla), so there is no dense-MHA (forward_mha) fallback.
        num_mqa_tokens = attn_metadata.num_decode_tokens
        num_mha_tokens = q.size(0) - num_mqa_tokens

        # Both the prefill and decode fused epilogues write their own cache
        # slice, so there is no separate do_kv_cache_update.

        # ---- Prefill: fused key-concat + cache-insert + attention ----
        if num_mha_tokens > 0:
            self._forward_prefill_fused(
                q[num_mqa_tokens:],
                kv_c_normed[num_mqa_tokens:],
                k_pe[num_mqa_tokens:],
                rope_positions[num_mqa_tokens:] if rope_positions is not None else None,
                cos_sin_cache,
                slot_mapping[num_mqa_tokens:num_actual_toks],
                attn_metadata,
                attn_out[num_mqa_tokens:],
            )

        # ---- Decode: latent multi-query attention ----
        if num_mqa_tokens > 0:
            mqa_q_nope, mqa_q_pe = q[:num_mqa_tokens].split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            # BMM1: absorb q_nope into latent space. (N,B,P) x (N,P,L) -> (B,N,L)
            ql_nope = torch.bmm(mqa_q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)
            # Fused: concat mqa_q = [ql_nope | q_pe] and insert the decode-token
            # latent into the paged cache (one launch, right before forward_mqa).
            mqa_q = self._decode_concat_cache(
                ql_nope,
                mqa_q_pe,
                kv_c_normed[:num_mqa_tokens],
                k_pe[:num_mqa_tokens],
                rope_positions[:num_mqa_tokens] if rope_positions is not None else None,
                cos_sin_cache,
                slot_mapping[:num_mqa_tokens],
            )
            latent_out, _lse = self.impl.forward_mqa(  # type: ignore[attr-defined]
                mqa_q, self._attn_read_kv_cache(), attn_metadata, self
            )
            self._v_up_proj(latent_out, out=attn_out[:num_mqa_tokens])

    def _decode_concat_cache(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor | None,
        cos_sin_cache: torch.Tensor | None,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Fused decode query-concat + latent cache insert, dispatched by cache
        dtype (same policy as prefill: fp8 cache -> fp8 query)."""
        if self.kv_cache_dtype == "fp8_ds_mla":
            cache = self.kv_cache
            if cache.dtype != torch.uint8:
                cache = cache.view(torch.uint8)
            return fused_mla_decode_q_concat_kv_cache_insert(
                ql_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                cache,
                slot_mapping,
                ds_mla=True,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
            )
        if is_quantized_kv_cache(self.kv_cache_dtype):
            cache = self.kv_cache
            if cache.dtype != torch.float8_e4m3fn:
                cache = cache.view(torch.float8_e4m3fn)
            mqa_q = fused_mla_decode_q_concat_kv_cache_insert(
                ql_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                cache,
                slot_mapping,
                q_scale_inv=self._q_scale_inv,
                cache_scale_inv=self._k_scale_inv,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
            )
            if not getattr(self.impl, "supports_quant_query_input", False):
                # Backend dequantizes fp8 KV on load and takes a bf16 query
                # (TRITON_MLA); undo the query quantization.
                mqa_q = mqa_q.to(ql_nope.dtype) * self._q_scale
            return mqa_q
        return fused_mla_decode_q_concat_kv_cache_insert(
            ql_nope,
            q_pe,
            kv_c_normed,
            k_pe,
            self.kv_cache,
            slot_mapping,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
        )

    def _forward_prefill_fused(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor | None,
        cos_sin_cache: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        attn_metadata,
        out: torch.Tensor,
    ) -> None:
        """Prefill using the fused key-concat + cache-insert kernel.

        Replaces ``_concat_k_nope_k_pe`` and the prefill cache write with one
        fused kernel launch, dispatched by cache dtype. The chunked context
        gather + online-softmax merge are delegated to the impl.

        Supported configs (K3 fp8 policy):
          - bf16 cache        -> bf16 prefill query
          - plain fp8 cache   -> fp8 prefill query (unscaled q/k/v; cache _k_scale)
          - fp8_ds_mla cache  -> bf16 prefill query (656B per-tile self-scaled)
        """
        prefill = attn_metadata.prefill
        has_context = prefill.chunked_context is not None
        fp8_prefill = prefill.q_data_type == current_platform.fp8_dtype()

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        if self.kv_cache_dtype == "fp8_ds_mla":
            # fp8_ds_mla cache (656B, per-tile self-scaled); bf16 attention.
            assert not fp8_prefill, (
                "Kimi-K3 fp8_ds_mla uses a bf16 prefill query; fp8 prefill "
                "query is not supported with fp8_ds_mla."
            )
            kv_cache = self.kv_cache
            if kv_cache.dtype != torch.uint8:
                kv_cache = kv_cache.view(torch.uint8)
            k = fused_mla_key_concat_ds_mla_insert(
                q,
                k_nope,
                k_pe,
                kv_c_normed,
                kv_cache,
                slot_mapping,
                positions,
                cos_sin_cache,
            )
        elif is_quantized_kv_cache(self.kv_cache_dtype):
            assert fp8_prefill, (
                "Kimi-K3 fp8 KV cache requires an fp8 prefill query; enable "
                "--attention-config '{\"use_prefill_query_quantization\": true}'."
            )
            # Plain per-tensor fp8: quant q/k/v (unscaled, matching forward_mha's
            # unscaled `.to(fp8)`) and insert the fp8 latent (scaled by _k_scale).
            kv_cache = self.kv_cache
            if kv_cache.dtype != torch.float8_e4m3fn:
                kv_cache = kv_cache.view(torch.float8_e4m3fn)
            q, k, v = fused_mla_qkv_quant_kv_cache_fp8_insert(
                q,
                k_nope,
                k_pe,
                kv_c_normed,
                v,
                kv_cache,
                slot_mapping,
                self._one_scale,
                self._one_scale,
                self._one_scale,
                self._k_scale_inv,
                positions,
                cos_sin_cache,
            )
        else:
            # Concat full K = [k_nope | k_pe] and insert [kv_c_normed | k_pe]
            # into the paged cache for these prefill tokens, in one launch.
            k = fused_mla_key_concat_kv_cache_insert(
                q,
                k_nope,
                k_pe,
                kv_c_normed,
                self.kv_cache,
                slot_mapping,
                positions,
                cos_sin_cache,
            )

        # When there is no chunked context, backends that honor `out` write the
        # attention result straight into it, avoiding a slice+flatten+copy.
        writes_out = not has_context and prefill.prefill_backend.supports_out()
        output_prefill = prefill.prefill_backend.run_prefill_new_tokens(
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
            out=(
                out.view(-1, self.num_local_heads, self.v_head_dim)
                if writes_out
                else None
            ),
        )

        if has_context:
            context_output, context_lse = self.impl._compute_prefill_context(  # type: ignore[attr-defined]
                q, self._attn_read_kv_cache(), attn_metadata, self._k_scale
            )
            suffix_output, suffix_lse = output_prefill
            out = out.view(-1, self.num_local_heads, self.v_head_dim)
            merge_attn_states(
                output=out,
                prefix_output=context_output[..., : self.v_head_dim],
                prefix_lse=context_lse,
                suffix_output=suffix_output[..., : self.v_head_dim],
                suffix_lse=suffix_lse,
            )
        elif not writes_out:
            out.copy_(output_prefill[..., : self.v_head_dim].flatten(start_dim=-2))
