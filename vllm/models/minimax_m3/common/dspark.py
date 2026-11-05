# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 DSpark draft model.

The Qwen3 DSpark block drafter (context-KV precompute, non-causal query block,
Markov and confidence heads) built from MiniMax-M3 layers: Gemma RMSNorm
(``x * (1 + w)``) everywhere, partial RoPE, and a dense SwiGLU-OAI MLP. The
checkpoint ships no token embedding or LM head; both are shared with the target.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3DecoderLayer,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import (
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.minimax_m3.common.ops.dspark_context_kv import (
    DSparkContextKVWriter,
    rotary_dims_supported,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

_CHECKPOINT_PREFIX = "language_model.model.dspark."


def _shared_slot_mapping(
    slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
) -> torch.Tensor | None:
    """The context slot mapping when every draft layer shares it, else None
    (dummy runs, and drafts whose layers span several KV cache groups)."""
    if slot_mapping is None or isinstance(slot_mapping, torch.Tensor):
        return slot_mapping
    first = slot_mapping[0]
    if first is None or any(
        s is None or s.data_ptr() != first.data_ptr() for s in slot_mapping
    ):
        return None
    return first


def _gemma_rms_norm(hidden_size: int, eps: float) -> nn.Module:
    """The target's Gemma RMSNorm (fused FlashInfer kernels on CUDA).

    vLLM's GemmaRMSNorm hands the IR op an fp32 ``1 + w`` weight, which the
    vllm_c kernel rejects for bf16 inputs, so every call ran the unfused native
    path (about ten kernels).
    """
    if current_platform.is_rocm():
        from vllm.models.minimax_m3.amd.model import MiniMAXGemmaRMSNorm
    else:
        from vllm.models.minimax_m3.nvidia.model import MiniMAXGemmaRMSNorm
    return MiniMAXGemmaRMSNorm(hidden_size, eps=eps)


def dspark_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    q_fp8_out: torch.Tensor | None,
    layer_name: LayerNameType,
) -> None:
    """The fused QK-norm/RoPE on ``qkv``, also storing the draft tokens' k/v.

    A custom op so the slot mapping is read when it runs: read in the compiled
    forward, it is fixed at trace time, when the draft layers have none.
    """
    _, layer, kv_cache, slot_mapping = get_attention_context(
        _resolve_layer_name(layer_name)
    )
    insert = slot_mapping is not None
    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache if insert else None,
        block_size=kv_cache.size(2) if insert else 0,
        kv_cache_dtype="fp8" if insert and kv_cache.dtype == torch.uint8 else "auto",
        q_fp8_out=q_fp8_out,
        q_fp8_scale=layer._q_scale_float,
    )


def dspark_qknorm_rope_kv_insert_fake(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    q_fp8_out: torch.Tensor | None,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="minimax_m3_dspark_qknorm_rope_kv_insert",
    op_func=dspark_qknorm_rope_kv_insert,
    mutates_args=["qkv", "q_fp8_out"],
    fake_impl=dspark_qknorm_rope_kv_insert_fake,
)


class MiniMaxM3DSparkAttention(DFlashQwen3Attention):
    def __init__(
        self,
        config: MiniMaxM3TextConfig,
        sliding_window: int | None,
        causal: bool,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        query_pre_quantized: bool = False,
    ) -> None:
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=sliding_window,
            causal=causal,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            query_pre_quantized=query_pre_quantized,
        )
        self.q_norm = _gemma_rms_norm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _gemma_rms_norm(self.head_dim, eps=config.rms_norm_eps)
        # With an FP8 KV cache the backend wants an FP8 query, and the fused
        # kernel below can write one. Producing it there instead of letting the
        # attention layer quantize afterwards saves a pass over q.
        self._emit_fp8_query = query_pre_quantized
        # FLASHINFER_TS writes an FP8 query's output as FP16 (FA4 only bf16);
        # an MXFP8 o_proj quantizes FP16 as readily and writes the model dtype.
        self._attn_output_dtype = (
            torch.float16
            if query_pre_quantized
            and self.attn.attn_backend.get_name() == "FLASHINFER_TS"
            and get_input_quant_key(self.o_proj) == kMxfp8Dynamic
            else None
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        # The target's fused per-head Gemma QK-norm + partial NeoX RoPE on q/k,
        # in place: one launch instead of the q-norm, k-norm and RoPE kernels.
        # q_fp8_out additionally stores the normed/roped q as E4M3, which is
        # what an FP8-KV backend consumes.
        q_fp8 = (
            torch.empty(
                (qkv.shape[0], self.q_size),
                dtype=current_platform.fp8_dtype(),
                device=qkv.device,
            )
            if self._emit_fp8_query
            else None
        )
        # Also stores the draft tokens' k/v, so the attention layer gets q alone.
        torch.ops.vllm.minimax_m3_dspark_qknorm_rope_kv_insert(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            q_fp8,
            _encode_layer_name(self.attn.layer_name),
        )
        q = qkv[:, : self.q_size] if q_fp8 is None else q_fp8
        # output_dtype must be stated: it otherwise follows the query, which
        # may be FP8.
        attn_output = self.attn(
            q, None, None, output_dtype=self._attn_output_dtype or hidden_states.dtype
        )
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3DSparkMLP(nn.Module):
    """Dense SwiGLU-OAI MLP: gate * sigmoid(alpha * gate) * (up + beta)."""

    def __init__(
        self,
        config: MiniMaxM3TextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if config.hidden_act != "swigluoai":
            raise ValueError(
                f"Unsupported MiniMax DSpark activation {config.hidden_act!r}; "
                "only swigluoai is supported."
            )
        intermediate_size = config.dspark_ffn_hidden_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMulWithClamp(
            swiglu_limit=config.swiglu_limit,
            alpha=config.swiglu_alpha,
            beta=config.swiglu_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class MiniMaxM3DSparkDecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: MiniMaxM3TextConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        if not getattr(config, "dspark_use_dense_ffn", False):
            raise NotImplementedError("MiniMax DSpark supports dense FFN drafts only.")
        self.hidden_size = config.hidden_size
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        # An FP8 draft cache means the backend reads an FP8 query, which the
        # fused QK-norm/RoPE kernel emits directly.
        kv_cache_dtype = getattr(cache_config, "cache_dtype", "auto")
        self.self_attn = MiniMaxM3DSparkAttention(
            config,
            sliding_window=sliding_window,
            causal=causal,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            query_pre_quantized=str(kv_cache_dtype).startswith("fp8"),
        )
        self.mlp = MiniMaxM3DSparkMLP(config, quant_config, prefix=f"{prefix}.mlp")
        hidden_size, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = _gemma_rms_norm(hidden_size, eps=eps)
        self.post_attention_layernorm = _gemma_rms_norm(hidden_size, eps=eps)


class MiniMaxM3DSparkModel(Qwen3DSparkModel):
    decoder_layer_cls = MiniMaxM3DSparkDecoderLayer
    quant_config_mapper = WeightsMapper(
        orig_to_new_prefix={_CHECKPOINT_PREFIX: "model."}
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        hidden_size, eps = self.config.hidden_size, self.config.rms_norm_eps
        self.hidden_norm = _gemma_rms_norm(hidden_size, eps=eps)
        self.norm = _gemma_rms_norm(hidden_size, eps=eps)

    def _normalize_context_states(self, context_states: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(context_states)

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [layer.self_attn.k_norm(k) for layer, k in zip(self.layers, all_k)]
        )

    def _context_kv_writer(self) -> DSparkContextKVWriter | None:
        """The fused context K/V writer, or None where the base path runs."""
        if not hasattr(self, "_ctx_kv_writer"):
            if any(a.kv_cache.numel() == 0 for a in self._attn_layers):
                return None  # KV caches not bound yet: decide once they are
            rotary_dim = self._rope_cos_sin_cache.shape[-1]
            supported = (
                self._rope_is_neox
                and self.layers[0].self_attn.v_scale is None
                and rotary_dims_supported(rotary_dim, self._head_dim)
                and self._rope_cos_sin_cache.dtype == torch.bfloat16
                and DSparkContextKVWriter.supported(self._attn_layers)
            )
            self._ctx_kv_writer = (
                DSparkContextKVWriter(self._attn_layers) if supported else None
            )
        return self._ctx_kv_writer

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """One kernel norms, ropes and stores the context K/V of every layer,
        straight from the projection output."""
        slot_mapping = _shared_slot_mapping(context_slot_mapping)
        writer = (
            self._context_kv_writer()
            if slot_mapping is not None
            and hasattr(self, "_num_attn_layers")
            and context_states.dtype == torch.bfloat16
            else None
        )
        if writer is None:
            super().precompute_and_store_context_kv(
                context_states, context_positions, context_slot_mapping
            )
            return
        all_kv = self._project_context_kv_token_major(
            context_states,
            context_states.shape[0],
            self._num_attn_layers,
            self._num_kv_heads,
            self._head_dim,
        )
        writer(
            all_kv,
            self._k_norm_weights,
            self._rms_norm_eps,
            context_positions,
            self._rope_cos_sin_cache,
            self._rope_cos_sin_cache.shape[-1],
            slot_mapping,
        )


class MiniMaxM3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    model_cls = MiniMaxM3DSparkModel

    @staticmethod
    def _rename_checkpoint_weights(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            name = name.removeprefix(_CHECKPOINT_PREFIX)
            name = name.replace(".decoder_layer.", ".")
            if name.startswith("final_norm."):
                name = "norm." + name.removeprefix("final_norm.")
            # MiniMax stores the MXFP8 E8M0 block scales as weight_scale_inv.
            name = name.replace(".weight_scale_inv", ".weight_scale")
            yield name, tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return super().load_weights(self._rename_checkpoint_weights(weights))
