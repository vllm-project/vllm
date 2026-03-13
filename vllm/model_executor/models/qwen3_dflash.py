# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from flashinfer.rope import apply_rope_with_cos_sin_cache
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.platforms import current_platform
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionType

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# vllm::dflash_attn  –  slim custom op for DFlash cross-attention.
#
# QKV projection, per-head norms, and RoPE for query tokens all live in the
# compiled graph (DFlashQwen3Attention.forward).  This op only:
#   1. Concats precomputed context K/V with query K/V
#   2. Runs the self-attention backend
# ---------------------------------------------------------------------------


def dflash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Run DFlash cross-attention with precomputed context K/V.

    q, k, v are the fully-processed (projected, normed, RoPE'd) QKV of the
    query tokens (padded, fixed shape – produced in the compiled graph).
    *output* is a pre-allocated buffer whose address stays stable across
    CUDA graph replays.

    Precomputed context K/V is read from the ForwardContext side-channel.
    """
    fc = get_forward_context()
    context_k, context_v = fc.dflash_context_kv[layer_name]

    # Concat precomputed context K/V with query K/V
    k = torch.cat([context_k, k], dim=0)
    v = torch.cat([context_v, v], dim=0)

    # Run the self-attention (KV-cache update + backend) eagerly.
    layer = fc.no_compile_layers[layer_name]  # DFlashQwen3Attention
    attn_layer = fc.no_compile_layers[layer.attn.layer_name]
    attn_output = attn_layer(q, k, v)

    # Write into the pre-allocated output buffer so its address stays stable
    # for piecewise CUDA graph replay.
    output.copy_(attn_output)


def dflash_attn_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="dflash_attn",
    op_func=dflash_attn,
    mutates_args=["output"],
    fake_impl=dflash_attn_fake,
    dispatch_key=current_platform.dispatch_key,
)


class DFlashQwen3Attention(nn.Module):
    """Cross-attention for DFlash: K/V from concat(context, query),
    Q from query only. Adapted from Qwen3Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dtype=torch.float32,  # required for float32
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # QKV projection + per-head norms + RoPE run in the compiled graph
        # (fixed-shape query tokens only).
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm (reshape so last dim = head_dim, use the
        # vllm RMSNorm modules which are torch.compile-friendly).
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        # Rotary embeddings (CustomOp, torch.compile-friendly)
        q, k = self.rotary_emb(positions, q, k)

        # Pre-allocate output buffer in the FX graph so that split_graph
        # places it in the compiled piece before the splitting op.  The
        # CUDA graph pool then keeps its address stable across replays.
        attn_output = torch.empty(
            hidden_states.shape[0],
            self.q_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.dflash_attn(
            q,
            k,
            v,
            attn_output,
            layer_name=self.layer_name,
        )
        output, _ = self.o_proj(attn_output)
        return output


class DFlashQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER

        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# NOTE: torch.compile is not stable here
@support_torch_compile
class DFlashQwen3Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        if drafter_config is not None and "use_aux_hidden_state" in drafter_config:
            self.use_aux_hidden_state = drafter_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                DFlashQwen3DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                    config=self.config,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            num_features_to_use = self.config.num_hidden_layers
            if "target_layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["target_layer_ids"])
            elif "layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["layer_ids"])
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * num_features_to_use
            else:
                fc_input_size = self.config.hidden_size * num_features_to_use
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for batched precompute_context_kv.

        Must be called after weights are loaded.  Stacks the KV-projection
        weights, K-norm weights, and RoPE parameters from every attention
        layer so that precompute_context_kv can run one GEMM, one norm,
        and one RoPE call for *all* layers at once.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights: list of [head_dim] tensors, one per layer.
        # Used with flashinfer rmsnorm (one fused kernel per layer).
        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]

        # RoPE params (shared across all layers)
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._attn_layer_names = [a.layer_name for a in layers_attn]
        self._rms_norm_eps = attn0.q_norm.variance_epsilon

    @torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
    def precompute_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Precompute projected, normed, and RoPE'd K/V for context states.

        All layers are processed in one fused GEMM, one fused RMSNorm, and
        one fused RoPE call.  Requires _build_fused_kv_buffers() to have
        been called after weight loading.

        Returns a dict mapping layer_name -> (k, v).
        """
        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        # --- Fused KV projection (one GEMM for all layers) ---
        # [num_ctx, L * 2 * kv]
        all_kv_flat = F.linear(
            context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # [L, num_ctx, 2 * kv]
        all_kv = all_kv_flat.view(num_ctx, L, 2 * kv).transpose(0, 1).contiguous()
        # Split K / V – each [L, num_ctx, kv]  (non-contiguous views)
        all_k, all_v = all_kv.split(kv, dim=-1)

        # --- Per-head RMSNorm on K (one flashinfer rmsnorm kernel per layer) ---
        # [L, num_ctx, nkv, hd] -> per-layer [num_ctx * nkv, hd] rmsnorm
        all_k = all_k.reshape(L, num_ctx * nkv, hd).contiguous()
        all_k = torch.stack(
            [
                RMSNorm.forward_static(
                    x=all_k[i],
                    weight=self._k_norm_weights[i],
                    variance_epsilon=self._rms_norm_eps,
                    hidden_size=hd,
                    orig_dtype=all_k.dtype,
                )
                for i in range(L)
            ]
        )
        # [L, num_ctx, kv]
        all_k = all_k.reshape(L, num_ctx, kv).contiguous()

        # --- Fused RoPE on K (flatten layers into batch, tile positions) ---
        # [L * num_ctx, kv]
        k_flat = all_k.reshape(L * num_ctx, kv)
        positions_tiled = context_positions.repeat(L)
        # Use torch.compile-able vllm rope
        _, k_rope = apply_rope_with_cos_sin_cache(
            positions=positions_tiled,
            query=k_flat,
            key=k_flat,
            head_size=self._rope_head_size,
            cos_sin_cache=self._rope_cos_sin_cache,
            is_neox=self._rope_is_neox,
        )
        # [L, num_ctx, kv]
        all_k = k_rope.view(L, num_ctx, kv)
        all_v = all_v.contiguous()

        return {
            name: (all_k[i], all_v[i]) for i, name in enumerate(self._attn_layer_names)
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            prefix="model",
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def precompute_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Precompute projected + RoPE'd K/V for context states, all layers."""
        return self.model.precompute_context_kv(context_states, context_positions)

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        result = self.model.hidden_norm(self.model.fc(hidden_states))
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash should use mask_token_id to embed the padding hidden state"
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
