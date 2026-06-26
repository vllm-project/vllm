# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash speculator for Laguna target models.

Laguna DFlash uses a uniform drafter layer flavor (`layer_types` all full
or all sliding). The draft checkpoint shares token embedding and lm_head
weights with the target model through the generic spec-decode proposer.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.multimodal.inputs import NestedTensors
from vllm.v1.attention.backend import AttentionType

from .laguna import LagunaMLP
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


def _get_dflash_layer_types(config) -> tuple[str, ...]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        raise ValueError("Laguna DFlash config requires `layer_types`.")
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"DFlash layer_types length {len(layer_types)} does not match "
            f"num_hidden_layers {config.num_hidden_layers}."
        )
    # Laguna DFlash checkpoints use a uniform drafter attention flavor.
    if len(set(layer_types)) > 1:
        raise NotImplementedError(
            "Laguna DFlash drafter requires a uniform `layer_types` "
            f"(got {sorted(set(layer_types))})."
        )
    return tuple(layer_types)


def _resolve_rope_params(config, layer_type: str) -> dict:
    """Return a flat rope-parameters dict for one Laguna layer flavor.

    Laguna configs may store RoPE either as a flat dict or nested by layer
    type. SWA layers may also override it via `swa_rope_parameters`.
    """
    is_sliding = layer_type == "sliding_attention"

    top_rope = getattr(config, "rope_parameters", None) or {}
    if any(isinstance(v, dict) for v in top_rope.values()):
        base_rope = top_rope.get(layer_type) or top_rope.get("full_attention") or {}
    else:
        base_rope = top_rope

    swa_rope = getattr(config, "swa_rope_parameters", None)
    rope_params = dict(swa_rope if (is_sliding and swa_rope is not None) else base_rope)

    top_partial = getattr(config, "partial_rotary_factor", None)
    if top_partial is not None and "partial_rotary_factor" not in rope_params:
        rope_params["partial_rotary_factor"] = top_partial
    top_theta = getattr(config, "rope_theta", None)
    if top_theta is not None and "rope_theta" not in rope_params:
        rope_params["rope_theta"] = top_theta
    return rope_params


class DFlashLagunaAttention(nn.Module):
    """Laguna attention variant used by the DFlash drafter.

    This mirrors `LagunaAttention` for query projection, RoPE, QK norm,
    attention, and softplus gating.  The DFlash-specific part is
    KV ownership: verifier-context K/V is projected and inserted into this
    layer's KV cache before forward, so forward only receives draft query
    tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 131072,
        head_dim: int | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        rope_parameters: dict | None = None,
        rms_norm_eps: float = 1e-06,
        sliding_window: int | None = None,
        gating: bool | str = True,
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
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        assert rope_parameters is not None
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position_embeddings,
            is_neox_style=True,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )
        if sliding_window is not None:
            # Keep full KV allocation: context K/V is inserted manually at
            # absolute slots, while SWA is only a compute-time attention limit.
            self.attn.sliding_window = None
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        if gating is True:
            gating = "per-head"
        elif gating is False:
            gating = "disabled"
        if gating not in ("per-head", "per-element", "disabled"):
            raise NotImplementedError(
                "Laguna DFlash drafter only supports per-head, per-element, "
                f"or disabled gating, got {gating!r}."
            )
        self.gating = gating
        if self.gating != "disabled":
            self.gate_per_head = self.gating == "per-head"
            g_out = (
                self.total_num_heads
                if self.gate_per_head
                else self.total_num_heads * self.head_dim
            )
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                g_out,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        else:
            self.g_proj = None
            self.gate_per_head = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)

        if self.gating and self.g_proj is not None:
            gate, _ = self.g_proj(hidden_states)
            gate = F.softplus(gate.float()).type_as(attn_output)
            if self.gate_per_head:
                attn_shape = attn_output.shape
                attn_output = (
                    attn_output.view(*attn_shape[:-1], self.num_heads, self.head_dim)
                    * gate.unsqueeze(-1)
                ).view(attn_shape)
            else:
                attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output


class DFlashLagunaDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        layer_type: str = "full_attention",
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = layer_type
        sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )
        rope_params = _resolve_rope_params(config, layer_type)

        self.self_attn = DFlashLagunaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            rope_parameters=rope_params,
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=sliding_window,
            gating=getattr(config, "gating", True),
        )
        self.mlp = LagunaMLP(
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


@support_torch_compile
class DFlashLagunaModel(nn.Module, EagleModelMixin):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        weight_layer_offset: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        target_layer_ids = self.config.dflash_config["target_layer_ids"]
        if not target_layer_ids:
            raise ValueError(
                "Laguna DFlash config requires non-empty "
                "`dflash_config.target_layer_ids`."
            )

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layer_types = _get_dflash_layer_types(self.config)
        self.layers = nn.ModuleList(
            [
                DFlashLagunaDecoderLayer(
                    prefix=maybe_prefix(
                        prefix, f"layers.{layer_idx + weight_layer_offset}"
                    ),
                    config=self.config,
                    cache_config=vllm_config.cache_config,
                    quant_config=self.quant_config,
                    layer_type=self.layer_types[layer_idx],
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        num_features_to_use = len(target_layer_ids)
        target_hidden_size = vllm_config.model_config.get_hidden_size()
        fc_input_size = target_hidden_size * num_features_to_use
        self.num_aux_slices = num_features_to_use
        self.aux_hidden_norms = nn.ModuleList(
            [
                RMSNorm(
                    fc_input_size // num_features_to_use,
                    eps=self.config.rms_norm_eps,
                )
                for _ in range(num_features_to_use)
            ]
        )
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
        """Cache per-layer tensors used to precompute context K/V.

        DFlash receives verifier hidden states as context. We project those
        states into the drafter's K/V space and write them directly into each
        layer's Attention KV cache before query-token decoding starts.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._input_layernorm_weights = [
            layer.input_layernorm.weight.data for layer in self.layers
        ]

        self._kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        if has_bias:
            self._kv_biases: list[torch.Tensor | None] = [
                a.qkv_proj.bias[a.q_size :] for a in layers_attn
            ]
        else:
            self._kv_biases = [None for _ in layers_attn]

        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]

        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must have the same RoPE parameters for DFlash precomputation"

        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All layers must have the same attn config for DFlash precomputation"

        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Project verifier context states and insert K/V into Attention cache.

        The normal forward path only handles synthetic query tokens. Context
        K/V is precomputed here because it comes from verifier hidden states,
        not from the drafter token stream.
        """
        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "DFlash buffer initialization was skipped. If dummy weights are not "
                "in use, this may indicate an error in weight loading."
            )
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        all_k = torch.empty(
            (L, num_ctx, nkv, hd),
            dtype=context_states.dtype,
            device=context_states.device,
        )
        all_v = torch.empty_like(all_k)
        for i in range(L):
            normed_context_states = self.layers[i].input_layernorm(context_states)
            kv_i = F.linear(
                normed_context_states,
                self._kv_weights[i],
                self._kv_biases[i],
            ).view(num_ctx, 2, nkv, hd)
            all_k[i] = kv_i[:, 0]
            all_v[i] = kv_i[:, 1]

        all_k_normed = torch.empty_like(all_k)
        for i in range(L):
            all_k_normed[i] = self.layers[i].self_attn.k_norm(all_k[i])

        # Apply RoPE to every layer's context K before writing it into the
        # drafter cache. The temporary buffer is sized by drafter layers times
        # active context tokens, not by the full model depth.
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        if context_slot_mapping is None:
            return

        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        for i in range(L):
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                context_slot_mapping,
            )

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
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
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
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DFlashLagunaForCausalLM(nn.Module, SupportsEagle3):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            raise ValueError("Laguna DFlash config requires `draft_vocab_size`.")
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            raise ValueError(
                "Laguna DFlash shares the target lm_head and requires "
                "`draft_vocab_size` to match the target vocabulary size "
                f"({self.config.draft_vocab_size} != {target_vocab_size})."
            )
        self.model = DFlashLagunaModel(
            vllm_config=vllm_config,
            prefix="model",
            # DFlash drafter layers are saved after the target verifier layers.
            weight_layer_offset=target_layer_num,
        )

        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.draft_vocab_size)

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
        return self.logits_processor(self.lm_head, hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize each verifier hidden-state slice, concatenate them, then
        # project into the drafter hidden size used as DFlash context.
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        num_slices = self.model.num_aux_slices
        slice_size = hidden_states.shape[-1] // num_slices
        slices = hidden_states.view(hidden_states.shape[0], num_slices, slice_size)
        normed = torch.empty_like(slices)
        for i, norm in enumerate(self.model.aux_hidden_norms):
            normed[:, i, :] = norm(slices[:, i, :])
        hidden_states = normed.reshape(hidden_states.shape[0], -1)
        result = self.model.fc(hidden_states)
        result = self.model.hidden_norm(result)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        for name, loaded_weight in weights:
            if "lm_head" not in name:
                name = "model." + name
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        loader = AutoWeightsLoader(self)
        loaded_weight_names = loader.load_weights(model_weights.items())
        loaded_weight_names.add("lm_head.weight")
        loaded_weight_names.add("model.embed_tokens.weight")
        self.model._build_fused_kv_buffers()
        return loaded_weight_names
