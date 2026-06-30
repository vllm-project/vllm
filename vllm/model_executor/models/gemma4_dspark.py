# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 DSpark draft model (semi-autoregressive speculative decoding).

Same flow as Qwen3 DSpark (see ``qwen3_dspark.py`` and the generic
``DSparkSpeculator``): the target's auxiliary hidden states at
``target_layer_ids`` are combined via ``fc`` into a per-layer attention context;
that context's K/V is pre-inserted into each draft layer's cache
(``precompute_and_store_context_kv``); a non-causal block of
``num_speculative_tokens`` query tokens runs in one parallel pass; and the
sequential Markov head injects intra-block dependency.

Gemma4-specific pieces (vs the Qwen3 backbone in ``qwen3_dflash.py``):
  * Layer internals: sandwich norms + per-layer ``layer_scalar``, gelu-tanh
    gated MLP, per-head Q/K/V RMSNorm, ``scaling=1.0``, partial rotary,
    ``global_head_dim``.
  * ``attention_k_eq_v``: no ``v_proj`` — V = ``v_norm(k_proj(x))`` (no RoPE),
    K = ``RoPE(k_norm(k_proj(x)))`` (the checkpoint ships only q/k per layer).
  * ``final_logit_softcapping`` on the base logits.

Self-contained checkpoint (own ``embed_tokens``/``lm_head``, ``tie=False``), so
``dspark_shares_target_embeddings = False``. The Markov/confidence heads are
shared with the Qwen3 DSpark model.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
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
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .qwen3_dspark import DSparkConfidenceHead, DSparkMarkovHead
from .utils import extract_layer_index, maybe_prefix

logger = init_logger(__name__)


def _layer_head_dim(config, layer_idx: int) -> tuple[int, int, bool]:
    """Return (head_dim, num_kv_heads, use_k_eq_v) for a draft layer."""
    is_full = config.layer_types[layer_idx] == "full_attention"
    head_dim = (
        getattr(config, "global_head_dim", config.head_dim)
        if is_full
        else config.head_dim
    )
    use_k_eq_v = is_full and getattr(config, "attention_k_eq_v", False)
    if use_k_eq_v:
        num_kv_heads = getattr(
            config, "num_global_key_value_heads", config.num_key_value_heads
        )
    else:
        num_kv_heads = config.num_key_value_heads
    return head_dim, num_kv_heads, use_k_eq_v


class Gemma4DSparkAttention(nn.Module):
    """Gemma4 full-attention block for DSpark drafting.

    Context K/V are pre-inserted into the cache (see
    ``Gemma4DSparkModel.precompute_and_store_context_kv``); this forward handles
    the query tokens only and reuses the same projections/norms/RoPE so the two
    paths stay numerically consistent.
    """

    def __init__(
        self,
        config,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        head_dim, num_kv_heads, use_k_eq_v = _layer_head_dim(config, layer_idx)
        self.use_k_eq_v = use_k_eq_v

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = 1.0  # Gemma4: Q/K norms handle scaling implicitly.

        attn_bias = getattr(config, "attention_bias", False)
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        # k_eq_v: V is derived from k_proj's output (no separate v_proj weight).
        self.v_proj = None
        if not use_k_eq_v:
            self.v_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_kv_heads * self.head_dim,
                bias=attn_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.v_proj",
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # V norm has no learnable scale (pure normalization), matching Gemma4.
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)

        if layer_type in config.rope_parameters:
            rope_parameters = dict(config.rope_parameters[layer_type])
        else:
            rope_parameters = dict(config.rope_parameters)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=getattr(config, "attn_logit_softcapping", None),
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def _k_v_from_proj(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (k_normed_flat, v_normed_flat) before RoPE.

        K is the k_norm'd k_proj output; V is the v_norm'd source (k_proj output
        for k_eq_v, else v_proj output). RoPE is applied to K by the caller.
        """
        k, _ = self.k_proj(hidden_states)
        k_normed = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        if self.use_k_eq_v:
            v_src = k
        else:
            v_src, _ = self.v_proj(hidden_states)
        v_normed = self.v_norm(v_src.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        return k_normed.flatten(-2, -1), v_normed.flatten(-2, -1)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim)))
        q = q.flatten(-2, -1)
        k, v = self._k_v_from_proj(hidden_states)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    @torch.inference_mode()
    def store_context_kv(
        self,
        normed_context: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        """Project normed context hidden -> K/V, RoPE K, write to this cache.

        Mirrors ``forward`` (same projections/norms/RoPE) but for context tokens
        that carry no query. ``slot_mapping`` None runs the projection only
        (profiling / dummy run).
        """
        k, v = self._k_v_from_proj(normed_context)
        # RoPE applies to K only; pass a dummy query so we reuse the same module
        # (handles partial rotary identically to the query path).
        dummy_q = k.new_empty((k.shape[0], self.q_size))
        _, k = self.rotary_emb(positions, dummy_q, k)
        if slot_mapping is None:
            return
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        inner = self.attn
        inner.impl.do_kv_cache_update(inner, k, v, inner.kv_cache, slot_mapping)


class Gemma4DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.self_attn = Gemma4DSparkAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = Gemma4DSparkMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Per-layer learnable scalar (loaded from the checkpoint).
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states * self.layer_scalar


class Gemma4DSparkMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = get_act_and_mul_fn(hidden_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma4DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        # DSpark target layers may be named target_layer_ids (Gemma4 ckpt) or
        # dspark_target_layer_ids (DSV4 ckpt).
        self.target_layer_ids = tuple(
            getattr(config, "dspark_target_layer_ids", None)
            or getattr(config, "target_layer_ids")
        )

        current_vllm_config = get_current_vllm_config()
        cache_config = current_vllm_config.cache_config
        quant_config = current_vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # Gemma scales embeddings by sqrt(hidden_size).
        self.register_buffer(
            "normalizer",
            torch.tensor(config.hidden_size**0.5, dtype=vllm_config.model_config.dtype),
            persistent=False,
        )

        # Combine the per-target-layer aux hidden states into the draft context.
        self.fc = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc"),
        )
        # Normalizes the combined context before the per-layer KV projection.
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers = nn.ModuleList(
            [
                Gemma4DSparkDecoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{i}"),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # DSpark heads (shared definitions with the Qwen3 DSpark model).
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", False):
            input_dim = config.hidden_size
            if getattr(config, "confidence_head_with_markov", False):
                input_dim += config.markov_rank
            self.confidence_head = DSparkConfidenceHead(
                input_dim,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """context = fc(concat of target aux hidden states) ([T, H*len]->[T, H])."""
        return self.fc(aux_hidden_states)

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        normed = self.hidden_norm(context_states)
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            layer.self_attn.store_context_kv(normed, context_positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Gemma4DSparkForCausalLM(nn.Module):
    # Self-contained checkpoint: own embed_tokens / lm_head (not aliased from
    # the target). See load_dspark_model.
    dspark_shares_target_embeddings = False
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.model = Gemma4DSparkModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            soft_cap=getattr(self.config, "final_logit_softcapping", None),
        )

    # --- Hooks used by the DSpark speculator ------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mappings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        return self.model.confidence_head(head_hidden, markov_embed)

    # --- Weight loading ----------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the self-contained Gemma4 DSpark checkpoint.

        Checkpoint names are flat (``embed_tokens.*``, ``layers.{i}.*``, ``fc``,
        ``hidden_norm``, ``norm``, ``markov_head.*``, ``confidence_head.*``,
        ``lm_head``). Everything but ``lm_head`` lives under ``self.model``.
        """
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "lm_head" not in name:
                name = "model." + name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                if name not in params_dict:
                    logger.warning("Gemma4 DSpark: unexpected weight %s", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        logger.info_once(
            "Gemma4 DSpark draft model loaded: %d params", len(loaded_params)
        )
        return loaded_params
