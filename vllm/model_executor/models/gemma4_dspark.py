# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 DSpark draft model for speculative decoding."""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .gemma4_mtp import Gemma4MTPAttention, Gemma4MTPDecoderLayer
from .qwen3_dflash import DFlashQwen3Model
from .qwen3_dspark import DSparkMarkovHead, Qwen3DSparkForCausalLM
from .utils import extract_layer_index, maybe_prefix


class Gemma4DSparkAttention(Gemma4MTPAttention):
    """Gemma4 attention with its own KV cache and K/V projections."""

    def __init__(
        self,
        config,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        is_full = config.layer_types[extract_layer_index(prefix)] == "full_attention"
        head_dim = (
            getattr(config, "global_head_dim", config.head_dim)
            if is_full
            else config.head_dim
        )
        use_k_eq_v = is_full and getattr(config, "attention_k_eq_v", False)
        num_kv_heads = (
            getattr(config, "num_global_key_value_heads", config.num_key_value_heads)
            if use_k_eq_v
            else config.num_key_value_heads
        )
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=getattr(config, "attn_logit_softcapping", None),
            prefix=prefix,
        )
        self.is_kv_shared_layer = False
        self.use_k_eq_v = use_k_eq_v
        self.kv_size = self.num_kv_heads * self.head_dim
        attn_bias = getattr(config, "attention_bias", False)
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = (
            None
            if use_k_eq_v
            else ColumnParallelLinear(
                config.hidden_size,
                self.total_num_kv_heads * self.head_dim,
                bias=attn_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.v_proj",
            )
        )
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)

    def _kv_proj(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, _ = self.k_proj(hidden_states)
        k_normed = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        v_src = k if self.use_k_eq_v else self.v_proj(hidden_states)[0]
        v_normed = self.v_norm(v_src.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        return k_normed.flatten(-2, -1), v_normed.flatten(-2, -1)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim))).flatten(
            -2, -1
        )
        k, v = self._kv_proj(hidden_states)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4DSparkDecoderLayer(Gemma4MTPDecoderLayer):
    """Gemma4 MTP decoder layer using the KV-owning DSpark attention."""

    def __init__(
        self,
        config,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config, cache_config=cache_config, quant_config=quant_config, prefix=prefix
        )
        get_current_vllm_config().compilation_config.static_forward_context.pop(
            f"{prefix}.self_attn.attn", None
        )
        self.self_attn = Gemma4DSparkAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.self_attn"
        )


@support_torch_compile
class Gemma4DSparkModel(DFlashQwen3Model):
    """Gemma4 DSpark draft backbone (Gemma4 layers + DSpark Markov head)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.use_aux_hidden_state = True
        self.target_layer_ids = tuple(
            getattr(config, "dspark_target_layer_ids", None) or config.target_layer_ids
        )
        current_vllm_config = get_current_vllm_config()
        cache_config = current_vllm_config.cache_config
        quant_config = current_vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.register_buffer(
            "normalizer",
            torch.tensor(config.hidden_size**0.5, dtype=vllm_config.model_config.dtype),
            persistent=False,
        )
        self.fc = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc"),
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            Gemma4DSparkDecoderLayer(
                config,
                cache_config,
                quant_config,
                prefix=maybe_prefix(prefix, f"layers.{i}"),
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def _build_fused_kv_buffers(self) -> None:
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        assert all(a.use_k_eq_v for a in layers_attn), (
            "Gemma4 DSpark fused precompute assumes uniform attention_k_eq_v layers"
        )
        self._build_context_kv_buffers(layers_attn, attn0.k_proj.bias is not None)
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def _build_context_kv_buffers(
        self, layers_attn: list[nn.Module], has_bias: bool
    ) -> None:
        self._hidden_norm_weight = self.hidden_norm.weight.data
        self._fused_k_weight = torch.cat([a.k_proj.weight for a in layers_attn], dim=0)
        self._fused_k_bias: torch.Tensor | None = (
            torch.cat([a.k_proj.bias for a in layers_attn], dim=0) if has_bias else None
        )
        self._k_norm_weights = torch.stack(
            [a.k_norm.weight.data for a in layers_attn], dim=0
        ).contiguous()
        # v_norm has no learnable scale; ones matching the K-norm call shape.
        self._v_norm_weights = torch.ones(
            len(layers_attn),
            layers_attn[0].head_dim,
            dtype=self._k_norm_weights.dtype,
            device=self._k_norm_weights.device,
        )

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Project once via k_proj for all layers. K is raw (the inherited path
        # applies k_norm + RoPE); V = v_norm(same projection), no RoPE (k_eq_v).
        normed = torch.empty_like(context_states)
        ops.rms_norm(
            normed, context_states, self._hidden_norm_weight, self._rms_norm_eps
        )
        all_k_flat = F.linear(normed, self._fused_k_weight, self._fused_k_bias)
        all_k = (
            all_k_flat.view(num_ctx, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        all_v = torch.empty_like(all_k)
        ops.rms_norm(all_v, all_k, self._v_norm_weights, self._rms_norm_eps)
        return all_k, all_v

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_input_ids(input_ids) if input_embeds is None else input_embeds
        )
        for layer in self.layers:
            hidden_states, _ = layer(positions, hidden_states, None)
        return self.norm(hidden_states)


class Gemma4DSparkForCausalLM(Qwen3DSparkForCausalLM):
    """Gemma4 DSpark speculator over a self-contained draft checkpoint."""

    dspark_shares_target_embeddings = False
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
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
        self.draft_id_to_target_id = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)]
        params = dict(self.named_parameters())
        params.update(dict(self.named_buffers()))
        loaded: set[str] = set()
        for name, w in weights:
            if "confidence_head" in name:
                continue
            if "lm_head" not in name:
                name = "model." + name
            for pn, wn, shard in stacked:
                if wn in name and (mapped := name.replace(wn, pn)) in params:
                    params[mapped].weight_loader(params[mapped], w, shard)
                    loaded.add(mapped)
                    break
            else:
                if name in params:
                    p = params[name]
                    getattr(p, "weight_loader", default_weight_loader)(p, w)
                    loaded.add(name)
        self.model._build_fused_kv_buffers()
        return loaded
