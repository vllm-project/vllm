# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm DFlash2 draft model for Beam targets."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Attention
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3DecoderLayer
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)

from .dflash2_ops import candidate_edges

logger = init_logger(__name__)


class DFlash2BeamAttention(DFlashQwen3Attention):
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        from .attention_ops import beam_qk_norm_rope

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.v_scale is not None:
            v = v * self.v_scale
        rope = self.rotary_emb
        q, k = beam_qk_norm_rope(
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.variance_epsilon,
            positions,
            rope.cos_sin_cache,
            self.head_dim,
            rope.rotary_dim,
            rope.is_neox_style,
        )
        # Let the selected ROCm backend update its own cache. Cache layouts and
        # quantization are backend details and must not be reproduced here.
        output = self.attn(q, k, v)
        output, _ = self.o_proj(output)
        return output


class DFlash2BeamDecoderLayer(DFlash2Qwen3DecoderLayer):
    attention_cls = DFlash2BeamAttention


class DFlash2BeamCandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert hidden_states.dtype == torch.bfloat16
        return candidate_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            self.hidden_projection(hidden_states),
            anchor_token_ids,
            self.top_k,
        )


class DFlash2BeamModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        draft_config = config.dflash_config
        if getattr(config, "attention_bias", False) or get_draft_quant_config(
            vllm_config
        ):
            raise NotImplementedError(
                "The Beam DFlash2 drafter supports only unquantized drafts "
                "without attention biases."
            )
        params_dtype = vllm_config.model_config.dtype
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        current_vllm_config = get_current_vllm_config()
        self.layers = nn.ModuleList(
            [
                DFlash2BeamDecoderLayer(
                    current_vllm_config,
                    config=config,
                    layer_idx=layer_idx,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=None,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        target_hidden_size = (
            getattr(config, "target_hidden_size", None) or config.hidden_size
        )
        self.fc = ReplicatedLinear(
            target_hidden_size * len(draft_config["target_layer_ids"]),
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.candidate_selector = DFlash2BeamCandidateSelector(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            rank=int(draft_config["selector_rank"]),
            top_k=int(draft_config["selector_top_k"]),
            params_dtype=params_dtype,
            prefix=maybe_prefix(prefix, "candidate_selector"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        if self.input_embedding_scale != 1.0:
            embeds = embeds * self.input_embedding_scale
        return embeds

    def build_context_kv_weights(self) -> None:
        """Stack every layer's K/V projection and K-norm weight."""
        attns = [layer.self_attn for layer in self.layers]
        self._context_kv_weight = torch.cat(
            [attn.qkv_proj.weight[attn.q_size :] for attn in attns]
        )
        self._context_k_norm_weight = torch.stack(
            [attn.k_norm.weight for attn in attns]
        ).contiguous()

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Project and transform context K/V, then ask each backend to cache it."""
        if not hasattr(self, "_context_kv_weight"):
            logger.warning_once(
                "The Beam DFlash2 context K/V weights were built lazily; this is "
                "expected only with dummy weights."
            )
            self.build_context_kv_weights()

        attn0 = self.layers[0].self_attn
        num_layers = len(self.layers)
        num_ctx = context_states.shape[0]
        num_kv_heads, head_dim = attn0.num_kv_heads, attn0.head_dim
        normed = torch.empty_like(context_states)
        ops.rms_norm(
            normed,
            context_states,
            self.hidden_norm.weight,
            self.hidden_norm.variance_epsilon,
        )
        kv = F.linear(normed, self._context_kv_weight).view(
            num_ctx, num_layers, 2, num_kv_heads, head_dim
        )
        self._store_context_kv(kv, context_positions, context_slot_mapping)

    def _store_context_kv(
        self,
        kv: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        from .dflash2_context_ops import transform_context_kv

        first = self.layers[0].self_attn
        rotary = first.rotary_emb
        assert kv.dtype == torch.bfloat16
        assert rotary.rotary_dim == first.head_dim
        assert rotary.cos_sin_cache.dtype == kv.dtype
        keys, values = transform_context_kv(
            kv,
            self._context_k_norm_weight,
            positions,
            rotary.cos_sin_cache,
            first.k_norm.variance_epsilon,
            first.v_scale if first.v_scale is not None else 1.0,
            rotary.is_neox_style,
        )
        if slot_mapping is None:
            return

        per_layer = isinstance(slot_mapping, (list, tuple))
        if per_layer and len(slot_mapping) != len(self.layers):
            raise ValueError(
                "DFlash2 context slot mappings must have one entry per draft layer."
            )
        for index, layer in enumerate(self.layers):
            slots = slot_mapping[index] if per_layer else slot_mapping
            if slots is None:
                continue
            attn = layer.self_attn.attn
            if not hasattr(attn.impl, "do_kv_cache_update"):
                raise NotImplementedError(
                    f"{attn.impl.__class__.__name__} cannot update the KV cache."
                )
            attn.impl.do_kv_cache_update(
                attn,
                keys[index],
                values[index],
                attn.kv_cache,
                slots,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        residual = inputs_embeds
        hidden_states = self.layers[0].input_layernorm(inputs_embeds)
        for i, layer in enumerate(self.layers):
            next_norm = (
                self.layers[i + 1].input_layernorm
                if i + 1 < len(self.layers)
                else self.norm
            )
            hidden_states, residual = _forward_normalized(
                layer, positions, hidden_states, residual, next_norm
            )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def _forward_normalized(
    layer: DFlash2Qwen3DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    next_norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .dflash2_conv_ops import conv_residual_norm

    hidden_states, coefficients = layer.attention_conv.prepare(hidden_states)
    hidden_states = layer.self_attn(positions, hidden_states)
    hidden_states, residual = conv_residual_norm(
        hidden_states,
        coefficients,
        layer.attention_conv.base_kernel[1],
        residual,
        layer.post_attention_layernorm.weight,
        layer.post_attention_layernorm.variance_epsilon,
        layer.attention_conv.block_size,
        layer.attention_conv.group_size,
    )
    hidden_states, coefficients = layer.mlp_conv.prepare(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return conv_residual_norm(
        hidden_states,
        coefficients,
        layer.mlp_conv.base_kernel[1],
        residual,
        next_norm.weight,
        next_norm.variance_epsilon,
        layer.mlp_conv.block_size,
        layer.mlp_conv.group_size,
    )


class DFlash2BeamForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.model = DFlash2BeamModel(
            vllm_config=vllm_config,
            start_layer_id=vllm_config.model_config.get_total_num_hidden_layers(),
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=float(draft_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected = self.model.fc.input_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"The DFlash2 drafter expects {expected} concatenated target "
                f"features but received {hidden_states.shape[-1]}; its "
                "target_layer_ids do not match what the target captures."
            )
        return self.model.fc(hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [layer.self_attn.causal for layer in self.model.layers]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = {f"model.{name}" for name in self.model.load_weights(weights)}
        self.model.build_context_kv_weights()
        return loaded | {"model.embed_tokens.weight", "lm_head.weight"}
