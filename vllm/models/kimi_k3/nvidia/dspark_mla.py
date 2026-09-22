# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 dense MLA draft model for DSpark speculative decoding."""

from collections.abc import Iterable

import torch
import torch.nn as nn

import vllm._custom_ops as ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.models.kimi_k3.nvidia.model import KimiMLP
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.worker.workspace import current_workspace_manager


def _duplicate_context_kv_weights(
    weights: Iterable[tuple[str, torch.Tensor]], num_layers: int
) -> Iterable[tuple[str, torch.Tensor]]:
    """Load each layer's KV projection into the cross-layer linear."""
    for name, weight in weights:
        yield name, weight
        layer_prefix, marker, param_name = name.partition(
            ".self_attn.kv_a_proj_with_mqa."
        )
        if not marker:
            continue
        layer_idx_str = layer_prefix.rsplit(".", 1)[-1]
        if not layer_idx_str.isdecimal():
            continue
        layer_idx = int(layer_idx_str)
        if layer_idx >= num_layers:
            continue
        fused_weight = weight.detach()
        fused_weight.shard_id = layer_idx
        yield f"context_kv_proj.{param_name}", fused_weight


class K3DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        quant_config = get_draft_quant_config(vllm_config)
        self.self_attn = MultiHeadLatentAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(
                prefix, f"layers.{start_layer_id + layer_idx}.self_attn"
            ),
            use_rope=True,
            non_causal_multi_token_decode=True,
        )
        # Both row-parallel outputs stay un-reduced; their all-reduces are fused
        # into the RMSNorm that follows via fused_allreduce_rms_norm.
        self.self_attn.o_proj.reduce_results = False
        self.mlp = KimiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}.mlp"),
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
        if residual is None:
            # First layer: hidden_states is the (already reduced) embedding.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_allreduce_rms_norm(
                hidden_states, residual, self.input_layernorm
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = fused_allreduce_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        # The MLP output is reduced by the next layer's input_layernorm (or by
        # the model's final_norm).
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class K3DSparkModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = get_draft_quant_config(vllm_config)

        # The frozen target embedding is aliased after the draft checkpoint loads.
        self.embed_tokens: nn.Module | None = None

        self.context_proj = ReplicatedLinear(
            self.config.target_hidden_size * self.config.num_target_layers,
            self.config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.context_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        kv_width = self.config.kv_lora_rank + self.config.qk_rope_head_dim
        self.context_kv_proj = MergedColumnParallelLinear(
            self.config.hidden_size,
            [kv_width] * self.config.num_hidden_layers,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(
                prefix,
                f"layers.{start_layer_id}.self_attn.fused_qkv_a_proj",
            ),
            disable_tp=True,
        )
        self.final_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            self.config.draft_vocab_size,
            self.config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self._max_num_context_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.context_norm(self.context_proj(hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Project target-derived context into each draft layer's latent cache."""
        if not hasattr(self, "_num_context_layers"):
            self._build_fused_context_kv_metadata()
        self._precompute_fused_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def _build_fused_context_kv_metadata(self) -> None:
        """Build cross-layer metadata after checkpoint loading."""
        attentions = [layer.self_attn for layer in self.layers]
        assert attentions
        attn0 = attentions[0]
        assert attn0.q_lora_rank is not None
        kv_width = attn0.kv_lora_rank + attn0.qk_rope_head_dim
        for attn in attentions:
            assert attn.q_lora_rank is not None
            assert (
                attn.q_lora_rank == attn0.q_lora_rank
                and attn.kv_lora_rank == attn0.kv_lora_rank
                and attn.qk_rope_head_dim == attn0.qk_rope_head_dim
                and attn.kv_a_layernorm.variance_epsilon
                == attn0.kv_a_layernorm.variance_epsilon
            ), "All MLA DSpark layers must share their latent KV geometry."
        self._context_kv_norm_weights = torch.stack(
            [attn.kv_a_layernorm.weight.detach() for attn in attentions], dim=0
        ).contiguous()
        self._num_context_layers = len(attentions)
        self._context_kv_width = kv_width
        self._context_kv_lora_rank = attn0.kv_lora_rank
        self._context_rope_dim = attn0.qk_rope_head_dim
        self._context_rms_norm_eps = attn0.kv_a_layernorm.variance_epsilon

    def _precompute_fused_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        num_ctx = context_states.shape[0]
        num_layers = self._num_context_layers

        # One KV-only GEMM replaces five full Q+KV GEMMs. For K3 this projects
        # 5*576 rows rather than 5*2112 rows (72.7% fewer A-projection FLOPs).
        all_kv = self.context_kv_proj(context_states)
        all_kv = all_kv.view(num_ctx, num_layers, self._context_kv_width)
        all_kv_c = all_kv[..., : self._context_kv_lora_rank]
        all_k_pe = all_kv[..., self._context_kv_lora_rank :]

        # Layer-major layout lets the 2-D RMSNorm weights select a distinct row
        # for each draft layer in one grouped kernel.
        all_kv_c = all_kv_c.permute(1, 0, 2).contiguous()
        all_kv_c_normed = torch.empty_like(all_kv_c)
        ops.rms_norm(
            all_kv_c_normed,
            all_kv_c,
            self._context_kv_norm_weights,
            self._context_rms_norm_eps,
        )

        all_k_pe = all_k_pe.permute(1, 0, 2).contiguous()
        all_k_pe_flat = all_k_pe.view(num_layers * num_ctx, 1, self._context_rope_dim)
        (repeated_positions,) = current_workspace_manager().get_simultaneous(
            ((num_layers * self._max_num_context_tokens,), torch.int64),
        )
        repeated_positions = repeated_positions[: num_layers * num_ctx]
        repeated_positions.view(num_layers, num_ctx).copy_(context_positions)
        # Keep the single-tensor context RoPE on vLLM's optimized CUDA op;
        # DeepSeek YaRN's FlashInfer wrapper assumes a non-null key tensor.
        rotary_emb = self.layers[0].self_attn.rotary_emb
        assert rotary_emb is not None
        ops.rotary_embedding(
            repeated_positions,
            all_k_pe_flat,
            None,
            rotary_emb.head_size,
            rotary_emb.cos_sin_cache,
            rotary_emb.is_neox_style,
        )
        all_k_pe = all_k_pe_flat.view(num_layers, num_ctx, 1, self._context_rope_dim)

        if context_slot_mapping is None:
            return

        cache_layers = [layer.self_attn for layer in self.layers]
        if (
            not is_quantized_kv_cache(cache_layers[0].kv_cache_dtype)
            and self._has_uniform_block_layout(cache_layers)
            and (
                isinstance(context_slot_mapping, torch.Tensor)
                or all(s is not None for s in context_slot_mapping)
            )
        ):
            # Grouped context KV insert only supports unquantized (bf16) KV cache
            # and assumes that all layers share the same block layout.

            if isinstance(context_slot_mapping, (list, tuple)):
                per_layer_slot_mappings = [
                    s for s in context_slot_mapping if s is not None
                ]
                if len({s.data_ptr() for s in per_layer_slot_mappings}) == 1:
                    # All rows alias to the same slot mapping.
                    slot_mapping = (
                        per_layer_slot_mappings[0].unsqueeze(0).expand(num_layers, -1)
                    )
                else:
                    slot_mapping = torch.stack(per_layer_slot_mappings, dim=0)
            else:
                # Broadcast the single shared context_slot_mapping tensor.
                slot_mapping = context_slot_mapping.unsqueeze(0).expand(num_layers, -1)

            ref_cache = cache_layers[0].kv_cache
            ops.concat_and_cache_mla_grouped(
                all_kv_c_normed,
                all_k_pe.squeeze(2),
                self._get_context_kv_cache_ptrs(cache_layers),
                slot_mapping,
                ref_cache.size(1),
                ref_cache.stride(0),
                ref_cache.stride(1),
            )
            return

        for layer_idx, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mapping[layer_idx]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            attn = layer.self_attn
            attn.impl.do_kv_cache_update(
                all_kv_c_normed[layer_idx],
                all_k_pe[layer_idx],
                attn.kv_cache,
                slot_mapping,
                attn.kv_cache_dtype,
                attn._k_scale,
            )

    def _has_uniform_block_layout(
        self,
        cache_layers: list[MultiHeadLatentAttention],
    ) -> bool:
        if not hasattr(self, "_layers_share_kv_block_layout"):
            ref_cache = cache_layers[0].kv_cache
            self._layers_share_kv_block_layout = all(
                cl.kv_cache.size(1) == ref_cache.size(1)
                and cl.kv_cache.stride(0) == ref_cache.stride(0)
                and cl.kv_cache.stride(1) == ref_cache.stride(1)
                for cl in cache_layers
            )
        return self._layers_share_kv_block_layout

    def _get_context_kv_cache_ptrs(
        self,
        cache_layers: list[MultiHeadLatentAttention],
    ) -> torch.Tensor:
        # The per-layer KV cache base pointers are stable after allocation, so
        # build the pointer array once and return it on every call.
        if not hasattr(self, "_context_cache_ptrs"):
            ref_cache = cache_layers[0].kv_cache
            cache_ptrs = torch.tensor(
                [cl.kv_cache.data_ptr() for cl in cache_layers],
                dtype=torch.int64,
                device=ref_cache.device,
            )
            self._context_cache_ptrs = cache_ptrs
        return self._context_cache_ptrs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = fused_allreduce_rms_norm(
            hidden_states, residual, self.final_norm
        )
        return hidden_states


class K3DSparkForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    checkpoint_skip_substrs = ("confidence_head", "embed_tokens", "lm_head")

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"": "model."},
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = K3DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=target_layer_num,
            prefix=maybe_prefix(prefix, "model"),
        )

        # Assigned by load_dspark_model from the target. Keeping no placeholder
        # avoids a transient full-vocabulary allocation for this 163k-vocab model.
        self.lm_head: nn.Module | None = None
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.layer_name for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # confidence_head is training-only. The frozen target embedding and LM
        # head are shared after this draft-specific checkpoint is loaded.
        loader = AutoWeightsLoader(
            self,
            skip_substrs=list(self.checkpoint_skip_substrs),
        )
        # read: 1. all weights. 2. context kv weights
        weights = _duplicate_context_kv_weights(weights, len(self.model.layers))
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model._build_fused_context_kv_metadata()
        return loaded_weights
