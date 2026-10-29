# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral Large 4 DSpark draft model (dense MLA backbone + Markov head).

AMD/ROCm counterpart of ``vllm.models.mistral_large4.nvidia.dspark``: same
checkpoint layout and draft architecture, built on the AMD MLA attention and
MLP/MoE blocks of ``mistral_large4.amd.model``.

Draft layers are non-causal dense MLA blocks with the target's attention
geometry. Each draft layer caches latent K/V projected from the normalized
target taps (``combine_hidden_states``) and from its own query block. Draft
weights are BF16 and the draft RoPE is the target's YaRN without LeRoPE.
Checkpoint options select the trained variant:

* ``dspark_head_type="rnn"``: the Markov bias comes from a gated recurrent
  state carried across the draft positions of one proposal instead of from
  the previous token alone.
* ``dspark_draft_input="position_specific"``: ``mask_embedding`` has one row
  per mask slot; slot ``k`` of a ``[anchor, mask_1, ...]`` block reads row
  ``k - 1`` instead of sharing one vector.
* ``layer_types``/``sliding_window``: windowed draft layers see the last
  ``sliding_window`` context tokens before the anchor plus their whole block.
"""

from collections.abc import Iterable

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import get_draft_quant_config, maybe_prefix
from vllm.models.mistral.common.layer_spec import MistralLayerSpec
from vllm.models.mistral.common.rope import MistralRotaryEmbedding
from vllm.models.mistral_large4.amd.model import (
    _STACKED_PARAMS,
    MistralLarge4DecoderLayer,
    _to_vllm_weight_name,
)

# Mistral DSpark checkpoint name (after the ``mtp_dspark.`` prefix) -> module.
_DSPARK_NAME_MAP: list[tuple[str, str]] = [
    (r"^backbone\.target_projection\.", "model.target_projection."),
    (r"^backbone\.target_norm\.", "model.target_norm."),
    (r"^backbone\.final_norm\.", "model.final_norm."),
    (r"^backbone\.mask_embedding$", "model.mask_embedding"),
    (r"^autoregressive_head\.embedding\.", "model.markov_head.markov_w1."),
    (r"^autoregressive_head\.output\.", "model.markov_head.markov_w2."),
    (r"^autoregressive_head\.joint_proj\.", "model.joint_proj."),
]
_DSPARK_PREFIX = "mtp_dspark."


def _to_draft_weight_name(ckpt_name: str) -> str:
    name = ckpt_name.removeprefix(_DSPARK_PREFIX)
    if name.startswith("backbone.layers."):
        return _to_vllm_weight_name(name.removeprefix("backbone."))
    for pattern, repl in _DSPARK_NAME_MAP:
        name = re.sub(pattern, repl, name)
    return name


def draft_attention_windows(config, num_speculative_tokens: int) -> list[int | None]:
    """Per-layer decode window (keys counted back from the end of the block).

    The draft block is part of the attended sequence, so the window is widened
    by its length: the queries then see the ``sliding_window`` context tokens up
    to the anchor's source position, as in training.
    """
    layer_types = getattr(config, "layer_types", None) or (
        ["full_attention"] * config.num_hidden_layers
    )
    return [
        config.sliding_window + num_speculative_tokens
        if layer_type == "sliding_attention"
        else None
        for layer_type in layer_types
    ]


class MistralLarge4DSparkModel(nn.Module):
    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int, prefix: str = ""
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = get_draft_quant_config(vllm_config)
        hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.block_size = config.dspark_block_size
        num_spec = vllm_config.speculative_config.num_speculative_tokens

        # Aliased to the target input embedding after loading.
        self.embed_tokens: nn.Module | None = None
        self.position_specific = config.dspark_draft_input == "position_specific"
        mask_shape: tuple[int, ...] = (hidden_size,)
        if self.position_specific:
            assert num_spec == self.block_size, (
                "Position-specific DSpark masks need num_speculative_tokens equal "
                f"to the trained depth {self.block_size}, got {num_spec}"
            )
            mask_shape = (self.block_size - 1, hidden_size)
        self.mask_embedding = nn.Parameter(torch.zeros(mask_shape), requires_grad=False)

        self.target_projection: ReplicatedLinear | None = None
        if config.num_target_layers > 1:
            self.target_projection = ReplicatedLinear(
                config.target_hidden_size * config.num_target_layers,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "target_projection"),
            )
        self.target_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        # No LeRoPE alpha is set: the draft uses the target's plain YaRN.
        self.rotary_emb = MistralRotaryEmbedding(config, config.qk_rope_head_dim)
        windows = draft_attention_windows(config, num_spec)
        self.layers = nn.ModuleList(
            MistralLarge4DecoderLayer(
                vllm_config,
                config,
                MistralLayerSpec(
                    layer_idx=start_layer_id + i,
                    use_rope=True,
                    kv_source_layer=start_layer_id + i,
                    use_sparse=False,
                    has_indexer=False,
                    indexer_use_rope=False,
                    is_moe=False,
                ),
                prefix=maybe_prefix(prefix, f"layers.{start_layer_id + i}"),
                topk_indices_buffer=None,
                index_group_builder=None,
                # Only read by KV-sharing and NoPE layers, which drafts lack.
                zero_kv_inputs=(torch.empty(0), torch.empty(0)),
                quant_config=quant_config,
                non_causal=True,
                sliding_window=windows[i],
            )
            for i in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        markov_rank = config.markov_rank
        self.markov_head = DSparkMarkovHead(
            self.vocab_size,
            self.vocab_size,
            markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=quant_config,
        )
        self.joint_proj: ReplicatedLinear | None = None
        if config.dspark_head_type == "rnn":
            self.joint_proj = ReplicatedLinear(
                hidden_size + 2 * markov_rank,
                3 * markov_rank,
                bias=True,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "joint_proj"),
            )
        self.confidence_head = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed draft blocks, replacing the out-of-vocabulary mask slots."""
        assert self.embed_tokens is not None
        if self.position_specific:
            # Contiguous [anchor, mask_1, ..., mask_{D-1}] blocks per request.
            assert input_ids.numel() % self.block_size == 0, (
                f"Expected whole draft blocks of {self.block_size} tokens, got "
                f"{input_ids.numel()}"
            )
            anchors = self.embed_tokens(
                input_ids.view(-1, self.block_size)[:, 0].contiguous()
            )
            masks = self.mask_embedding.to(anchors.dtype).expand(
                anchors.shape[0], -1, -1
            )
            return torch.cat((anchors.unsqueeze(1), masks), dim=1).flatten(0, 1)
        is_mask = input_ids >= self.vocab_size
        embeds = self.embed_tokens(input_ids.masked_fill(is_mask, 0))
        return torch.where(
            is_mask.unsqueeze(-1), self.mask_embedding.to(embeds.dtype), embeds
        )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.target_projection is not None:
            hidden_states = self.target_projection(hidden_states)[0]
        return self.target_norm(hidden_states)

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Write each draft layer's latent K/V for the target context tokens.

        The context is already normalized (``target_norm``), so it bypasses the
        layers' input norms, as in training.
        """
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            kv = attn.kv_a_proj_with_mqa(context_states)[0]
            kv_c_normed = attn.kv_a_layernorm(kv[:, : attn.kv_lora_rank])
            k_pe = kv[:, None, attn.kv_lora_rank :]
            _, k_pe = self.rotary_emb(context_positions, torch.empty_like(k_pe), k_pe)
            slot_mapping = (
                context_slot_mapping[i]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            mla = attn.attn
            mla.impl.do_kv_cache_update(
                kv_c_normed,
                k_pe,
                mla.kv_cache,
                slot_mapping,
                mla.kv_cache_dtype,
                mla._k_scale,
            )

    def markov_step(
        self,
        token_ids: torch.Tensor,
        draft_hidden: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent Markov step: (output embedding, next state)."""
        assert self.joint_proj is not None
        embeds = self.markov_head.embed(token_ids)
        if state is None:
            state = torch.zeros_like(embeds)
        joint = self.joint_proj(torch.cat((state, embeds, draft_hidden), dim=-1))[0]
        gate_raw, candidate_raw, output_raw = joint.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        next_state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        return torch.tanh(output_raw), next_state

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # Only consumed by SSMax, which draft layers do not use.
        log_positions = positions
        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer.input_norm(hidden_states, residual)
            hidden_states, residual = layer(
                positions, hidden_states, residual, log_positions, self.rotary_emb
            )
        hidden_states, _ = self.layers[-1].residual_norm(
            self.final_norm, hidden_states, residual
        )
        return hidden_states


class MistralLarge4DSparkForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = draft_model_config.hf_config
        self.model = MistralLarge4DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=vllm_config.model_config.get_total_num_hidden_layers(),
            prefix=maybe_prefix(prefix, "model"),
        )
        self.has_recurrent_markov_head = self.model.joint_proj is not None
        # Shared with the target after loading.
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            soft_cap=getattr(self.config, "softmax_tanh_gating", None),
        )
        # Training soft-caps the base logits only, not the Markov bias.
        self.markov_logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.model.layers)

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

    def markov_step(
        self,
        token_ids: torch.Tensor,
        draft_hidden: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.markov_step(token_ids, draft_hidden, state)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.markov_logits_processor)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for ckpt_name, weight in weights:
            if not ckpt_name.startswith(_DSPARK_PREFIX):
                continue
            name = _to_draft_weight_name(ckpt_name)
            for param_part, ckpt_part, shard_id in _STACKED_PARAMS:
                if ckpt_part not in name:
                    continue
                name = name.replace(ckpt_part, param_part)
                param = params[name]
                if shard_id < 0:
                    param.weight_loader(param, weight)
                else:
                    param.weight_loader(param, weight, shard_id)
                break
            else:
                if name not in params:
                    raise ValueError(f"Unexpected DSpark weight: {ckpt_name}")
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)

        # A missing mask or head would silently serve zeros.
        required = {"model.mask_embedding"}
        if self.model.target_projection is not None:
            required.add("model.target_projection.weight")
        if self.model.joint_proj is not None:
            required |= {
                name
                for name in params
                if name.startswith(("model.markov_head.", "model.joint_proj."))
            }
        if missing := required - loaded:
            raise ValueError(f"DSpark checkpoint is missing {sorted(missing)}")
        return loaded
