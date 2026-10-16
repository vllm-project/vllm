# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 draft model for Beam targets.

A DFlash2 drafter predicts a block of tokens at once from hidden states the
Beam target captures at `dflash_config.target_layer_ids`, then a low-rank
candidate selector rescores the top-k tokens at each block position against
the token before it. The drafter shares the target's embeddings and LM head.

Nothing here is torch-compiled. The DFlash speculator captures the whole draft
step (decoder layers, candidate head and selector) in its own CUDA graphs, and
the context K/V precompute runs eagerly as a few fused kernels.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
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
        cache, slots = None, None
        mapping = get_forward_context().slot_mapping
        assert isinstance(mapping, dict)
        if mapping and self.attn.kv_cache.dtype == q.dtype == torch.bfloat16:
            slots = mapping[self.attn.layer_name]
            cache = self.attn.kv_cache
            assert self.attn.attn_backend.get_name() in ("FLASH_ATTN", "FLASHINFER")
            assert self.attn.impl.dcp_world_size == 1
            assert self.attn.kv_sharing_target_layer_name is None
            assert cache.dtype == q.dtype == torch.bfloat16
            assert cache.ndim == 4
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
            value=v,
            kv_cache=cache,
            slot_mapping=slots,
        )
        output = self.attn(q, k, v) if cache is None else self.attn(q, None, None)
        output, _ = self.o_proj(output)
        return output


class DFlash2BeamDecoderLayer(DFlash2Qwen3DecoderLayer):
    attention_cls = DFlash2BeamAttention


class DFlash2BeamCandidateSelector(nn.Module):
    """Scores each candidate token against every candidate one position back.

    For block position `p`, candidate `c` and predecessor `q`, the score is the
    candidate's own logit plus `<predecessor[q] * h_p, successor[c]>`, where
    `h_p` is the drafter's hidden state projected to the codebook rank. The
    predecessor of the first position is the anchor token.
    """

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
        """Edge scores `[requests, steps, top_k (predecessor), top_k]`."""
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

        # Replaced by the target's embedding when the speculator loads the draft.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Numbered after the target's layers so the attention layer names, which
        # key the KV cache, do not collide with the target's.
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
        self._context_cache_ptrs: torch.Tensor | None = None
        self._context_scale_ptrs: torch.Tensor | None = None
        self._context_cache_addresses: tuple[int, ...] | None = None
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
        """Stack every layer's K/V projection and K-norm weight.

        The context K/V for all layers then comes from one GEMM and one grouped
        norm. Must run after the weights are loaded, since it copies them.
        """
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
        """Project the combined target states to each layer's K/V and cache them.

        The drafter's queries attend to these context keys, which carry the
        layer's K-norm and RoPE like the queries' own keys. Profiling uses a
        `None` slot mapping to run the projection without writing the cache.
        """
        if not hasattr(self, "_context_kv_weight"):
            # Dummy weights skip `load_weights`.
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
        self._store_fused_context_kv(kv, context_positions, context_slot_mapping)

    def _store_fused_context_kv(
        self,
        kv: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        from .dflash2_context_ops import store_context_kv

        if slot_mapping is None:
            return
        if isinstance(slot_mapping, (list, tuple)):
            slots = slot_mapping[0]
            if slots is None:
                assert all(s is None for s in slot_mapping)
                return
            assert all(
                s is not None
                and s.data_ptr() == slots.data_ptr()
                and s.shape == slots.shape
                and s.stride() == slots.stride()
                for s in slot_mapping
            )
        else:
            slots = slot_mapping
        first = self.layers[0].self_attn
        rotary = first.rotary_emb
        assert kv.dtype == torch.bfloat16
        assert rotary.rotary_dim == first.head_dim
        assert rotary.cos_sin_cache.dtype == kv.dtype
        assert slots.stride(0) == 1
        attns = [layer.self_attn.attn for layer in self.layers]
        caches = [attn.kv_cache for attn in attns]
        cache = caches[0]
        cache_dtype = attns[0].kv_cache_dtype
        fp8_dtype = {
            "fp8": torch.float8_e4m3fn,
            "fp8_e4m3": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }.get(cache_dtype)
        for attn, c in zip(attns, caches):
            assert attn.attn_backend.get_name() in ("FLASH_ATTN", "FLASHINFER")
            assert attn.impl.dcp_world_size == 1
            assert attn.kv_sharing_target_layer_name is None
            assert attn.kv_cache_dtype == cache_dtype
            assert c.dtype == cache.dtype and c.device == kv.device
            assert c.dtype == kv.dtype or fp8_dtype is not None
            assert c.ndim == 4 and c.stride(-1) == 1
            assert c.shape[1] == kv.shape[3]
            assert c.shape[-1] == 2 * kv.shape[-1]
            assert c.shape == cache.shape and c.stride() == cache.stride()
        cache_pointers = tuple(c.data_ptr() for c in caches)
        scale_pointers = (
            tuple(
                s.data_ptr() for attn in attns for s in (attn._k_scale, attn._v_scale)
            )
            if fp8_dtype is not None
            else ()
        )
        pointers = cache_pointers + scale_pointers
        if self._context_cache_addresses != pointers:
            self._context_cache_ptrs = torch.tensor(
                cache_pointers, dtype=torch.int64, device=kv.device
            )
            self._context_scale_ptrs = (
                torch.tensor(scale_pointers, dtype=torch.int64, device=kv.device)
                if scale_pointers
                else None
            )
            self._context_cache_addresses = pointers
        assert self._context_cache_ptrs is not None
        assert self._context_cache_ptrs.device == kv.device
        store_context_kv(
            kv,
            self._context_k_norm_weight,
            positions,
            rotary.cos_sin_cache,
            self._context_cache_ptrs,
            slots,
            cache.view(fp8_dtype) if fp8_dtype is not None else cache,
            first.k_norm.variance_epsilon,
            first.v_scale if first.v_scale is not None else 1.0,
            rotary.is_neox_style,
            self._context_scale_ptrs,
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
    """Carry normalized activations across layers to fuse both convolutions."""
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
    # The checkpoint ships neither, so the speculator always aliases the target's.
    has_own_embed_tokens = False
    has_own_lm_head = False
    # Full vocabulary: draft ids are target ids.
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
        """Project the concatenated target-layer states to the drafter's width."""
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
        """Per-layer causality, aligned with `get_draft_kv_cache_layer_names`."""
        return [layer.self_attn.causal for layer in self.model.layers]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The selector's top-k candidate ids and their logits at each position."""
        return self.logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = {f"model.{name}" for name in self.model.load_weights(weights)}
        self.model.build_context_kv_weights()
        # Aliased to the target's after loading.
        return loaded | {"model.embed_tokens.weight", "lm_head.weight"}
