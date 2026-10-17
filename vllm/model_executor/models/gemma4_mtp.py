# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Gemma4 MTP (Multi-Token Prediction) model.

The Gemma4 assistant model is a lightweight decoder that shares KV cache
with the target (backbone) model.  All assistant decoder layers are
KV-shared: they only have Q projections (no K/V projections or norms),
and read K/V from the target model's cache at runtime.

Checkpoint layout (``gemma4_assistant``)::

    model.embed_tokens.*          -- token embeddings
    model.layers.{i}.*            -- decoder layers (Q-only attention + MLP)
    model.norm.*                  -- final RMSNorm
    pre_projection.*              -- Linear(2 * backbone_hidden_size, hidden_size)
    post_projection.*             -- Linear(hidden_size, backbone_hidden_size)
    lm_head.*                     -- language model head (tied to embed_tokens)
    masked_embedding.centroids.*  -- centroid projection (when use_ordered_embeddings)
    masked_embedding.token_ordering -- token-to-centroid mapping buffer
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import HAS_TRITON, tl, triton

from .gemma4 import Gemma4MLP, _get_text_config
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    get_draft_quant_config,
    maybe_prefix,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Fused MTP Triton Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_gather_gemv_kernel(
    hidden_states_ptr,
    lm_head_weight_ptr,
    selected_indices_ptr,
    out_ptr,
    stride_h_t,
    stride_h_h,
    stride_w_v,
    stride_w_h,
    stride_idx_t,
    stride_idx_s,
    stride_out_t,
    stride_out_s,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_s = tl.program_id(1).to(tl.int64)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    idx_ptrs = selected_indices_ptr + pid_t * stride_idx_t + offs_s * stride_idx_s
    vocab_idxs = tl.load(idx_ptrs, mask=mask_s, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        h_vals = tl.load(
            hidden_states_ptr + pid_t * stride_h_t + offs_h * stride_h_h,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)
        w_ptrs = (
            lm_head_weight_ptr
            + vocab_idxs[:, None] * stride_w_v
            + offs_h[None, :] * stride_w_h
        )
        mask_2d = mask_s[:, None] & mask_h[None, :]
        w_block = tl.load(w_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
        acc += tl.sum(w_block * h_vals[None, :], axis=1)

    out_ptrs = out_ptr + pid_t * stride_out_t + offs_s * stride_out_s
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_s)


def fused_mtp_sparse_gather_gemv(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute sparse dot products einsum('td,tsd->ts') in-register."""
    T, H = hidden_states.shape
    S = selected_indices.shape[1]
    if T == 0 or S == 0:
        return torch.empty(
            (T, S), dtype=hidden_states.dtype, device=hidden_states.device
        )

    if not HAS_TRITON or not hidden_states.is_cuda:
        embeddings = lm_head_weight[selected_indices.reshape(-1)].view(T, S, H)
        return torch.einsum("td,tsd->ts", hidden_states, embeddings)

    if hidden_states.stride(-1) != 1:
        hidden_states = hidden_states.contiguous()
    if lm_head_weight.stride(-1) != 1:
        lm_head_weight = lm_head_weight.contiguous()
    if selected_indices.stride(-1) != 1:
        selected_indices = selected_indices.contiguous()

    out = torch.empty((T, S), dtype=hidden_states.dtype, device=hidden_states.device)
    BLOCK_H = min(256, max(16, triton.next_power_of_2(H)))
    BLOCK_S = 16 if T <= 2 else 32
    num_warps = 4

    grid = (T, triton.cdiv(S, BLOCK_S))
    _sparse_gather_gemv_kernel[grid](
        hidden_states,
        lm_head_weight,
        selected_indices,
        out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        lm_head_weight.stride(0),
        lm_head_weight.stride(1),
        selected_indices.stride(0),
        selected_indices.stride(1),
        out.stride(0),
        out.stride(1),
        S=S,
        H=H,
        BLOCK_S=BLOCK_S,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return out


class Gemma4MTPMaskedEmbedder(nn.Module):
    """Sparse logit computation via centroid-based vocabulary masking.

    Instead of computing logits against the full vocabulary, projects
    hidden states to centroid scores, selects top-K centroids, and
    computes logits only for the ~top_k * (vocab_size / num_centroids)
    tokens belonging to those centroids.
    """

    token_ordering: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_centroids: int,
        centroid_intermediate_top_k: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        self.vocab_size_per_centroid = vocab_size // num_centroids
        self.num_selected = centroid_intermediate_top_k * self.vocab_size_per_centroid

        self.centroids = nn.Linear(hidden_size, num_centroids, bias=False)
        self.register_buffer(
            "token_ordering",
            torch.empty(vocab_size, dtype=torch.long),
        )

    def _select_and_score(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Centroid selection + sparse dot product.

        Returns:
            logits: (num_tokens, num_selected) sparse logits.
            indices: (num_tokens, num_selected) corresponding vocab indices.

        """
        num_tokens = hidden_states.shape[0]
        _, top_k_indices = torch.topk(
            self.centroids(hidden_states),
            k=self.centroid_intermediate_top_k,
            dim=-1,
        )
        clusters = self.token_ordering.view(
            self.num_centroids,
            self.vocab_size_per_centroid,
        )
        selected = clusters[top_k_indices]
        selected_flat = selected.view(num_tokens, self.num_selected)
        logits = fused_mtp_sparse_gather_gemv(
            hidden_states, lm_head_weight, selected_flat
        )
        return logits, selected_flat

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Full-vocab logits with non-selected positions masked to -inf."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        output = torch.full(
            (hidden_states.shape[0], self.vocab_size),
            fill_value=torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return output.scatter_(-1, indices, logits)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse argmax — returns vocab token IDs without full-vocab tensor."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        return indices.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)


class Gemma4MTPAttention(nn.Module):
    """Q-only attention for Gemma4 MTP layers.

    K/V come from the target model's KV cache via
    ``kv_sharing_target_layer_name`` (set by the proposer after
    model construction).
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        attn_logits_soft_cap: float | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.scaling = 1.0

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = config.sliding_window if self.is_sliding else None

        if layer_type in config.rope_parameters:
            rope_parameters = dict(config.rope_parameters[layer_type])
        else:
            rope_parameters = dict(config.rope_parameters.copy())
            if self.is_sliding:
                rope_parameters["rope_theta"] = getattr(
                    config, "rope_local_base_freq", 10000.0
                )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        # kv_sharing_target_layer_name is set after model construction
        # by Gemma4Proposer._setup_gemma4_kv_sharing().
        self.is_kv_shared_layer = True
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=attn_logits_soft_cap,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        q, _ = self.rotary_emb(positions, q, None)

        # Attention reads K/V from the target's cache via KV sharing.
        attn_output = self.attn(q, None, None)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4MTPDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        layer_idx = extract_layer_index(prefix)
        layer_config = config.per_layer_config[layer_idx]
        head_dim = layer_config.head_dim
        num_kv_heads = layer_config.num_key_value_heads

        self.self_attn = Gemma4MTPAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=getattr(config, "attn_logit_softcapping", None),
            prefix=f"{prefix}.self_attn",
        )

        text_config = _get_text_config(config)
        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_activation=text_config.hidden_activation,
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

        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


class Gemma4MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        config = speculative_config.draft_model_config.hf_config
        text_config = _get_text_config(config)
        quant_config = get_draft_quant_config(vllm_config)
        self.config = text_config
        self.quant_config = quant_config

        self.hidden_size = text_config.hidden_size
        self.backbone_hidden_size = getattr(
            config, "backbone_hidden_size", self.hidden_size
        )
        self.vocab_size = text_config.vocab_size
        self.num_mtp_layers = text_config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.pre_projection = ColumnParallelLinear(
            2 * self.backbone_hidden_size,
            self.hidden_size,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.pre_projection",
        )

        self.post_projection = RowParallelLinear(
            self.hidden_size,
            self.backbone_hidden_size,
            bias=False,
            input_is_parallel=False,
            quant_config=quant_config,
            prefix=f"{prefix}.post_projection",
        )

        self.layers = nn.ModuleList(
            Gemma4MTPDecoderLayer(
                text_config,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )

        self.norm = RMSNorm(self.hidden_size, eps=text_config.rms_norm_eps)

        # After embedding sharing, embed_tokens is replaced with the
        # target model's backbone-dim embedding.  Scale by
        # sqrt(backbone_hidden_size) to match the target's convention.
        self.register_buffer(
            "normalizer",
            torch.tensor(self.backbone_hidden_size**0.5),
            persistent=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (draft_hidden_states, backbone_hidden_states).

        draft_hidden_states: draft-dim, used by compute_logits via lm_head.
        backbone_hidden_states: backbone-dim, stored in the proposer's
            hidden-state buffer and fed back as input to the next step.
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        combined = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states, _ = self.pre_projection(combined)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        draft_hidden_states = self.norm(hidden_states)

        backbone_hidden_states, _ = self.post_projection(draft_hidden_states)
        return draft_hidden_states, backbone_hidden_states


@support_torch_compile
class Gemma4MTP(nn.Module):
    """Gemma4 Multi-Token Prediction model for speculative decoding.

    forward() returns (draft_hidden_states, backbone_hidden_states).
    The proposer uses draft_hidden_states for compute_logits (via
    the draft-dim lm_head) and backbone_hidden_states for the
    hidden-state feedback buffer.
    """

    has_own_lm_head = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "pre_projection.": "model.pre_projection.",
            "post_projection.": "model.post_projection.",
        },
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        config = speculative_config.draft_model_config.hf_config
        text_config = _get_text_config(config)
        self.quant_config = get_draft_quant_config(vllm_config)
        self.config = config
        self._stable_full_lm_head_weight: torch.Tensor | None = None

        self.model = Gemma4MultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "draft_model"),
        )

        # lm_head operates in draft-dim.  Tied to embed_tokens at init
        # so load_weights populates both from a single checkpoint entry.
        # After embedding sharing, lm_head.weight still references the
        # original draft-dim tensor.
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            text_config.vocab_size,
            soft_cap=getattr(text_config, "final_logit_softcapping", None),
        )

        self.masked_embedding: Gemma4MTPMaskedEmbedder | None
        if getattr(config, "use_ordered_embeddings", False):
            num_centroids = getattr(config, "num_centroids", 2048)
            top_k = getattr(config, "centroid_intermediate_top_k", 32)
            self.masked_embedding = Gemma4MTPMaskedEmbedder(
                hidden_size=text_config.hidden_size,
                vocab_size=text_config.vocab_size,
                num_centroids=num_centroids,
                centroid_intermediate_top_k=top_k,
            )
            logger.info(
                "Gemma4 MTP: centroids masking enabled "
                "(num_centroids=%d, top_k=%d, active_tokens=%d/%d).",
                num_centroids,
                top_k,
                top_k * (text_config.vocab_size // num_centroids),
                text_config.vocab_size,
            )
        else:
            self.masked_embedding = None

        draft_cfg = speculative_config.draft_model_config
        gen_cfg = draft_cfg.try_get_generation_config()
        self._suppress_token_ids = gen_cfg.get("suppress_tokens") if gen_cfg else None
        # Materialized on-device in load_weights: compute_logits runs under CUDA
        # graph capture in the V2 speculator, where indexing with a Python list
        # would issue an unpinned H2D copy (illegal during capture).
        self._suppress_idx: torch.Tensor | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx,
        )

    def _get_full_lm_head_weight(self) -> torch.Tensor:
        if self._stable_full_lm_head_weight is not None:
            return self._stable_full_lm_head_weight
        assert self.masked_embedding is not None
        lm_head_weight = self.lm_head.weight
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            lm_head_weight = tensor_model_parallel_all_gather(
                lm_head_weight,
                dim=0,
            )
        lm_head_weight = lm_head_weight[: self.masked_embedding.vocab_size]
        if tp_size > 1:
            lm_head_weight = lm_head_weight.contiguous()
            self._stable_full_lm_head_weight = lm_head_weight
        return lm_head_weight

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        if self.masked_embedding is not None:
            logits = self.masked_embedding(
                hidden_states,
                self._get_full_lm_head_weight(),
            )
        else:
            logits = self.logits_processor(self.lm_head, hidden_states)
        suppress_idx = getattr(self, "_suppress_idx", None)
        if logits is not None and suppress_idx is not None:
            logits.index_fill_(1, suppress_idx, -float("inf"))
        return logits

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return greedy top token IDs directly without materializing full-vocab
        softcapped logits.
        """
        if self.masked_embedding is not None:
            return self.masked_embedding.get_top_tokens(
                hidden_states,
                self._get_full_lm_head_weight(),
            )
        if self.lm_head.tp_size == 1 and self.logits_processor.scale > 0.0:
            logits = self.logits_processor._apply_head(
                self.lm_head, hidden_states, None
            )
            suppress_idx = getattr(self, "_suppress_idx", None)
            if suppress_idx is not None:
                logits.index_fill_(1, suppress_idx, -float("inf"))
            return logits[:, : self.logits_processor.org_vocab_size].argmax(dim=-1)
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self._stable_full_lm_head_weight = None
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if self._suppress_token_ids:
            self._suppress_idx = torch.tensor(
                self._suppress_token_ids,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        return loaded
