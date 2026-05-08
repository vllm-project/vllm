# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeBERTa-v2/v3 model for vLLM.

Implements DebertaV2ForSequenceClassification as a cross-encoder (reranker)
scoring model using pooling task="classify".

Papers:
  DeBERTa:    https://arxiv.org/abs/2006.03654
  DeBERTa-v3: https://arxiv.org/abs/2111.09543
"""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DebertaV2Config
from transformers.activations import ACT2FN

from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.seqwise import (
    CLSPool,
    SequencePoolingMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.sequence import IntermediateTensors
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces import SupportsCrossEncoding
from .interfaces_base import default_pooling_type
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class DebertaV2Embeddings(nn.Module):
    """Word (+ optional token_type + position) embeddings for DeBERTa-v2/v3.

    DeBERTa-v3 sets ``position_biased_input=False`` and ``type_vocab_size=0``,
    so only word embeddings + LayerNorm are used in that variant.
    """

    def __init__(self, config: DebertaV2Config) -> None:
        super().__init__()
        self.config = config
        self.position_biased_input = getattr(config, "position_biased_input", True)

        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        if self.position_biased_input:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )

        if getattr(config, "type_vocab_size", 0) > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Register position_ids as a persistent buffer so the weight loader
        # accepts the 'embeddings.position_ids' tensor present in all HF
        # DeBERTa-v2/v3 checkpoints (even when position_biased_input=False).
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)

        if self.position_biased_input:
            embeddings = embeddings + self.position_embeddings(positions)

        if token_type_ids is not None and hasattr(self, "token_type_embeddings"):
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        return self.LayerNorm(embeddings)


# ---------------------------------------------------------------------------
# Disentangled (relative-position) self-attention
# ---------------------------------------------------------------------------


class DebertaV2DisentangledSelfAttention(nn.Module):
    """DeBERTa disentangled attention: c2c + c2p + p2c.

    Uses pure-PyTorch matmuls (no FlashAttention) to support the custom
    attention-bias computation required by relative-position embeddings.
    Processes each sequence in the batch independently, which is correct for
    vLLM's flat [total_tokens, hidden] tensor layout.
    """

    def __init__(
        self,
        config: DebertaV2Config,
        max_relative_positions: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_heads})"
        )

        self.num_heads = num_heads // tp_size  # local heads per TP rank
        self.head_dim = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_dim  # local hidden

        self.max_relative_positions = max_relative_positions
        # For models with position_buckets (DeBERTa-v3), log-scale bucketing maps
        # linear relative distances to bucket indices; max_position_embeddings is
        # used as the "max_position" argument in the bucketing formula.
        self.position_buckets = getattr(config, "position_buckets", -1)
        self.max_position_embeddings = config.max_position_embeddings

        pos_att_type_raw = getattr(config, "pos_att_type", "p2c|c2p")
        # HF configs may store pos_att_type as a list or a "|"-separated string
        if isinstance(pos_att_type_raw, (list, tuple)):
            self.pos_att_type = [p.strip().lower() for p in pos_att_type_raw]
        else:
            self.pos_att_type = [p.strip() for p in pos_att_type_raw.lower().split("|")]
        self.share_att_key = getattr(config, "share_att_key", True)

        # scale = sqrt(head_dim * scale_factor) where scale_factor counts
        # the number of attention terms (1 c2c + n position terms)
        self.scale_factor = 1 + len(self.pos_att_type)
        self.scale = math.sqrt(self.head_dim * self.scale_factor)

        self.query_proj = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            prefix=f"{prefix}.query_proj",
        )
        self.key_proj = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            prefix=f"{prefix}.value_proj",
        )

        if not self.share_att_key:
            if "c2p" in self.pos_att_type:
                self.pos_key_proj = ColumnParallelLinear(
                    hidden_size,
                    hidden_size,
                    bias=True,
                    prefix=f"{prefix}.pos_key_proj",
                )
            if "p2c" in self.pos_att_type:
                self.pos_query_proj = ColumnParallelLinear(
                    hidden_size,
                    hidden_size,
                    bias=True,
                    prefix=f"{prefix}.pos_query_proj",
                )

    def _log_bucket_positions(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """Apply DeBERTa-v2 log-scale position bucketing.

        Mirrors HF's ``make_log_bucket_position``:
          - positions within ±mid are kept linear
          - larger distances are log-compressed into [±mid, ±(bucket_size-1)]

        Args:
            rel_pos: [L, L] LongTensor of raw (linear) relative positions q-k.

        Returns:
            [L, L] LongTensor of bucketed positions in
            [-(bucket_size-1), bucket_size-1].
        """
        mid = self.position_buckets // 2  # half the bucket count (e.g. 128)
        max_pos = self.max_position_embeddings  # e.g. 512

        sign = torch.sign(rel_pos).float()
        abs_pos = torch.where(
            (rel_pos < mid) & (rel_pos > -mid),
            torch.full_like(rel_pos, mid - 1, dtype=torch.float),
            rel_pos.abs().float(),
        )
        log_ratio = math.log((max_pos - 1) / mid)
        log_pos = (
            torch.ceil(torch.log(abs_pos / mid) / log_ratio * (mid - 1)).float() + mid
        )
        bucket_pos = torch.where(
            rel_pos.abs() < mid,
            rel_pos.float(),
            log_pos * sign,
        )
        return bucket_pos.long()

    def _project_rel_emb(
        self, rel_emb: torch.Tensor, proj: ColumnParallelLinear
    ) -> torch.Tensor:
        """Project relative embeddings and reshape to [num_heads, head_dim, N]."""
        out, _ = proj(rel_emb)  # [N, all_head_size]
        # [N, num_heads, head_dim] → [num_heads, head_dim, N]
        return out.view(-1, self.num_heads, self.head_dim).permute(1, 2, 0)

    def _compute_disentangled_attn(
        self,
        q_s: torch.Tensor,  # [num_heads, L, head_dim]
        k_s: torch.Tensor,  # [num_heads, L, head_dim]
        v_s: torch.Tensor,  # [num_heads, L, head_dim]
        rel_embeddings: torch.Tensor,  # [2*max_pos, hidden_size]
        L: int,
        device: torch.device,
    ) -> torch.Tensor:  # [L, all_head_size]
        # Always use the full embedding table (no clipping to L).
        # rel_embeddings already has shape [2*max_relative_positions, hidden_size]
        # which equals [2*att_span, hidden_size].
        att_span = self.max_relative_positions

        # --- c2c (content to content) ----------------------------------------
        attn_scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / self.scale
        # [num_heads, L, L]

        # Raw relative positions: rel_pos[q, k] = q - k  (linear)
        q_idx = torch.arange(L, dtype=torch.long, device=device)
        rel_pos = q_idx.unsqueeze(1) - q_idx.unsqueeze(0)  # [L, L]

        # Apply log-scale bucketing when the model uses position_buckets
        # (DeBERTa-v3 style); mirrors HF's make_log_bucket_position.
        if self.position_buckets > 0:
            rel_pos = self._log_bucket_positions(rel_pos)

        # Shared position index for both c2p and p2c:
        #   pos_idx[q, k] = clamp(bucketed(q-k) + att_span, 0, 2*att_span-1)
        # Both terms use the same q→k direction (verified against HF source).
        pos_idx = (rel_pos + att_span).clamp(0, 2 * att_span - 1)  # [L, L]

        # --- c2p (content query → relative-position key) ----------------------
        if "c2p" in self.pos_att_type:
            proj = self.key_proj if self.share_att_key else self.pos_key_proj
            # pos_key: [num_heads, head_dim, 2*att_span]
            pos_key = self._project_rel_emb(rel_embeddings, proj)
            # c2p_raw[h, q, r] = Q_q · PK_r
            c2p_raw = torch.matmul(q_s, pos_key) / self.scale  # [H, L, 2*att_span]
            c2p_idx = pos_idx.unsqueeze(0).expand(self.num_heads, -1, -1)  # [H, L, L]
            attn_scores = attn_scores + torch.gather(c2p_raw, dim=-1, index=c2p_idx)

        # --- p2c (content key → relative-position query) ----------------------
        if "p2c" in self.pos_att_type:
            proj = self.query_proj if self.share_att_key else self.pos_query_proj
            # pos_query: [num_heads, head_dim, 2*att_span]
            pos_query = self._project_rel_emb(rel_embeddings, proj)
            # p2c_raw[h, k, r] = K_k · PQ_r
            p2c_raw = torch.matmul(k_s, pos_query) / self.scale  # [H, L_k, 2*att_span]
            # Desired: result[h, q, k] = p2c_raw[h, k, pos_idx[q, k]]
            # Expand p2c_raw along the query axis → [H, L_q, L_k, 2*att_span]
            p2c_expanded = p2c_raw.unsqueeze(1).expand(-1, L, -1, -1)
            p2c_idx_4d = (
                pos_idx.unsqueeze(0).unsqueeze(-1).expand(self.num_heads, -1, -1, 1)
            )
            attn_scores = attn_scores + torch.gather(
                p2c_expanded, dim=-1, index=p2c_idx_4d
            ).squeeze(-1)

        # Softmax + weighted sum
        attn_probs = F.softmax(attn_scores, dim=-1)  # [H, L, L]
        context = torch.matmul(attn_probs, v_s)  # [H, L, head_dim]
        # [H, L, head_dim] → [L, H, head_dim] → [L, all_head_size]
        context = context.permute(1, 0, 2).contiguous().view(L, self.all_head_size)
        return context

    def forward(
        self,
        hidden_states: torch.Tensor,  # [total_tokens, hidden_size]
        positions: torch.Tensor,  # [total_tokens] — resets to 0 at each sequence
        rel_embeddings: torch.Tensor,  # [2*max_pos, hidden_size]
    ) -> torch.Tensor:  # [total_tokens, all_head_size]
        total_tokens = hidden_states.shape[0]
        device = hidden_states.device

        q, _ = self.query_proj(hidden_states)  # [total_tokens, all_head_size]
        k, _ = self.key_proj(hidden_states)
        v, _ = self.value_proj(hidden_states)

        # Sequence boundaries: position resets to 0 at the start of each sequence
        seq_starts = (positions == 0).nonzero(as_tuple=True)[0]
        seq_ends = torch.cat([seq_starts[1:], positions.new_tensor([total_tokens])])

        outputs: list[torch.Tensor] = []
        for start, end in zip(seq_starts.tolist(), seq_ends.tolist()):
            L = end - start

            # [H, L, head_dim]
            q_s = q[start:end].view(L, self.num_heads, self.head_dim).permute(1, 0, 2)
            k_s = k[start:end].view(L, self.num_heads, self.head_dim).permute(1, 0, 2)
            v_s = v[start:end].view(L, self.num_heads, self.head_dim).permute(1, 0, 2)

            ctx = self._compute_disentangled_attn(
                q_s, k_s, v_s, rel_embeddings, L, device
            )  # [L, all_head_size]
            outputs.append(ctx)

        return torch.cat(outputs, dim=0)  # [total_tokens, all_head_size]


# ---------------------------------------------------------------------------
# Attention output, intermediate, layer-output sub-modules
# ---------------------------------------------------------------------------


class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config: DebertaV2Config, prefix: str = "") -> None:
        super().__init__()
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + residual)


class DebertaV2Attention(nn.Module):
    def __init__(
        self,
        config: DebertaV2Config,
        max_relative_positions: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self = DebertaV2DisentangledSelfAttention(
            config, max_relative_positions, prefix=f"{prefix}.self"
        )
        self.output = DebertaV2SelfOutput(config, prefix=f"{prefix}.output")

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        self_out = self.self(hidden_states, positions, rel_embeddings)
        return self.output(self_out, hidden_states)


class DebertaV2Intermediate(nn.Module):
    def __init__(self, config: DebertaV2Config, prefix: str = "") -> None:
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            prefix=f"{prefix}.dense",
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        return self.act_fn(hidden_states)


class DebertaV2LayerOutput(nn.Module):
    def __init__(self, config: DebertaV2Config, prefix: str = "") -> None:
        super().__init__()
        self.dense = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + residual)


# ---------------------------------------------------------------------------
# Transformer layer and encoder
# ---------------------------------------------------------------------------


class DebertaV2Layer(nn.Module):
    def __init__(
        self,
        config: DebertaV2Config,
        max_relative_positions: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attention = DebertaV2Attention(
            config, max_relative_positions, prefix=f"{prefix}.attention"
        )
        self.intermediate = DebertaV2Intermediate(
            config, prefix=f"{prefix}.intermediate"
        )
        self.output = DebertaV2LayerOutput(config, prefix=f"{prefix}.output")

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attention(hidden_states, positions, rel_embeddings)
        intermediate_out = self.intermediate(attn_out)
        return self.output(intermediate_out, attn_out)


class DebertaV2Encoder(nn.Module):
    """Stack of DeBERTa transformer layers with a shared relative-position table."""

    def __init__(
        self,
        config: DebertaV2Config,
        max_relative_positions: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [
                DebertaV2Layer(
                    config,
                    max_relative_positions,
                    prefix=f"{prefix}.layer.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Shared relative position embedding table
        self.rel_embeddings = nn.Embedding(
            2 * max_relative_positions, config.hidden_size
        )

        # Optional LayerNorm applied to rel_embeddings before each forward pass
        norm_rel_ebd = getattr(config, "norm_rel_ebd", "none")
        if norm_rel_ebd.lower() != "none":
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _get_rel_embeddings(self) -> torch.Tensor:
        rel_emb = self.rel_embeddings.weight
        if hasattr(self, "LayerNorm"):
            rel_emb = self.LayerNorm(rel_emb)
        return rel_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        rel_embeddings = self._get_rel_embeddings()
        for layer in self.layer:
            hidden_states = layer(hidden_states, positions, rel_embeddings)
        return hidden_states


# ---------------------------------------------------------------------------
# Backbone model
# ---------------------------------------------------------------------------


class DebertaV2Model(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: DebertaV2Config = vllm_config.model_config.hf_config
        self.config = config

        # Resolve max_relative_positions from config
        max_relative_positions = getattr(config, "max_relative_positions", -1)
        if max_relative_positions < 1:
            position_buckets = getattr(config, "position_buckets", -1)
            max_relative_positions = (
                position_buckets
                if position_buckets > 0
                else config.max_position_embeddings
            )
        self.max_relative_positions = max_relative_positions

        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(
            config,
            max_relative_positions,
            prefix=f"{prefix}.encoder",
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            hidden_states = self.embeddings(input_ids, positions, token_type_ids)
        else:
            hidden_states = inputs_embeds
        return self.encoder(hidden_states, positions)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["pooler."])
        return loader.load_weights(weights)


# ---------------------------------------------------------------------------
# ContextPooler — matches HuggingFace's ContextPooler weight naming
# ---------------------------------------------------------------------------


class DebertaV2ContextPooler(SequencePoolingMethod):
    """Extracts the CLS token and applies a dense projection + activation.

    Matches HF's ``ContextPooler`` (weights at ``pooler.dense.*`` in HF
    checkpoints — remapped to ``context_pooler.dense.*`` in vLLM via
    ``WeightsMapper`` in the top-level model).
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        config: DebertaV2Config = model_config.hf_config
        pooler_size = getattr(config, "pooler_hidden_size", config.hidden_size)
        pooler_act = getattr(config, "pooler_hidden_act", "gelu")
        head_dtype = model_config.head_dtype

        self.dense = nn.Linear(pooler_size, pooler_size, dtype=head_dtype)
        self.act_fn = ACT2FN[pooler_act]
        self._cls_pool = CLSPool()

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        # Extract one CLS (position-0) token per sequence → [N_seqs, pooler_size]
        cls_tokens = self._cls_pool(hidden_states, pooling_metadata)
        cls_tokens = cls_tokens.to(self.dense.weight.dtype)
        return self.act_fn(self.dense(cls_tokens))


# ---------------------------------------------------------------------------
# Top-level sequence-classification model
# ---------------------------------------------------------------------------


@default_pooling_type(seq_pooling_type="CLS")
class DebertaV2ForSequenceClassification(nn.Module, SupportsCrossEncoding):
    """DeBERTa-v2/v3 cross-encoder reranker for vLLM.

    Supports any ``DebertaV2ForSequenceClassification`` checkpoint with
    ``num_labels=1`` (e.g. ``cross-encoder/nli-deberta-v3-small``,
    ``Capreolus/deberta-v3-base-msmarco``, etc.).

    HuggingFace weight layout (simplified):
      deberta.embeddings.*
      deberta.encoder.rel_embeddings.*
      deberta.encoder.LayerNorm.*
      deberta.encoder.layer.N.attention.self.{query,key,value}_proj.*
      deberta.encoder.layer.N.attention.output.dense.*
      deberta.encoder.layer.N.{intermediate,output}.dense.*
      pooler.dense.*          ← remapped to context_pooler.dense.*
      classifier.{weight,bias}
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: DebertaV2Config = vllm_config.model_config.hf_config

        self.deberta = DebertaV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "deberta"),
        )

        pooler_size = getattr(config, "pooler_hidden_size", config.hidden_size)
        self.classifier = nn.Linear(
            pooler_size,
            config.num_labels,
            dtype=vllm_config.model_config.head_dtype,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        # HF name: "pooler" — we use "context_pooler" to avoid clash with
        # self.pooler (DispatchPooler). WeightsMapper handles the rename.
        self.context_pooler = DebertaV2ContextPooler(vllm_config.model_config)
        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config,
            pooling=self.context_pooler,
            classifier=self.classifier,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.deberta.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Remap HF "pooler.*" → "context_pooler.*" for the ContextPooler dense
        mapper = WeightsMapper(orig_to_new_prefix={"pooler.": "context_pooler."})
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.deberta(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
            token_type_ids=token_type_ids,
        )
