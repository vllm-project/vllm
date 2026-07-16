# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 DSpark draft model for semi-autoregressive drafting.

DSpark drafts a whole block in one parallel pass (DFlash-style: context-KV
precompute + a non-causal query-block forward) and then injects intra-block
dependency with a lightweight sequential Markov head.

The parallel backbone is a standard Qwen3 decoder stack reused from the
DFlash Qwen3 draft (see qwen3_dflash.py). DSpark adds:
  * ``markov_head``: low-rank V x r / r x V transition bias added to the base
    logits, sampled left-to-right by the speculator (the sequential stage).
  * ``confidence_head``: per-position acceptance-probability estimate.

DSparkMarkovHead and DSparkConfidenceHead are shared with the DSV4-style
DSpark model.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)


class DSparkMarkovHead(nn.Module):
    """Sequential transition-bias head (low-rank V x r, r x V).

    ``markov_w1[token]`` embeds the previously sampled token (target vocab,
    ``vocab_size``); ``markov_w2`` projects it to a draft-vocab bias
    (``draft_vocab_size``) added to the base draft logits. The two sizes
    coincide for full-vocab drafts.
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
    ) -> None:
        super().__init__()
        # TODO(ben): profile for which (if any) it makes sense to replicate or TP-shard
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w1")
        )
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w2")
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """r-dim Markov embedding of ``token_ids`` ([B] -> [B, r])."""
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:
        """Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V])."""
        return logits_processor(self.markov_w2, markov_embed)


class DSparkConfidenceHead(nn.Module):
    """Per-position acceptance-probability head."""

    def __init__(
        self,
        input_dim: int,
        prefix: str,
        *,
        bias: bool = False,
        params_dtype: torch.dtype,
        with_markov: bool = True,
    ) -> None:
        super().__init__()
        self.with_markov = with_markov
        self.proj = ReplicatedLinear(
            input_dim,
            1,
            bias=bias,
            return_bias=False,
            params_dtype=params_dtype,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden, markov_embed], dim=-1) if self.with_markov else hidden
        return self.proj(x).float().squeeze(-1)


class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash Qwen3 backbone + DSpark Markov and confidence heads."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        config = self.config
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        assert vllm_config.speculative_config is not None
        if vllm_config.speculative_config.adaptive_verification:
            with_markov = getattr(config, "confidence_head_with_markov", True)
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + (config.markov_rank if with_markov else 0),
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                params_dtype=vllm_config.model_config.dtype,
                with_markov=with_markov,
            )


class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
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
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Draft-vocab logits without the d2t scatter: the speculator adds the
        # Markov bias in draft space, then remaps via map_draft_to_target.
        return self.logits_processor(self.lm_head, hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        # Map draft-vocab ids to target ids (identity for full-vocab drafts).
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor | None:
        if self.model.confidence_head is None:
            return None
        return self.model.confidence_head(head_hidden, markov_embed)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        includes_draft_id_mapping = False
        includes_confidence_head = False
        for name, loaded_weight in weights:
            if "confidence_head" in name and self.model.confidence_head is None:
                continue
            # t2d is training-only; the draft remaps via d2t at sampling time.
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            if "confidence_head" in name:
                includes_confidence_head = True
            model_weights[name] = loaded_weight
            # Sets has_own_embed_tokens / has_own_lm_head so load_dspark_model
            # knows whether to keep these or alias the target's.
            process_eagle_weight(self, name)

        # mask_embedding is an unused placeholder param; DSpark masks via the vocab row.
        # embed_tokens / lm_head are optional; when omitted they are shared from
        # the target by load_dspark_model, so skip the unloaded params here.
        skip_substrs = ["mask_embedding"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if self.model.confidence_head is None:
            skip_substrs.append("confidence_head")
        elif not includes_confidence_head:
            raise ValueError(
                "adaptive_verification requires confidence-head weights, but none "
                "were found in the Qwen3 DSpark checkpoint."
            )
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
