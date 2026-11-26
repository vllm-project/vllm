# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DNeo parallel drafter (arXiv:2611.21925) on the DFlash runtime.

DNeo is a DFlash-family block drafter. Relative to DFlash it:

* encodes the target's context with MoFA (``fc``): each captured target layer's
  hidden state is normalized with that layer's own (frozen) attention-input
  RMSNorm gain, and the normalized states are mixed with one softmax weight per
  layer. Nothing follows it: no ``hidden_norm``;
* uses DFlash-2's 2-tap dynamic convolutions around attention and MLP;
* attends prefix-only: a draft query sees the context and its block's anchor
  (the bonus token), not the other draft queries. With non-causal attention this
  is the context plus the first query key, so the speculator stops the draft
  attention's ``seq_lens`` one past the context;
* corrects each draft position's logits with a causal head conditioned on the
  previously drafted token, sampled left to right (see ``DNeoSpeculator``):
  ``(a + b) * base + W2(a * E[prev] + b * W_lat(C[prev] * P(h)))`` with
  per-position scales ``a`` and ``b``.

Checkpoints are in the speculators format (``speculators_model_type: dneo``).
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.v1.worker.gpu.spec_decode.dneo.config import DNeoSpeculatorConfig

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .qwen3_dflash2 import DFlash2Qwen3DecoderLayer
from .utils import maybe_prefix


class DNeoFeatureEncoder(nn.Module):
    """MoFA over the target's normalized attention inputs.

    ``[num_tokens, num_layers * hidden]`` concatenated target hidden states (the
    residual stream at the captured layer ids) to ``[num_tokens, hidden]``.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        eps: float,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_layers, self.hidden_size, self.eps = num_layers, hidden_size, eps
        # read by DFlashQwen3ForCausalLM.combine_hidden_states
        self.input_size = num_layers * hidden_size
        self.params_dtype = params_dtype
        self.layer_logits = nn.Parameter(
            torch.zeros(num_layers, dtype=torch.float32), requires_grad=False
        )
        self.input_norm_weight = nn.Parameter(
            torch.ones(num_layers, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.view(-1, self.num_layers, self.hidden_size)
        xf = x.float()
        # each layer's own RMSNorm, rounded to the model dtype as the target does
        normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(
            self.params_dtype
        ) * self.input_norm_weight
        weights = self.layer_logits.float().softmax(-1).to(self.params_dtype)
        return torch.einsum("l,nlh->nh", weights, normed)


class DNeoCausalHead(nn.Module):
    """DNeo's folded causal head (see the module docstring)."""

    def __init__(
        self,
        *,
        vocab_size: int,
        draft_vocab_size: int,
        hidden_size: int,
        rank: int,
        folded_rank: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Embeddings are indexed by the previous token, in the target vocabulary.
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size, folded_rank, prefix=maybe_prefix(prefix, "markov_w1")
        )
        self.predecessor_codebook = VocabParallelEmbedding(
            vocab_size, rank, prefix=maybe_prefix(prefix, "predecessor_codebook")
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )
        self.latent_projection = ReplicatedLinear(
            rank,
            folded_rank,
            bias=False,
            params_dtype=params_dtype,
            prefix=maybe_prefix(prefix, "latent_projection"),
            return_bias=False,
        )
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size, folded_rank, prefix=maybe_prefix(prefix, "markov_w2")
        )
        self.markov_scale = nn.Parameter(
            torch.ones(block_size, dtype=params_dtype), requires_grad=False
        )
        self.context_scale = nn.Parameter(
            torch.zeros(block_size, dtype=params_dtype), requires_grad=False
        )

    def step_logits(
        self,
        base_logits: torch.Tensor,
        projected: torch.Tensor,
        prev_token_ids: torch.Tensor,
        slot: int,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        """Corrected logits of draft slot ``slot`` for one row per request.

        ``base_logits`` are the backbone's draft-vocab logits ``[R, V]``,
        ``projected`` is ``hidden_projection`` of the slot's hidden ``[R, r]`` and
        ``prev_token_ids`` are the previous slot's tokens ``[R]``.
        """
        a = self.markov_scale[slot]
        b = self.context_scale[slot]
        interaction = self.predecessor_codebook(prev_token_ids) * projected
        latent = a * self.markov_w1(prev_token_ids) + b * self.latent_projection(
            interaction
        )
        correction = logits_processor(self.markov_w2, latent)
        return ((a + b) * base_logits + correction).to(base_logits.dtype)


class _NoHiddenNorm(nn.Module):
    """DNeo's encoder output is used as is. ``weight`` is a plain attribute (no
    parameter to load) for the DFlash code that aliases it."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.empty(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DNeoQwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

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
        dneo = config.dneo_config
        dtype = vllm_config.model_config.dtype
        self.fc = DNeoFeatureEncoder(
            len(config.eagle_aux_hidden_state_layer_ids),
            config.hidden_size,
            eps=config.rms_norm_eps,
            params_dtype=dtype,
        )
        self.hidden_norm = _NoHiddenNorm()
        self.causal_head = DNeoCausalHead(
            vocab_size=config.vocab_size,
            draft_vocab_size=getattr(config, "draft_vocab_size", None)
            or config.vocab_size,
            hidden_size=config.hidden_size,
            rank=int(dneo["causal_head_rank"]),
            folded_rank=int(dneo["causal_head_folded_rank"]),
            block_size=int(config.block_size),
            params_dtype=dtype,
            prefix=maybe_prefix(prefix, "causal_head"),
        )
        # DSparkSpeculator reads it; DNeo has none
        self.confidence_head = None

    def _normalize_context_states(self, context_states: torch.Tensor) -> torch.Tensor:
        return context_states

    def context_kv_modules(self) -> list[nn.Module]:
        return [self.fc, *(layer.self_attn for layer in self.layers)]


class DNeoQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    # The speculator this is a draft model of (see SpeculativeDraftModel).
    speculator_config_class = DNeoSpeculatorConfig
    model_cls = DNeoQwen3Model

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Draft-vocab logits of the backbone, before the causal head."""
        return self.logits_processor(self.lm_head, hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def project_head_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.causal_head.hidden_projection(hidden_states)

    def head_step_logits(
        self,
        base_logits: torch.Tensor,
        projected: torch.Tensor,
        prev_token_ids: torch.Tensor,
        slot: int,
    ) -> torch.Tensor:
        return self.model.causal_head.step_logits(
            base_logits, projected, prev_token_ids, slot, self.logits_processor
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # speculators-format checkpoints may carry verifier-owned tensors that the
        # drafter shares with the target at load time
        def keep(item: tuple[str, torch.Tensor]) -> bool:
            return not item[0].startswith(("verifier_", "embed_tokens", "lm_head"))

        return super().load_weights(filter(keep, weights))
