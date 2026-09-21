# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.
  * Candidate pruning (``markov_topk``, opt-in): that bias is evaluated only
    for a small candidate set per position, shrinking the projection from
    ``[B, rank] @ [rank, V]`` to ``[B, k, rank] @ [B, rank, 1]`` and keeping
    selection/sampling inside the candidate set. The candidates are the union of
    the ``markov_topk`` highest base logits and the ``markov_bias_topk`` bigram
    top-m of the previously sampled token (precomputed once from the trained
    weights): the backbone sees a mask token in every draft slot, so the base
    logits alone keep missing what the Markov head actually predicts. Unset (or
    ``markov_topk=0``) keeps the original full-vocab Markov projection; both
    paths reuse the trained weights unchanged.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

from typing import Any

import torch
from flashinfer import top_k as _flashinfer_topk

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.speculative import resolve_markov_bias_topk, resolve_markov_topk
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.topk_markov import (
    cache_markov_candidates,
    compute_markov_bias_top_ids,
    markov_walk_topk,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None

        # Candidate-pruned vanilla Markov head. Each draft position keeps only
        # the top-k base-logit candidates, and one fused kernel walks the block:
        # W1[prev] embedding, gathered W2 rows, the [k, r] @ [r] correction,
        # selection/sampling inside the candidate set and chaining the winner
        # into the next position. 0 keeps the full-vocab projection below.
        self.markov_topk = resolve_markov_topk(self.speculative_config)
        self.markov_bias_topk = resolve_markov_bias_topk(self.speculative_config)
        self._markov_walk_enabled = False
        self._markov_walk_w1: torch.Tensor | None = None
        self._markov_walk_w2: torch.Tensor | None = None
        self._markov_walk_scale = 1.0
        self._markov_walk_d2t: torch.Tensor | None = None
        self._base_cand_values: torch.Tensor | None = None
        self._base_cand_ids: torch.Tensor | None = None
        self._union_cand_ids: torch.Tensor | None = None
        self._markov_static_ids: torch.Tensor | None = None
        self._markov_static_biases: torch.Tensor | None = None
        # Probabilistic drafting / adaptive verification only.
        self._realized_scores: torch.Tensor | None = None
        self._cached_candidate_ids: torch.Tensor | None = None
        self._markov_walk_embeds: torch.Tensor | None = None

        self.use_confidence_head: bool = False

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        self._init_markov_walk(model)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        # The pruned walk maps candidate ids to target ids itself, so it needs
        # neither.
        if (
            not self._markov_walk_enabled
            and self.draft_logits is not None
            and model.draft_id_to_target_id is not None
        ):
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        self.use_confidence_head = (
            self.enable_adaptive_verification
            and model.model.confidence_head is not None
        )
        if self.use_confidence_head:
            # The acceptance estimator is not needed when a trained confidence head
            # is available.
            self.use_acceptance_estimator = False
        return model

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # With candidate pruning the cached proposal distribution is written
        # incrementally -- only the k candidate columns move -- so every other
        # column has to hold the "impossible" -inf fill (zero draft probability
        # outside the candidate set). fp32 keeps the published scores identical
        # to the ones the walk sampled from.
        if resolve_markov_topk(vllm_config.speculative_config) > 0:
            return torch.float32, -float("inf")
        return super().draft_logits_spec(vllm_config)

    def _init_markov_walk(self, model: torch.nn.Module) -> None:
        """Resolve the pruned Markov walk and allocate its persistent buffers.

        Buffers are allocated here (and reused every step) so that CUDA graph
        capture sees stable addresses. Falls back to the full-vocab head when the
        checkpoint cannot be read by the walk kernel -- e.g. a quantized or
        tensor-sharded ``markov_w2``.
        """
        self._markov_walk_enabled = False
        topk = self.markov_topk
        if topk <= 0:
            return
        supports_walk = getattr(model, "supports_markov_candidate_walk", None)
        if supports_walk is None or not supports_walk():
            logger.warning_once(
                "markov_topk=%d was requested but this DSpark checkpoint's Markov "
                "head cannot be read by the candidate-pruned walk (quantized or "
                "tensor-sharded markov_w2); falling back to the full-vocab Markov "
                "projection.",
                topk,
            )
            self.markov_topk = 0
            return

        w1, w2, scale = model.markov_walk_inputs()
        rank = int(w1.shape[1])
        num_steps = self.num_speculative_steps
        device = self.device
        logits_dtype = getattr(model.logits_processor, "head_dtype", None) or self.dtype
        d2t = getattr(model, "draft_id_to_target_id", None)

        self._markov_walk_enabled = True
        self._markov_walk_w1 = w1
        self._markov_walk_w2 = w2
        self._markov_walk_scale = float(scale)
        self._markov_walk_d2t = d2t
        bias_topk = min(self.markov_bias_topk, int(w2.shape[0]))
        self.markov_bias_topk = bias_topk
        union_k = topk + bias_topk
        base_shape = (self.max_num_reqs, num_steps, topk)
        self._base_cand_values = torch.empty(
            base_shape, dtype=logits_dtype, device=device
        )
        self._base_cand_ids = torch.empty(base_shape, dtype=torch.int64, device=device)
        if bias_topk > 0:
            # Bigram side of the candidate union: the top-m rows of the dense
            # Markov projection for every possible `prev`, precomputed once from
            # the trained weights (and disk-cached) so no step projects [r, V].
            # The fp32 bias values let the walk kernel skip W1[prev]·W2[cand]
            # for static candidates, eliminating scattered W2 row reads.
            self._markov_static_ids, self._markov_static_biases = (
                compute_markov_bias_top_ids(w1, w2, bias_topk, self._markov_walk_scale)
            )
        shape = (self.max_num_reqs, num_steps, union_k)
        if self.draft_logits is not None:
            # Pre-temperature candidate scores + the union ids currently living
            # in the draft-logit cache (reset before each rewrite).
            self._realized_scores = torch.empty(
                shape, dtype=torch.float32, device=device
            )
            self._cached_candidate_ids = torch.zeros(
                shape, dtype=torch.int64, device=device
            )
            if bias_topk > 0:
                self._union_cand_ids = torch.empty(
                    shape, dtype=torch.int64, device=device
                )
        if self.enable_adaptive_verification:
            self._markov_walk_embeds = torch.empty(
                (self.max_num_reqs, num_steps, rank), dtype=w1.dtype, device=device
            )
        logger.info_once(
            "DSpark Markov head candidate pruning: markov_topk=%d + "
            "markov_bias_topk=%d = %d candidates per position (draft vocab %d, "
            "rank %d). Set markov_topk=0 for the full-vocab Markov projection.",
            topk,
            bias_topk,
            union_k,
            int(w2.shape[0]),
            rank,
            scope="process",
        )

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        self._maybe_predict_acceptance(logits, idx_map, self._step_cols[step])
        if self.draft_logits is None:
            draft_ids = logits.argmax(dim=-1)
            return self.model.map_draft_to_target(draft_ids)

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sampled = gumbel_sample(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )
        if self.draft_watermarker is not None:
            sampled = self.draft_watermarker.sample(
                logits,
                sampled,
                idx_map,
                self.temperature,
            )
        return sampled

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._markov_walk_enabled:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return

        # Full-vocab sequential Markov sampling over the backbone's output
        # hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Sequential Markov drafting restricted to top-k base-logit candidates.

        The candidates are selected once for every draft position (a single
        ``topk`` over the base logits). The per-position Markov correction, the
        selection/sampling and the chaining into the next position then run
        inside one fused kernel over ``[num_reqs, k]`` tiles, so no full-vocab
        tensor is materialized per step: the top-k, the ``W2`` row gather and
        the candidate-space sampling are the only extra work over the dense path.

        With probabilistic drafting the realized (pre-temperature) candidate
        scores are published to the draft-logit cache, whose other columns keep
        the ``-inf`` fill -- i.e. the verifier reads exactly the truncated,
        renormalized distribution the drafter sampled from, and its full-vocab
        rejection correction needs no change.
        """
        assert self._markov_walk_enabled
        assert self._base_cand_values is not None and self._base_cand_ids is not None
        assert self._markov_walk_w1 is not None and self._markov_walk_w2 is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; candidate ids are remapped to target vocab by the
        # walk kernel itself.
        base_logits = self.model.compute_draft_logits(sample_hidden).view(
            num_reqs, n_spec, -1
        )
        base_values = self._base_cand_values[:num_reqs]
        base_ids = self._base_cand_ids[:num_reqs]
        # Candidate order is irrelevant (the walk re-argmaxes inside the union).
        # flashinfer top_k is ~6x faster than torch.topk for large vocabularies
        # but requires 2D input; reshape is zero-copy.
        flat_logits = base_logits.view(-1, base_logits.shape[-1])
        fi_vals, fi_ids = _flashinfer_topk(flat_logits, self.markov_topk)
        base_values.copy_(fi_vals.view(num_reqs, n_spec, -1))
        base_ids.copy_(fi_ids.view(num_reqs, n_spec, -1))
        static_ids = self._markov_static_ids
        union_ids = (
            self._union_cand_ids[:num_reqs]
            if self._union_cand_ids is not None
            else base_ids
        )

        markov_walk_topk(
            num_reqs=num_reqs,
            cand_values=base_values,
            cand_ids=base_ids,
            static_ids=static_ids,
            static_biases=self._markov_static_biases,
            base_logits=base_logits if static_ids is not None else None,
            union_ids=(
                self._union_cand_ids[:num_reqs]
                if self._union_cand_ids is not None
                else None
            ),
            w1=self._markov_walk_w1,
            w2=self._markov_walk_w2,
            scale=self._markov_walk_scale,
            draft_tokens=self.draft_tokens,
            input_ids=self.input_buffers.input_ids,
            anchor_indices=self._anchor_idx,
            sample_pos=self.sample_pos[:num_sample],
            sample_idx_mapping=self.sample_idx_mapping[:num_sample],
            temperature=self.temperature,
            seeds=self.seeds,
            d2t=self._markov_walk_d2t,
            realized_scores=(
                self._realized_scores[:num_reqs]
                if self._realized_scores is not None
                else None
            ),
            markov_embeds=(
                self._markov_walk_embeds[:num_reqs]
                if self._markov_walk_embeds is not None
                else None
            ),
            probabilistic=self.draft_logits is not None,
            use_fp64=self.use_fp64_gumbel,
        )

        if self.draft_logits is not None and self._cached_candidate_ids is not None:
            assert self._realized_scores is not None
            cache_markov_candidates(
                draft_logits=self.draft_logits,
                cached_ids=self._cached_candidate_ids,
                cand_ids=union_ids,
                realized_scores=self._realized_scores[:num_reqs],
                sample_idx_mapping=self.sample_idx_mapping[:num_sample],
                d2t=self._markov_walk_d2t,
            )

        if self.use_confidence_head:
            assert self._markov_walk_embeds is not None
            confidence = self.model.compute_confidence(
                sample_hidden, self._markov_walk_embeds[:num_reqs].flatten(0, 1)
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
