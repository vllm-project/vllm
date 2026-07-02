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

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


@triton.jit
def _compute_prefix_survival_probabilities_kernel(
    confidence_logits_ptr,
    survival_probs_ptr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    survival_prob = tl.full((), 1.0, tl.float32)
    for step in tl.static_range(0, NUM_SPECULATIVE_STEPS):
        confidence_logit = tl.load(
            confidence_logits_ptr + req_idx * NUM_SPECULATIVE_STEPS + step
        ).to(tl.float32)
        confidence_prob = 1.0 / (1.0 + tl.exp(-confidence_logit))
        survival_prob *= confidence_prob
        tl.store(
            survival_probs_ptr + req_idx * NUM_SPECULATIVE_STEPS + step,
            survival_prob,
        )


@triton.jit
def _allocate_draft_token_capacity_kernel(
    survival_probs_ptr,
    capacity_ptr,
    sps_profile_ptr,
    num_reqs,
    min_survival_probability,
    REQ_BLOCK: tl.constexpr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
    MAX_ADMISSIONS: tl.constexpr,
    HAS_SPS_PROFILE: tl.constexpr,
    SPS_PROFILE_LEN: tl.constexpr,
):
    offsets = tl.arange(0, REQ_BLOCK)
    active = offsets < num_reqs
    lengths = tl.full((REQ_BLOCK,), 0, tl.int32)
    best_lengths = tl.full((REQ_BLOCK,), 0, tl.int32)

    batch_size = num_reqs
    expected_tokens = num_reqs.to(tl.float32)
    if HAS_SPS_PROFILE:
        profile_idx = tl.minimum(batch_size, SPS_PROFILE_LEN - 1)
        sps = tl.load(sps_profile_ptr + profile_idx).to(tl.float32)
    else:
        sps = 1.0
    best_throughput = expected_tokens * sps

    for _ in tl.static_range(0, MAX_ADMISSIONS):
        has_next = active & (lengths < NUM_SPECULATIVE_STEPS)
        next_scores = tl.load(
            survival_probs_ptr + offsets * NUM_SPECULATIVE_STEPS + lengths,
            mask=has_next,
            other=-1.0,
        )
        next_scores = tl.where(
            next_scores >= min_survival_probability,
            next_scores,
            -1.0,
        )
        best_score, best_idx = tl.max(next_scores, axis=0, return_indices=True)
        admit = best_score >= 0.0
        is_best_req = offsets == best_idx
        lengths += tl.where(admit & is_best_req, 1, 0)

        batch_size += tl.where(admit, 1, 0)
        expected_tokens += tl.where(admit, best_score, 0.0)
        if HAS_SPS_PROFILE:
            profile_idx = tl.minimum(batch_size, SPS_PROFILE_LEN - 1)
            sps = tl.load(sps_profile_ptr + profile_idx).to(tl.float32)
        else:
            sps = 1.0
        throughput = expected_tokens * sps
        better = admit & (throughput > best_throughput)
        best_throughput = tl.where(better, throughput, best_throughput)
        best_lengths = tl.where(better, lengths, best_lengths)

    tl.store(capacity_ptr + offsets, best_lengths, mask=active)


def compute_draft_token_capacity_from_confidence(
    confidence_logits: torch.Tensor,
    draft_token_capacity: torch.Tensor,
    min_survival_probability: float,
    num_reqs: int,
    num_speculative_steps: int,
    survival_probs: torch.Tensor | None = None,
    sps_profile: torch.Tensor | None = None,
) -> None:
    if num_reqs == 0 or num_speculative_steps == 0:
        return
    if survival_probs is None:
        survival_probs = torch.empty_like(confidence_logits)
    _compute_prefix_survival_probabilities_kernel[(num_reqs,)](
        confidence_logits,
        survival_probs,
        NUM_SPECULATIVE_STEPS=num_speculative_steps,
    )
    req_block = triton.next_power_of_2(max(num_reqs, 1))
    has_sps_profile = sps_profile is not None and sps_profile.numel() > 0
    if sps_profile is None:
        sps_profile = survival_probs
    _allocate_draft_token_capacity_kernel[(1,)](
        survival_probs,
        draft_token_capacity,
        sps_profile,
        num_reqs,
        min_survival_probability,
        REQ_BLOCK=req_block,
        NUM_SPECULATIVE_STEPS=num_speculative_steps,
        MAX_ADMISSIONS=num_reqs * num_speculative_steps,
        HAS_SPS_PROFILE=has_sps_profile,
        SPS_PROFILE_LEN=sps_profile.numel(),
    )


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Anchor-as-first (N slots) unless the checkpoint uses the 1+N fill-in
        # block, where the anchor is a separate bonus token.
        self.sample_from_anchor = not getattr(
            self.draft_model_config.hf_config, "dspark_bonus_anchor", False
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

        self.dflash_causal = False

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

        self.draft_token_confidence_logits = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.float32,
            device=device,
        )
        self.draft_token_survival_probs = torch.empty_like(
            self.draft_token_confidence_logits
        )
        self.draft_token_capacity = torch.full(
            (self.max_num_reqs,),
            self.num_speculative_steps,
            dtype=torch.int32,
            device=device,
        )
        self.min_survival_probability = getattr(
            self.speculative_config, "dspark_confidence_threshold", 0.5
        )
        sps_profile = getattr(self.speculative_config, "dspark_sps_profile", None)
        self.sps_profile = (
            None
            if sps_profile is None
            else torch.tensor(sps_profile, dtype=torch.float32, device=device)
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
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
        return model

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        sample_hidden = sample_hidden.view(num_reqs, n_spec, -1)
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(
            sample_hidden.reshape(num_sample, -1)
        )
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_logits = self.draft_token_confidence_logits[:num_reqs]
        min_survival_probability = self.min_survival_probability
        use_confidence_capacity = (
            min_survival_probability is not None or self.sps_profile is not None
        )

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if use_confidence_capacity:
                confidence_i = self.model.compute_confidence(
                    sample_hidden[:, i], markov_embed
                )
                if confidence_i is None:
                    use_confidence_capacity = False
                else:
                    confidence_logits[:, i] = confidence_i
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            if self.draft_logits is not None:
                # Probabilistic: sample in target vocab (a reduced draft vocab is
                # scattered into its target columns; full vocab is already there).
                if self._d2t_scatter_index is not None:
                    assert self._draft_scatter_buf is not None
                    buf = self._draft_scatter_buf[:num_reqs]
                    buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                    logits_i = buf
                # sample_pos is the predicted token's position Q; the target
                # verifies it with the predecessor's Gumbel key (Q-1). Pass Q-1.
                draft_sampled_i = gumbel_sample(
                    logits_i,
                    idx_map[:, i],
                    self.temperature,
                    self.seeds,
                    sample_pos[:, i] - 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self._step_cols[i],
                    use_fp64=self.use_fp64_gumbel,
                )
            else:
                draft_sampled_i = self.model.map_draft_to_target(
                    logits_i.argmax(dim=-1)
                )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if use_confidence_capacity:
            allocator_min_survival_probability = (
                0.0
                if self.sps_profile is not None or min_survival_probability is None
                else min_survival_probability
            )
            compute_draft_token_capacity_from_confidence(
                self.draft_token_confidence_logits,
                self.draft_token_capacity,
                allocator_min_survival_probability,
                num_reqs,
                self.num_speculative_steps,
                self.draft_token_survival_probs,
                self.sps_profile,
            )
        else:
            self.draft_token_capacity[:num_reqs].fill_(self.num_speculative_steps)

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
