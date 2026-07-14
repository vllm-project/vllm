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
import triton
import triton.language as tl

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


@triton.jit(do_not_specialize=["step_i"])
def _draft_penalties_kernel(
    logits_ptr,
    logits_stride,
    rows_ptr,  # [num_reqs] req-state indices
    draft_tokens_ptr,  # [max_num_reqs, n_spec] tokens sampled so far this block
    draft_tokens_stride,
    step_i,  # number of in-block tokens sampled before this step
    repetition_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Draft-side mirror of the target's _penalties_kernel (rep penalty only).

    Penalized set = prompt bin mask | output bin counts | draft tokens
    t_0..t_{step_i-1} sampled earlier in this block — identical to what the
    verify kernel accumulates for the row checking t_{step_i}.
    """
    req_idx = tl.program_id(0).to(tl.int64)
    state_idx = tl.load(rows_ptr + req_idx).to(tl.int64)
    rep = tl.load(repetition_penalty_ptr + state_idx)
    if rep == 1.0:
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + req_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    counts = tl.load(
        output_bin_counts_ptr + state_idx * output_bin_counts_stride + block,
        mask=mask,
        other=0,
    )
    pen_mask = counts > 0

    packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed = tl.load(
        prompt_bin_mask_ptr + state_idx * prompt_bin_mask_stride + packed_block,
        mask=packed_block < tl.cdiv(vocab_size, 32),
        other=0,
    )
    prompt_bits = (packed[:, None] >> (tl.arange(0, 32)[None, :])) & 1
    pen_mask |= prompt_bits.to(tl.int1).reshape(BLOCK_SIZE)

    for j in tl.range(step_i):
        t = tl.load(draft_tokens_ptr + req_idx * draft_tokens_stride + j)
        pen_mask |= block == t

    scale = tl.where(pen_mask, rep, 1.0)
    logits *= tl.where(logits > 0, 1.0 / scale, scale)
    tl.store(logits_ptr + req_idx * logits_stride + block, logits, mask=mask)


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

        # PATCH (local): draft-side repetition penalty. When enabled (env
        # VLLM_DRAFT_REP_PENALTY=1 + probabilistic drafting), the drafter
        # mirrors the target's repetition-penalty logit transform so the
        # rejection test compares aligned p/q distributions. Without this,
        # every context-repeated token has q suppressed but p not, which
        # costs speculation acceptance heavily on repetition-rich outputs
        # (e.g. TTS speech tokens). Set via set_penalties_state().
        self._penalties_state = None
        self._sampling_states = None
        self._topk_arange: torch.Tensor | None = None

    # Static cap for the draft-side top-k/top-p mirror: requests with
    # top_k > _TOPK_CAP (or disabled, stored as vocab_size) skip truncation.
    _TOPK_CAP = 64

    def set_penalties_state(self, penalties_state, sampling_states=None) -> None:
        import os

        if self.draft_logits is None:
            # Greedy drafting: rejection uses exact match, p/q alignment moot.
            return
        if os.environ.get("VLLM_DRAFT_REP_PENALTY", "0") == "1":
            self._penalties_state = penalties_state
        if (
            sampling_states is not None
            and os.environ.get("VLLM_DRAFT_TOPK_TOPP", "0") == "1"
        ):
            self._sampling_states = sampling_states
            self._topk_arange = torch.arange(
                self._TOPK_CAP, dtype=torch.int64, device=self.device
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
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        # PATCH (local): per-request state for the draft-side sampling-param
        # mirrors (repetition penalty via _draft_penalties_kernel; top-k/top-p
        # truncation). rows/k/p/temp are block-constant; the kernel adds the
        # in-block draft tokens t_0..t_{i-1} per step from self.draft_tokens.
        pen_rows = None
        if self._penalties_state is not None and self.draft_logits is not None:
            pen_rows = idx_map[:, 0].contiguous()
        tk_state = None
        if self._sampling_states is not None and self.draft_logits is not None:
            rows = idx_map[:, 0].to(torch.long)
            k_req = self._sampling_states.top_k.gpu[rows]
            p_req = self._sampling_states.top_p.gpu[rows].to(torch.float32)
            temp = self.temperature[rows].to(torch.float32).clamp_min(1e-5)
            k_enabled = (k_req <= self._TOPK_CAP).unsqueeze(1)
            k_eff = k_req.clamp(min=1, max=self._TOPK_CAP).to(torch.long)
            within_k = self._topk_arange.unsqueeze(0) < k_eff.unsqueeze(1)
            tk_state = (k_eff, k_enabled, within_k, p_req, temp)

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
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
                # PATCH (local): mirror the target's repetition-penalty logit
                # transform on the draft logits (fused triton kernel, mutates
                # logits_i in place; per-row early exit when rep == 1.0).
                if pen_rows is not None:
                    ps = self._penalties_state
                    vocab = logits_i.shape[-1]
                    blk = 8192
                    _draft_penalties_kernel[(num_reqs, triton.cdiv(vocab, blk))](
                        logits_i,
                        logits_i.stride(0),
                        pen_rows,
                        self.draft_tokens,
                        self.draft_tokens.stride(0),
                        i,
                        ps.repetition_penalty.gpu,
                        ps.prompt_bin_mask,
                        ps.prompt_bin_mask.stride(0),
                        ps.output_bin_counts,
                        ps.output_bin_counts.stride(0),
                        vocab,
                        BLOCK_SIZE=blk,
                    )
                # PATCH (local): mirror the target's top-k/top-p truncation so
                # the draft never proposes tokens the target has zeroed out.
                # Order matches the target sampler: penalties -> temperature
                # -> top-k/top-p (top-p is computed on temperature-scaled
                # logits within the top-k set).
                if tk_state is not None:
                    k_eff, k_enabled, within_k, p_req, temp = tk_state
                    vals = torch.topk(
                        logits_i.float(), self._TOPK_CAP, dim=-1
                    ).values
                    kth = vals.gather(1, (k_eff - 1).unsqueeze(1))
                    scaled = torch.where(
                        within_k,
                        vals / temp.unsqueeze(1),
                        torch.full((), float("-inf"), device=vals.device),
                    )
                    probs = torch.softmax(scaled, dim=-1)
                    keep = (probs.cumsum(-1) - probs) < p_req.unsqueeze(1)
                    last = (keep.sum(-1).clamp(min=1) - 1).unsqueeze(1)
                    thresh = torch.maximum(kth, vals.gather(1, last))
                    thresh = torch.where(
                        k_enabled,
                        thresh,
                        torch.full((), float("-inf"), device=vals.device),
                    )
                    logits_i = torch.where(
                        logits_i >= thresh,
                        logits_i,
                        torch.full((), float("-inf"), device=vals.device),
                    )
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
