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

import time
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.scheduler import DSparkScheduler
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


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

        # Confidence head + hardware-aware scheduler (opt-in): adapts a
        # per-step uniform verify length L; DSparkScheduler owns all policy
        # state. Read at init so the confidence ops enter the draft graph.
        spec_cfg = vllm_config.speculative_config
        assert spec_cfg is not None
        self._sched = spec_cfg.dspark_scheduler
        self.dspark_scheduler_enabled = self._sched
        if self._sched:
            gamma = self.num_speculative_steps
            self._survival = torch.zeros(
                self.max_num_reqs, gamma, dtype=torch.float32, device=device
            )
            self._scheduler = DSparkScheduler(
                spec_cfg, gamma, self.max_num_reqs, device
            )
            # Positions beyond valid_draft_len[r] are pads the rejection
            # sampler force-rejects. Created only with a confidence threshold;
            # otherwise the runner passes None and the mask path costs nothing.
            self._tau = spec_cfg.dspark_confidence_threshold
            self._perreq = spec_cfg.dspark_per_request
            if self._tau > 0.0:
                self.valid_draft_len = torch.full(
                    (self.max_num_reqs,), gamma, dtype=torch.int32, device=device
                )
            # Dynamic SD: the runner writes the engine-scheduled next-step width
            # here (0 -> skip drafting); the engine decision is authoritative.
            self.next_k_hint: int | None = None
            # Ask the runner to capture FULL decode graphs at every trimmed width
            # 1+L (L in 0..gamma-1); 1+gamma is already the decode_query_len graph.
            self.extra_uniform_decode_lens = list(range(1, gamma + 1))
            # perreq_len is req-state-indexed; _perreq_batch_len is the batch-ordered
            # view handed to the draft-tokens handler in the same step.
            self.perreq_len = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=device
            )
            self._perreq_batch_len: torch.Tensor | None = None

    def set_cost_table(self, r_grid: list[int], times_by_l: list[list[float]]) -> None:
        """Install the shape-aware cost table T(R, L) on the scheduler policy."""
        self._scheduler.set_cost_table(r_grid, times_by_l)

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        if self._sched and getattr(model, "compute_confidence", None) is None:
            raise ValueError(
                "dspark_scheduler requires a draft model with a confidence "
                f"head; {type(model).__name__} does not implement "
                "compute_confidence."
            )
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

    def _schedule_draft_len(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        dummy_run: bool,
    ) -> int:
        # Pick this step's uniform verify length L in [0, gamma]: L < gamma
        # trims the verify tail, L == 0 skips drafting. Observe realized
        # survival from the just-verified step first, then decide.
        if not self._sched or dummy_run:
            return self.num_speculative_steps
        now = time.perf_counter()
        self._scheduler.observe_verified(
            input_batch.num_reqs,
            num_sampled,
            num_rejected,
            self.perreq_len,
            input_batch.idx_mapping,
        )
        length = self._scheduler.begin_step(now, input_batch.num_reqs)
        if self.next_k_hint is not None:
            # Dynamic SD: the engine's scheduled width is authoritative.
            length = min(self.next_k_hint, self.num_speculative_steps)
            self._scheduler.skip_next_overhead_sample()
        return length

    def _finalize_draft(
        self,
        input_batch: InputBatch,
        num_reqs: int,
        draft_len: int,
        dummy_run: bool,
    ) -> torch.Tensor:
        if not self._sched or dummy_run:
            return self.draft_tokens[:num_reqs]
        length = draft_len
        if length == 0:
            # Draft forward skipped -> empty draft; the engine runs a
            # normal decode step for this batch.
            self._scheduler.commit_length(0)
            self.perreq_len[input_batch.idx_mapping] = 0
            self._perreq_batch_len = None
            return self.draft_tokens[:num_reqs, :0]
        # The full block always runs under its captured graph, so survival is
        # measured for all gamma positions; update the EMA, then allocate.
        sv = self._survival[:num_reqs]
        self._scheduler.update_survival(sv, input_batch.idx_mapping)
        if self._perreq:
            # Per-request allocation (paper Algorithm 1) within the batch
            # width budget; ragged lengths cut the returned full-width block.
            keep = self._scheduler.allocate(sv, num_reqs, length)
            self.perreq_len[input_batch.idx_mapping] = keep
            self._perreq_batch_len = keep
            length = self.num_speculative_steps
        elif self._tau > 0.0:
            # Per-request confidence cutoff: keep each request's prefix while
            # survival >= tau, pad the rest to the uniform width.
            keep_len = self._scheduler.confidence_widths(sv, length)
            pad = self._step_cols[:length].unsqueeze(0) >= keep_len.unsqueeze(1)
            self.draft_tokens[:num_reqs, :length].masked_fill_(
                pad, self.parallel_drafting_token_id
            )
            self.valid_draft_len[input_batch.idx_mapping] = keep_len
        if not self._perreq:
            self.perreq_len[input_batch.idx_mapping] = length
            self._perreq_batch_len = None
        self._scheduler.commit_length(length)
        # Trim to the chosen verify length; with the multi-width FULL graphs
        # captured, the trimmed verify replays FULL, else it falls to PIECEWISE.
        return self.draft_tokens[:num_reqs, :length]

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
        # Per-position head hidden [num_reqs, n_spec, hidden] for the confidence head.
        sh = sample_hidden.view(num_reqs, n_spec, -1) if self._sched else None
        surv_run: torch.Tensor | None = None

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self._sched:
                # Confidence head -> per-position acceptance prob; cumulative product
                # is the prefix-survival prob the scheduler turns into a verify length.
                conf_i = self.model.compute_confidence(sh[:, i], markov_embed)
                conf_i = conf_i.reshape(num_reqs).sigmoid()
                surv_run = conf_i if surv_run is None else surv_run * conf_i
                self._survival[:num_reqs, i] = surv_run
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
