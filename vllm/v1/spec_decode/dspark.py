# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark proposer for the (legacy) V1 GPUModelRunner.

DSpark performs semi-autoregressive parallel drafting: it drafts a block of
``num_speculative_tokens`` tokens in one parallel pass (reusing the DFlash
machinery - context-KV precompute + a non-causal query-block forward), then
injects intra-block dependency with a lightweight sequential Markov head.

This is the V1-runner counterpart of
``vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`` (the V2 speculator). It
subclasses :class:`DFlashProposer` so it inherits the context-KV precompute,
query-block attention metadata, padded-drafter batch, dummy_run, CUDA-graph and
attention-backend plumbing, and only overrides:

  * the input layout (anchor-as-first): each request emits exactly
    ``num_speculative_tokens`` query tokens and every query position is a
    prediction. This is driven by ``sample_from_anchor`` / ``num_query_per_req``
    (see ``DFlashProposer.set_inputs_first_pass`` and the shared
    ``copy_and_expand_dflash_inputs_kernel``). Speculators-format checkpoints
    that set ``dspark_bonus_anchor`` instead reuse the DFlash ``1 + N`` fill-in
    layout where the anchor is a separate bonus token.
  * the draft-token sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step, plus an optional TreeFlash
    hidden-states correction.
"""

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import compute_probs_and_sample_next_token
from vllm.v1.spec_decode.utils import token_logprobs_from_logits

logger = init_logger(__name__)


class DSparkProposer(DFlashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)

        # Anchor-as-first (N query slots) unless the checkpoint uses the 1 + N
        # fill-in block, where the anchor is a separate bonus token. This mirrors
        # DSparkSpeculator in the V2 runner.
        self.sample_from_anchor = not getattr(
            self.draft_model_config.hf_config, "dspark_bonus_anchor", False
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_tokens
        else:
            self.num_query_per_req = 1 + self.num_speculative_tokens

        # In the 1+N bonus-anchor layout, position 0 belongs to the supplied
        # anchor and the N sampled draft tokens correspond to alpha[1:N+1].
        hf_config = self.draft_model_config.hf_config
        block_size = getattr(hf_config, "block_size", None)
        self.markov_position_offset = int(
            not self.sample_from_anchor
            and bool(getattr(hf_config, "markov_pos_adaptive", False))
            and block_size == self.num_query_per_req
        )

        # DSpark always uses non-causal attention (block drafting).
        self.dflash_causal = False

        # Anchor query index into the flat, request-major query buffer. The
        # anchor sits at query offset 0 of each request, i.e. its input id (the
        # bonus / next token) lives at ``req_idx * num_query_per_req``.
        self._anchor_idx = (
            torch.arange(self.max_batch_size, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        self.target_vocab_size = vllm_config.model_config.get_vocab_size()

        # Reduced-vocab probabilistic drafting scatter map/buffer; populated in
        # load_model when the draft model uses a reduced vocabulary.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None

    def _draft_model(self):
        """Return the underlying draft model, unwrapping any CUDA-graph wrapper."""
        model = self.model
        if isinstance(model, BreakableCUDAGraphWrapper):
            model = model.unwrap()
        return model

    def load_model(self, target_model) -> None:
        super().load_model(target_model)

        # Reduced draft vocab: probabilistic rejection sampling needs draft
        # probabilities in the target vocabulary. Precompute the draft->target
        # column map and a scratch buffer used to scatter draft logits into
        # target-vocab space before sampling. Full-vocab drafts (identity map)
        # skip this and sample directly in target space.
        model = self._draft_model()
        d2t = getattr(model, "draft_id_to_target_id", None)
        if self._enable_probabilistic_draft_probs and d2t is not None:
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            self._draft_scatter_buf = torch.full(
                (self.max_batch_size, self.target_vocab_size),
                float("-inf"),
                dtype=self.dtype,
                device=self.device,
            )

    def _scatter_to_target(
        self, draft_logits: torch.Tensor, num_reqs: int
    ) -> torch.Tensor:
        """Scatter draft-vocab logits into target-vocab space.

        Returns ``draft_logits`` unchanged when the draft uses the full target
        vocabulary (identity mapping).
        """
        if self._d2t_scatter_index is None:
            return draft_logits
        assert self._draft_scatter_buf is not None
        buf = self._draft_scatter_buf[:num_reqs]
        buf.fill_(float("-inf"))
        buf.index_copy_(1, self._d2t_scatter_index, draft_logits.to(buf.dtype))
        return buf

    def _finish_parallel_proposal(
        self,
        sample_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sequential Markov sampling over the parallel backbone's outputs.

        ``sample_hidden_states`` is ``[num_reqs * N, hidden]`` laid out
        request-major: rows ``[r*N : (r+1)*N]`` are the N query positions of
        request ``r`` (position 0 is the anchor). We sample left-to-right,
        feeding the previously sampled token into the Markov head (and the
        optional hidden-state correction) at each step.
        """
        self._dcut_keep_lens_cache = None
        model = self._draft_model()
        n_spec = self.num_speculative_tokens

        total = sample_hidden_states.shape[0]
        num_reqs = total // n_spec

        probabilistic = (
            self._enable_probabilistic_draft_probs and not sampling_metadata.all_greedy
        )

        use_hidden_correction = bool(
            getattr(model, "has_hidden_correction", lambda: False)()
        )
        hidden_per_step = sample_hidden_states.view(num_reqs, n_spec, -1)
        if not use_hidden_correction:
            base_logits = model.compute_draft_logits(sample_hidden_states).view(
                num_reqs, n_spec, -1
            )
        # Optional Markov head: skip per-step bias when the draft model was
        # built without one (checkpoint has no markov weights).
        use_markov = bool(getattr(model, "has_markov", lambda: True)())

        # Step 0 predecessor: the anchor (bonus / next) token, teacher-forced.
        prev = self.input_ids[self._anchor_idx[:num_reqs]].long()

        draft_tokens = torch.empty(
            (num_reqs, n_spec), dtype=torch.int64, device=self.device
        )
        draft_probs_list: list[torch.Tensor] | None = [] if probabilistic else None
        # D-Cut scores the drafts after the block is complete, so collect the
        # sampled-token logprob of every step and hand the stacked
        # ``[num_reqs, n_spec]`` matrix to the shared selector below.
        logprobs_list: list[torch.Tensor] | None = [] if self.dcut_enabled else None

        for i in range(n_spec):
            if use_hidden_correction:
                corrected = model.apply_hidden_correction(hidden_per_step[:, i], prev)
                logits_i = model.compute_draft_logits(corrected)
            else:
                logits_i = base_logits[:, i]

            if use_markov:
                # Prefix-dependent Markov transition bias (draft-vocab space).
                bias = model.markov_bias(
                    model.markov_embed(prev),
                    step=i + self.markov_position_offset,
                )
                logits_i = logits_i + bias

            if probabilistic:
                target_logits = self._scatter_to_target(logits_i, num_reqs)
                sampled_i, probs_i = compute_probs_and_sample_next_token(
                    target_logits, sampling_metadata, self.use_fp64_gumbel
                )
                if logprobs_list is not None:
                    logprobs_list.append(
                        torch.log(probs_i.gather(1, sampled_i.unsqueeze(1)).squeeze(1))
                    )
            else:
                draft_argmax = logits_i.argmax(dim=-1)
                sampled_i = model.map_draft_to_target(draft_argmax)
                probs_i = None
                if logprobs_list is not None:
                    # Score in draft-vocab space, where the argmax was taken.
                    logprobs_list.append(
                        token_logprobs_from_logits(logits_i, draft_argmax)
                    )

            draft_tokens[:, i] = sampled_i
            prev = sampled_i
            if draft_probs_list is not None:
                assert probs_i is not None
                draft_probs_list.append(probs_i)

        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        if logprobs_list is not None:
            self.select_dcut_keep_lens(torch.stack(logprobs_list, dim=1))
        return draft_tokens
