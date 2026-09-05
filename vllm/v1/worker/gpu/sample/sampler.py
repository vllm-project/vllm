# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.config.reasoning import ReasoningConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    flashinfer_sample,
    flashinfer_sampler_supported,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import (
    LogprobTokenIdsState,
    compute_topk_scores,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.sample.trace_replay import TraceReplayState
from vllm.v1.worker.gpu.states import RequestState


class Sampler:
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: RequestState,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
        enable_trace_replay: bool = False,
        reasoning_config: ReasoningConfig | None = None,
        return_sampling_mask: bool = False,
    ):
        self.logprobs_mode = logprobs_mode
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # False by default.
        self.use_fp64_gumbel = use_fp64_gumbel
        self.tp_size = get_tensor_model_parallel_world_size()

        self.req_states = req_states
        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)
        self.penalties_state = PenaltiesState(req_states)
        self.logit_bias_state = LogitBiasState(max_num_reqs, device)
        self.bad_words_state = BadWordsState(req_states)
        self.logprob_token_ids_state = LogprobTokenIdsState(max_num_reqs, device)
        self.thinking_budget_state = ThinkingBudgetState(req_states, reasoning_config)
        self.trace_replay_state = (
            TraceReplayState(req_states) if enable_trace_replay else None
        )
        self.needs_logits_processing = np.zeros(max_num_reqs, dtype=bool)
        self.num_speculative_tokens = num_speculative_tokens
        self.return_sampling_mask = return_sampling_mask
        self.use_flashinfer = (
            not return_sampling_mask and flashinfer_sampler_supported()
        )

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)
        self.logit_bias_state.add_request(req_idx, prompt_len, sampling_params)
        self.bad_words_state.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)
        self.thinking_budget_state.add_request(req_idx, sampling_params)
        if self.trace_replay_state is not None:
            self.trace_replay_state.add_request(req_idx, sampling_params)

        states = self.sampling_states
        temperature = states.temperature.np[req_idx]
        self.needs_logits_processing[req_idx] = (
            self.logit_bias_state.use_logit_bias[req_idx]
            or self.penalties_state.use_penalty[req_idx]
            or self.bad_words_state.num_bad_words.np[req_idx] > 0
            or (
                self.thinking_budget_state.enabled
                and self.thinking_budget_state.use_thinking_budget[req_idx]
            )
            or (temperature != 0.0 and temperature != 1.0)
            or states.min_p.np[req_idx] != 0.0
            or states.top_k.np[req_idx] != states.vocab_size
            or states.top_p.np[req_idx] != 1.0
        )

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes()
        self.logit_bias_state.apply_staged_writes()
        self.bad_words_state.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()
        self.thinking_budget_state.apply_staged_writes()
        if self.trace_replay_state is not None:
            self.trace_replay_state.apply_staged_writes()

    def get_logprobs_dims(
        self, idx_mapping_np: np.ndarray, include_token_ids: bool = True
    ) -> tuple[int, int] | None:
        """(num_logprobs, max_per_req_token_ids) for the given requests, or
        None when none of them want logprobs."""
        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        max_token_ids = (
            self.logprob_token_ids_state.max_num_token_ids(idx_mapping_np)
            if include_token_ids
            else 0
        )
        if max_num_logprobs == NO_LOGPROBS and max_token_ids == 0:
            return None
        num_logprobs = max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0
        return num_logprobs, max_token_ids

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        *,
        use_reduced_sampling: bool = False,
        vocab_start_index: int | None = None,
    ) -> SamplerOutput:
        expanded_idx_mapping = input_batch.expanded_idx_mapping
        idx_mapping = input_batch.idx_mapping
        idx_mapping_np = input_batch.idx_mapping_np
        cu_num_logits_np = input_batch.cu_num_logits_np
        expanded_local_pos = input_batch.expanded_local_pos
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]

        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.compute_nans else None

        logprobs_dims = self.get_logprobs_dims(idx_mapping_np)

        sampled, processed_logits = self.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            return_logprobs=logprobs_dims is not None,
            use_reduced_sampling=use_reduced_sampling,
            vocab_start_index=vocab_start_index,
        )

        if self.trace_replay_state is not None:
            # Overwrite sampled tokens with the replay trace up-front so that
            # computed logprobs reflect the real distribution of the forced token.
            self.trace_replay_state.apply_trace(sampled, idx_mapping)

        if logprobs_dims is not None:
            num_logprobs, max_per_req_token_ids = logprobs_dims
            if self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
                logits = processed_logits
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            logprobs_tensors = compute_topk_scores(
                logits,
                num_logprobs,
                sampled,
                cu_num_logits,
                logprob_token_ids_state=self.logprob_token_ids_state,
                expanded_idx_mapping=input_batch.expanded_idx_mapping,
                max_per_req_token_ids=max_per_req_token_ids,
                logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
            )
        else:
            logprobs_tensors = None

        # 1 sampled token per request, except chunked-prefill requests
        # (seq_len < prefill_len) which aren't done prefilling and produce no
        # output token. num_rejected is always 0 here (one logit per request).
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )

        sampling_mask_tensors = None
        if self.return_sampling_mask:
            # Size by the validated top_k batch max; wider supports use the bitmask.
            max_num_kept = int(np.max(self.sampling_states.top_k.np[idx_mapping_np]))
            sampling_mask_tensors = SamplingMaskTensors.from_logits(
                processed_logits, num_sampled, max_num_kept
            )

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            sampling_mask_tensors=sampling_mask_tensors,
        )
        return sampler_output

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        skip_top_k_top_p: bool = False,
    ) -> torch.Tensor:
        if not np.any(self.needs_logits_processing[idx_mapping_np]):
            return logits

        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply logit bias (e.g., allowed_token_ids, min_tokens) in place.
        self.logit_bias_state.apply_logit_bias(
            logits, expanded_idx_mapping, idx_mapping_np, pos
        )

        # Apply penalties in place.
        self.penalties_state.apply_penalties(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Apply bad words masking in place.
        self.bad_words_state.apply_bad_words(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Force the reasoning end marker once a request's thinking budget is
        # reached; applied before temperature so the forced token is always kept.
        self.thinking_budget_state.apply(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Apply temperature in place.
        self.sampling_states.apply_temperature(
            logits, expanded_idx_mapping, idx_mapping_np
        )

        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        if skip_top_k_top_p:
            return logits

        # Apply top_k and/or top_p. This might or might not return a new tensor.
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )

    def can_use_reduced_sampling(
        self, idx_mapping_np: np.ndarray, local_logits: torch.Tensor
    ) -> bool:
        """Return whether candidate reduction preserves the sampler contract."""
        if self.return_sampling_mask or self.compute_nans:
            return False

        if np.any(self.logit_bias_state.use_logit_bias[idx_mapping_np]):
            return False
        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):
            return False
        if np.any(self.penalties_state.use_penalty[idx_mapping_np]):
            return False

        thinking_state = self.thinking_budget_state
        if thinking_state.enabled and np.any(
            thinking_state.use_thinking_budget[idx_mapping_np]
        ):
            return False

        states = self.sampling_states
        if np.any(states.min_p.np[idx_mapping_np] != 0.0):
            return False
        if states.max_num_logprobs(idx_mapping_np) != NO_LOGPROBS:
            return False
        if self.logprob_token_ids_state.max_num_token_ids(idx_mapping_np) > 0:
            return False

        vocab_size = states.vocab_size
        if vocab_size >= 2**24:
            return False

        full_logits_bytes = local_logits.element_size() * local_logits.shape[-1]
        temperatures = states.temperature.np[idx_mapping_np]
        random_mask = temperatures != 0.0
        if not np.any(random_mask):
            return full_logits_bytes > 8

        random_req_indices = idx_mapping_np[random_mask]
        if states.any_explicit_seed(random_req_indices):
            # Candidate positions are not global token IDs, so their Gumbel RNG
            # keys differ from the full-vocabulary path.
            return False

        random_top_k = states.top_k.np[random_req_indices]
        max_top_k = int(random_top_k.max())
        if np.any(random_top_k >= vocab_size):
            return False

        # Candidate pairs are packed as two FP32 values. Only use the reduced
        # path when that payload is smaller than gathering the local logits.
        candidate_bytes = 8 * max_top_k
        return candidate_bytes < full_logits_bytes

    def _sample_reduced(
        self,
        local_logits: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        pos: torch.Tensor,
        vocab_start_index: int,
    ) -> torch.Tensor:
        """Reduced sampling via local top-k + TP all-gather.

        Each TP rank selects its top-k logits locally, all-gathers only those
        k values and global indices, then samples from that candidate set.
        Communication is O(B * 2 * k * tp_size), instead of O(B * V).

        All-greedy batches use a dedicated local-max/global-argmax path and
        skip top-k filtering and Gumbel sampling. The Gumbel fallback serves
        mixed and all-random batches.

        Args:
            local_logits: Processed shard-local logits with shape
                [num_tokens, local_vocab_size].
            top_k: Per-token top_k values [num_tokens] or None.
            top_p: Per-token top_p values [num_tokens] or None.
            expanded_idx_mapping: [num_tokens] mapping to request state idx.
            idx_mapping_np: [num_tokens] numpy mapping (for GPU-sync-free
                access to sampling states).
            temperature: [max_num_reqs] per-request temperature.
            seeds: [max_num_reqs] per-request random seeds.
            pos: [num_tokens] position within request (for RNG seeding).
            vocab_start_index: Global token ID of the first local logit.

        Returns:
            Sampled global token IDs [num_tokens].
        """
        _, local_vocab_size = local_logits.shape

        temperatures = self.sampling_states.temperature.np[idx_mapping_np]
        random_mask = temperatures != 0.0
        all_greedy = not np.any(random_mask)
        if all_greedy:
            k_for_topk = 1
        else:
            assert top_k is not None
            random_top_k = self.sampling_states.top_k.np[idx_mapping_np][random_mask]
            k_for_topk = int(random_top_k.max())
            assert 0 < k_for_topk < local_vocab_size

        if all_greedy:
            local_vals, local_idx = local_logits.max(dim=-1, keepdim=True)
        else:
            local_vals, local_idx = torch.topk(local_logits, k_for_topk, dim=-1)
        local_global_idx = local_idx + vocab_start_index

        # Pack values and token IDs into one FP32 tensor to avoid paying for
        # two tensor-parallel collectives. FP32 represents practical vocab IDs
        # exactly (up to 2**24), unlike FP16/BF16.
        packed_candidates = torch.cat(
            (local_vals.float(), local_global_idx.float()), dim=-1
        )
        gathered_candidates = tensor_model_parallel_all_gather(
            packed_candidates, dim=-1
        )

        # all_gather concatenates rank-local chunks, giving the layout
        # [rank0_vals, rank0_ids, rank1_vals, rank1_ids, ...].
        gathered_candidates = gathered_candidates.unflatten(
            -1, (self.tp_size, 2, k_for_topk)
        )
        gathered_vals = gathered_candidates[..., 0, :].flatten(-2).contiguous()
        gathered_idx = (
            gathered_candidates[..., 1, :].flatten(-2).to(torch.int64).contiguous()
        )

        if all_greedy:
            sample_pos = gathered_vals.argmax(dim=-1, keepdim=True)
            return gathered_idx.gather(dim=-1, index=sample_pos).squeeze(-1)

        # Clamp top_k to cand_size: rows with top_k = vocab_size (meaning
        # "no top_k") would otherwise produce negative gather indices when
        # cand_size < vocab_size.
        cand_size = gathered_vals.shape[-1]
        if top_k is not None:
            top_k = top_k.to(torch.long).clamp(min=1, max=cand_size).to(torch.int32)
        gathered_vals = apply_top_k_top_p(gathered_vals, top_k, top_p)
        sample_pos = gumbel_sample(
            gathered_vals,
            expanded_idx_mapping,
            temperature,
            seeds,
            pos,
            apply_temperature=False,
            use_fp64=self.use_fp64_gumbel,
        )
        sample_pos = sample_pos.unsqueeze(-1)

        # Map candidate position → global token ID.
        return gathered_idx.gather(dim=-1, index=sample_pos).squeeze(-1).to(torch.int64)

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool = False,
        use_reduced_sampling: bool = False,
        vocab_start_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
        )

        # The runner enables this path only for shard-local logits. Full-logits
        # fallbacks must use the standard path to avoid offsetting token IDs.
        if use_reduced_sampling:
            assert self.tp_size > 1
            assert vocab_start_index is not None
            top_k, top_p = self.sampling_states.get_top_k_top_p(
                expanded_idx_mapping, idx_mapping_np
            )
            sampled = self._sample_reduced(
                processed_logits,
                top_k,
                top_p,
                expanded_idx_mapping,
                idx_mapping_np,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                vocab_start_index,
            )
            return sampled, processed_logits

        # Standard sampling path for full-vocabulary logits.
        top_k, top_p = self.sampling_states.get_top_k_top_p(
            expanded_idx_mapping, idx_mapping_np
        )
        use_flashinfer = self.use_flashinfer and not (
            # Don't use FI sampler if no requests use top_k/top_p, if there are
            # any greedy requests or per-request seeds, or if post-processed
            # logprobs need to be returned for any requests.
            (top_k is None and top_p is None)
            or (return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES)
            or self.sampling_states.any_greedy(idx_mapping_np)
            or self.sampling_states.any_explicit_seed(idx_mapping_np)
        )

        # Sample the next token.
        if use_flashinfer:
            sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
            processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
            sampled = gumbel_sample(
                processed_logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                is_drafting=False,
                use_fp64=self.use_fp64_gumbel,
            )
        return sampled, processed_logits
