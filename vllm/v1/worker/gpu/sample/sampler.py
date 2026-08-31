# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.config.reasoning import ReasoningConfig
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
        self.device = device

    @torch.inference_mode()
    def warmup_top_k_top_p_buffer(self, num_rows: int) -> None:
        """Warm the shape-dependent top-k/top-p sampler workspace.

        Speculative verification applies filters to the flattened target rows
        (one row for each draft step plus the bonus row). The normal profile
        sampler only uses one row per request, so a larger Triton workspace
        could otherwise be allocated after the KV cache and CUDA graphs are
        already committed.
        """
        if num_rows < 8:
            # The dispatcher uses the eager PyTorch path for small batches.
            return

        vocab_size = self.sampling_states.vocab_size
        if vocab_size <= 0:
            return
        logits = torch.zeros(
            (num_rows, vocab_size), dtype=torch.float32, device=self.device
        )
        top_k = torch.full(
            (num_rows,), min(20, vocab_size), dtype=torch.int32, device=self.device
        )
        top_p = torch.full(
            (num_rows,), 0.95, dtype=torch.float32, device=self.device
        )
        apply_top_k_top_p(logits, top_k, top_p)
        # Profiling normally runs on CUDA, but keeping the helper usable by
        # CPU/unit-test runners avoids asking torch.accelerator to initialize
        # a nonexistent CUDA device.
        if self.device.type != "cpu":
            torch.accelerator.synchronize()
        del logits, top_k, top_p

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

    def get_vocab_parallel_sampling_params(
        self, input_batch: InputBatch
    ) -> tuple[str, int, float, float, bool] | None:
        """Return uniform parameters for a compact vocab-parallel sampler.

        The vocab-parallel path can preserve an unrestricted distribution via
        local Gumbel-max, and a bounded top-k distribution when the only logits
        processor is presence penalty. Repetition and frequency penalties are
        non-constant over a token's sign/count and therefore remain on the
        full-vocabulary path. The boolean in the return tuple indicates that
        presence-only metadata must be passed to the model-level sampler.
        """
        idx_mapping_np = input_batch.idx_mapping_np
        if (
            idx_mapping_np.size == 0
            or self.compute_nans
            or self.return_sampling_mask
            or self.trace_replay_state is not None
            or self.sampling_states.max_num_logprobs(idx_mapping_np) != NO_LOGPROBS
            or self.logprob_token_ids_state.max_num_token_ids(idx_mapping_np) > 0
            or self.sampling_states.any_explicit_seed(idx_mapping_np)
        ):
            return None

        # Logit bias, bad words and thinking-budget forcing alter individual
        # token scores and cannot be reconstructed from a compact candidate
        # matrix. Temperature/top-k/top-p are handled below and are expected to
        # make ``needs_logits_processing`` true for random requests.
        if np.any(self.logit_bias_state.use_logit_bias[idx_mapping_np]):
            return None
        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):
            return None
        if self.thinking_budget_state.enabled and np.any(
            self.thinking_budget_state.use_thinking_budget[idx_mapping_np]
        ):
            return None

        use_penalty = self.penalties_state.use_penalty[idx_mapping_np]
        presence_only = False
        if np.any(use_penalty):
            repetition = self.penalties_state.repetition_penalty.np[idx_mapping_np]
            frequency = self.penalties_state.frequency_penalty.np[idx_mapping_np]
            presence_only = bool(
                np.all(repetition == 1.0) and np.all(frequency == 0.0)
            )
            if not presence_only:
                return None

        temperatures = self.sampling_states.temperature.np[idx_mapping_np]
        top_ks = self.sampling_states.top_k.np[idx_mapping_np]
        top_ps = self.sampling_states.top_p.np[idx_mapping_np]
        min_ps = self.sampling_states.min_p.np[idx_mapping_np]
        if not np.all(min_ps == 0.0):
            return None

        default_filters = (
            np.all(top_ks == self.sampling_states.vocab_size)
            and np.all(top_ps == 1.0)
        )
        if np.all(temperatures == 0.0):
            # A presence penalty changes greedy argmax, so it must be applied
            # by the regular sampler. Without penalties, top-k/top-p are
            # argmax-invariant and the existing greedy path is safe.
            if presence_only or not default_filters:
                return None
            return (
                "greedy",
                self.sampling_states.vocab_size,
                1.0,
                0.0,
                False,
            )

        # Full-distribution random sampling is supported by a local Gumbel-max
        # reduction; only request modes with mixed temperature remain unsafe.
        if (
            self.use_fp64_gumbel
            or np.any(temperatures <= 0.0)
            or not np.all(temperatures == temperatures[0])
        ):
            return None

        top_k = int(top_ks[0])
        top_p = float(top_ps[0])
        temperature = float(temperatures[0])
        if default_filters:
            # Gumbel-max over each TP-local shard is exact for an unrestricted
            # softmax distribution and avoids gathering the full vocabulary.
            # A presence penalty changes every previously seen token and would
            # require a full-vocabulary penalty application before Gumbel-max.
            if presence_only:
                return None
            return (
                "full",
                self.sampling_states.vocab_size,
                1.0,
                temperature,
                False,
            )
        if (
            top_k <= 0
            or top_k > 64
            or top_p <= 0.0
            or top_p > 1.0
            or not np.all(top_ks == top_k)
            or not np.all(top_ps == top_p)
            or not np.all(temperatures == temperature)
        ):
            return None

        # Presence-only penalties are supported by the compact path. This is
        # intentionally returned only for bounded top-k, never for full random
        # sampling where a candidate approximation would change probabilities.
        return ("topk", top_k, top_p, temperature, presence_only)

    def get_vocab_parallel_presence_inputs(
        self, input_batch: InputBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return row-aligned presence penalties and persistent output counts."""
        request_indices = input_batch.idx_mapping.to(torch.int64)
        presence_penalties = self.penalties_state.presence_penalty.gpu.index_select(
            0, request_indices
        )
        return (
            presence_penalties,
            self.penalties_state.output_bin_counts,
            request_indices,
        )

    def make_sampler_output(
        self,
        sampled: torch.Tensor,
        input_batch: InputBatch,
        *,
        num_nans: torch.Tensor | None = None,
    ) -> SamplerOutput:
        """Build the standard one-token output for pre-sampled rows."""
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

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
            sampling_mask_tensors = SamplingMaskTensors.from_logits(
                processed_logits, num_sampled
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
                use_fp64=self.use_fp64_gumbel,
            )
        return sampled, processed_logits
