# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from collections.abc import Sequence

from vllm.sampling_params import SamplingKnobs, SamplingParams, tokens_in_reasoning
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p

NO_LOGPROBS = -1
_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max


class SamplingStates:
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        reasoning_start_token_ids: Sequence[int] | None = None,
        reasoning_end_token_ids: Sequence[int] | None = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size
        self.reasoning_start_token_ids = list(reasoning_start_token_ids or [])
        self.reasoning_end_token_ids = list(reasoning_end_token_ids or [])

        self.temperature = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.top_k = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.top_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.min_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.seeds = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        # Tracks whether `seed` was set explicitly by the user, so callers
        # can fall back from RNG paths that don't honor per-request seeds.
        self.seeds_set = np.zeros(max_num_reqs, dtype=bool)

        # Initialize top_k and top_p manually because 0 is an invalid value for them.
        self.top_k.np.fill(self.vocab_size)
        self.top_k.copy_to_uva()
        self.top_p.np.fill(1.0)
        self.top_p.copy_to_uva()

        self.num_logprobs = np.empty(self.max_num_reqs, dtype=np.int32)
        # -1 means no logprobs are requested.
        self.num_logprobs.fill(NO_LOGPROBS)

        self.post_thinking_params: dict[int, SamplingParams] = {}
        self.in_reasoning = np.ones(max_num_reqs, dtype=bool)
        self._marker_window: dict[int, list[int]] = {}
        self.knobs_dirty = False

    def add_request(
        self,
        req_idx: int,
        sampling_params: SamplingParams,
        token_ids: Sequence[int] | None = None,
    ) -> None:
        seed = sampling_params.seed
        self.seeds_set[req_idx] = seed is not None
        if seed is None:
            seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
        self.seeds.np[req_idx] = seed

        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = NO_LOGPROBS
        elif num_logprobs == -1:
            num_logprobs = self.vocab_size
        self.num_logprobs[req_idx] = num_logprobs

        in_reasoning = True
        if sampling_params.post_thinking is not None:
            self.post_thinking_params[req_idx] = sampling_params
            ids = list(token_ids or [])
            in_reasoning = tokens_in_reasoning(
                ids, self.reasoning_start_token_ids, self.reasoning_end_token_ids
            )
            overlap = self._marker_overlap()
            self._marker_window[req_idx] = ids[-overlap:] if overlap else []
        else:
            self.post_thinking_params.pop(req_idx, None)
            self._marker_window.pop(req_idx, None)
        self.in_reasoning[req_idx] = in_reasoning
        self._write_knobs(
            req_idx, sampling_params.resolve_sampling_knobs(in_reasoning)
        )

    def observe_tokens(self, req_idx: int, new_token_ids: Sequence[int]) -> bool:
        """Update in-reasoning state from newly sampled tokens.

        Returns True when the active overlay changed.
        """
        params = self.post_thinking_params.get(req_idx)
        if params is None or not new_token_ids:
            return False
        overlap = self._marker_overlap()
        prev = self._marker_window.get(req_idx, [])
        window = prev[-overlap:] + list(new_token_ids)
        self._marker_window[req_idx] = window[-max(overlap, 1) :]
        in_reasoning = self._in_reasoning_from_window(
            window, bool(self.in_reasoning[req_idx])
        )
        if in_reasoning == bool(self.in_reasoning[req_idx]):
            return False
        self.in_reasoning[req_idx] = in_reasoning
        self._write_knobs(req_idx, params.resolve_sampling_knobs(in_reasoning))
        self.knobs_dirty = True
        return True

    def _marker_overlap(self) -> int:
        return max(
            len(self.reasoning_start_token_ids),
            len(self.reasoning_end_token_ids),
            1,
        )

    def _in_reasoning_from_window(
        self, window: Sequence[int], current: bool
    ) -> bool:
        from vllm.sampling_params import last_subsequence_index

        last_start = last_subsequence_index(window, self.reasoning_start_token_ids)
        last_end = last_subsequence_index(window, self.reasoning_end_token_ids)
        if last_start < 0 and last_end < 0:
            return current
        return last_start > last_end

    def _write_knobs(self, req_idx: int, knobs: SamplingKnobs) -> None:
        self.temperature.np[req_idx] = knobs.temperature
        self.top_p.np[req_idx] = knobs.top_p
        top_k = knobs.top_k
        if top_k <= 0 or top_k > self.vocab_size:
            top_k = self.vocab_size
        self.top_k.np[req_idx] = top_k
        self.min_p.np[req_idx] = knobs.min_p

    def apply_staged_writes(self) -> None:
        self.temperature.copy_to_uva()
        self.top_p.copy_to_uva()
        self.top_k.copy_to_uva()
        self.min_p.copy_to_uva()
        self.seeds.copy_to_uva()

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        temp_np = self.temperature.np[idx_mapping_np]
        if np.all((temp_np == 0.0) | (temp_np == 1.0)):
            # No request requires temperature. Skip the kernel launch.
            return

        apply_temperature(logits, expanded_idx_mapping, self.temperature.gpu)

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if np.all(self.min_p.np[idx_mapping_np] == 0.0):
            # No request uses min_p. Skip the kernel launch.
            return
        apply_min_p(logits, expanded_idx_mapping, self.min_p.gpu)

    def get_top_k_top_p(
        self, expanded_idx_mapping: torch.Tensor, idx_mapping_np: np.ndarray
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        do_top_k = np.any(self.top_k.np[idx_mapping_np] != self.vocab_size)
        do_top_p = np.any(self.top_p.np[idx_mapping_np] != 1.0)
        top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None
        top_p = self.top_p.gpu[expanded_idx_mapping] if do_top_p else None
        return top_k, top_p

    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        top_k, top_p = self.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        if top_k is None and top_p is None:
            return logits
        return apply_top_k_top_p(logits, top_k, top_p)

    def any_greedy(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.temperature.np[idx_mapping_np] == 0.0))

    def any_explicit_seed(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.seeds_set[idx_mapping_np]))

    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int:
        return int(np.max(self.num_logprobs[idx_mapping_np]))
