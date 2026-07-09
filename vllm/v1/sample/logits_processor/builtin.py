# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch

from vllm import SamplingParams
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

T = TypeVar("T")


class MinPLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=is_pin_memory
        )
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            # Pre-allocated device tensor
            self.min_p_device: torch.Tensor = torch.empty(
                (max_num_reqs,), dtype=torch.float32, device=device
            )
        else:
            self.min_p_device = self.min_p_cpu_tensor
        # Current slice of the device tensor
        self.min_p: torch.Tensor = self.min_p_device[:0]

    def is_argmax_invariant(self) -> bool:
        """Min-p never impacts greedy sampling"""
        return True

    def get_min_p_by_index(self, index: int) -> float:
        return float(self.min_p_cpu[index])

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            min_p = params.min_p
            min_p_before = self.min_p_cpu[index]
            if min_p_before != min_p:
                needs_update = True
                self.min_p_cpu[index] = min_p
                if min_p and not min_p_before:
                    self.min_p_count += 1
                elif not min_p and min_p_before:
                    self.min_p_count -= 1

        if self.min_p_count:
            # Process removed requests.
            if batch_update.removed:
                needs_update = True
                for index in batch_update.removed:
                    if self.min_p_cpu[index]:
                        self.min_p_cpu[index] = 0
                        self.min_p_count -= 1

            # Process moved requests, unidirectional (a->b) and swap (a<->b).
            for adx, bdx, direct in batch_update.moved:
                min_p_a, min_p_b = self.min_p_cpu[adx], self.min_p_cpu[bdx]
                if min_p_a != min_p_b:
                    needs_update = True
                    self.min_p_cpu[bdx] = min_p_a
                    if direct == MoveDirectionality.SWAP:
                        self.min_p_cpu[adx] = min_p_b
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if min_p_a:
                        self.min_p_cpu[adx] = 0
                    if min_p_b:
                        self.min_p_count -= 1

        # Update tensors if needed.
        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
            self.min_p = self.min_p_device[:size]
            if self.use_double_tensor:
                self.min_p.copy_(self.min_p_cpu_tensor[:size], non_blocking=True)
            self.min_p.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits


class MinKLogitsProcessor(LogitsProcessor):
    """Min-k sampling (https://arxiv.org/abs/2604.11012).

    Min-k is a temperature-invariant truncation strategy that operates on the
    shape of the sorted logits. For each request it sorts the logits in
    descending order, normalizes the adjacent drops by the logit range, weights
    each drop by 1 / rank to emphasize the head of the distribution, and
    truncates at the position of the steepest weighted drop (the "semantic
    cliff") that separates confident tokens from the long tail. A fallback
    keeps floor(tau / logit_range) tokens when the distribution is nearly flat
    and no cliff exists, which guards against collapsing to a single token.

    vLLM applies temperature before logits processors run, so the logits seen
    here are already divided by the temperature. Temperature scaling preserves
    both the ordering of tokens and the normalized relative decay, so the
    cliff position is identical to the one computed on the raw logits, and the
    final sampling distribution over the kept set matches the paper exactly.
    The fallback uses the range of the temperature-scaled logits, so its size
    can shift with temperature, but it only activates on near-flat
    distributions where it merely prevents a degenerate single-token set.

    A per-request tau of -1 means Min-k is disabled for that request. Disabled
    rows are computed alongside the rest and then masked out, keeping the batch
    vectorized.
    """

    # Sentinel stored per request when Min-k is disabled (valid tau is >= 0).
    _DISABLED = -1.0

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        # Number of requests with Min-k enabled (tau != _DISABLED).
        self.min_k_count: int = 0

        self.tau_cpu_tensor = torch.full(
            (max_num_reqs,),
            self._DISABLED,
            dtype=torch.float32,
            device="cpu",
            pin_memory=is_pin_memory,
        )
        self.tau_cpu = self.tau_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            self.tau_device: torch.Tensor = torch.empty(
                (max_num_reqs,), dtype=torch.float32, device=device
            )
        else:
            self.tau_device = self.tau_cpu_tensor
        # Current slice of the device tensor
        self.tau: torch.Tensor = self.tau_device[:0]

    @classmethod
    def _req_tau(cls, params: SamplingParams) -> float:
        """Effective tau for a request: the requested tau if Min-k is enabled,
        otherwise the disabled sentinel."""
        return params.min_k_tau if params.min_k else cls._DISABLED

    def is_argmax_invariant(self) -> bool:
        """Min-k always keeps the top-ranked token, so it cannot change the
        outcome of greedy sampling."""
        return True

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return

        needs_update = False
        disabled = self._DISABLED
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            tau = self._req_tau(params)
            tau_before = self.tau_cpu[index]
            if tau_before != tau:
                needs_update = True
                self.tau_cpu[index] = tau
                enabled = tau != disabled
                enabled_before = tau_before != disabled
                if enabled and not enabled_before:
                    self.min_k_count += 1
                elif not enabled and enabled_before:
                    self.min_k_count -= 1

        if self.min_k_count:
            # Process removed requests.
            if batch_update.removed:
                needs_update = True
                for index in batch_update.removed:
                    if self.tau_cpu[index] != disabled:
                        self.tau_cpu[index] = disabled
                        self.min_k_count -= 1

            # Process moved requests, unidirectional (a->b) and swap (a<->b).
            for adx, bdx, direct in batch_update.moved:
                tau_a, tau_b = self.tau_cpu[adx], self.tau_cpu[bdx]
                if tau_a != tau_b:
                    needs_update = True
                    self.tau_cpu[bdx] = tau_a
                    if direct == MoveDirectionality.SWAP:
                        self.tau_cpu[adx] = tau_b
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if tau_a != disabled:
                        self.tau_cpu[adx] = disabled
                    if tau_b != disabled:
                        self.min_k_count -= 1

        # Update tensors if needed.
        size = batch_update.batch_size
        if self.min_k_count and (needs_update or self.tau.shape[0] != size):
            self.tau = self.tau_device[:size]
            if self.use_double_tensor:
                self.tau.copy_(self.tau_cpu_tensor[:size], non_blocking=True)
            self.tau.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_k_count:
            return logits

        vocab_size = logits.shape[-1]
        # Sort logits in descending order to inspect the head-to-tail shape.
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
        # Logit range, used to normalize drops so the decay is scale (and thus
        # temperature) invariant. A small epsilon guards a uniform row.
        logit_range = sorted_logits[:, :1] - sorted_logits[:, -1:] + 1e-8
        # Position-weighted relative decay between adjacent sorted logits.
        drops = sorted_logits[:, :-1] - sorted_logits[:, 1:]
        positions = torch.arange(
            1, vocab_size, device=logits.device, dtype=logits.dtype
        )
        weighted_decay = drops / logit_range / positions
        # The steepest weighted drop marks the cliff; keep that many tokens.
        k_cliff = weighted_decay.argmax(dim=-1) + 1
        # Fallback for near-flat rows; disabled rows carry tau -1, clamped to 0.
        tau = self.tau.squeeze(1)
        k_fallback = torch.floor(tau.clamp(min=0.0) / logit_range.squeeze(1)).long()
        k = torch.maximum(k_cliff, k_fallback).clamp_(1, vocab_size)
        # Threshold is the k-th largest logit per row; truncate everything
        # below it, but only for requests that enabled Min-k.
        kth_value = sorted_logits.gather(1, (k - 1).unsqueeze(1))
        invalid_token_mask = (logits < kth_value) & (self.tau != self._DISABLED)
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.biases, batch_update, lambda params, _, __: params.logit_bias or None
        )

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            biases: list[float] = []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(data, device=self.device, dtype=dtype)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        # index -> (min_toks, output_token_ids, stop_token_ids)
        self.device = device
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}

        # (req_idx_tensor,eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

        self.neg_inf_tensor = torch.tensor(
            -float("inf"), dtype=torch.float32, device=self.device
        )

    def is_argmax_invariant(self) -> bool:
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[int, Sequence[int], set[int]] | None:
        min_tokens = params.min_tokens
        if not min_tokens or len(output_tok_ids) >= min_tokens:
            return None
        return min_tokens, output_tok_ids, params.all_stop_token_ids

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.min_toks, batch_update, self.add_request
        )
        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            to_remove = tuple(
                index
                for index, (min_toks, out_tok_ids, _) in self.min_toks.items()
                if len(out_tok_ids) >= min_toks
            )
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            for req, (_, _, stop_tok_ids) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)

            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(data, device=self.device, dtype=dtype)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
        return logits

    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
    ) -> torch.Tensor:
        """Spec-decode version of apply().
        Priority: ``min_tokens`` > ``stop_token_ids`` / EOS.
        Example: ``num_draft_tokens = [2, 3, 1]``
          → ``logits`` shape ``[6, V]``, ``cumsum = [0, 2, 5, 6]``
          → request 0 owns rows 0‑1, request 1 rows 2‑4, request 2 row 5.
        """
        if not self.min_toks:
            return logits

        num_draft_arr = np.array(num_draft_tokens, dtype=np.int64)
        cumsum = np.concatenate([[0], np.cumsum(num_draft_arr)])

        entries = [
            (req_idx, min_tok, len(out_tok_ids), list(stop_tok_ids))
            for req_idx, (min_tok, out_tok_ids, stop_tok_ids) in self.min_toks.items()
            if stop_tok_ids
        ]

        if not entries:
            return logits

        all_rows: list[np.ndarray] = []  # row indices to mask
        all_toks: list[np.ndarray] = []  # stop-token ids at those rows

        for req_idx, min_tok, current_len, stop_toks in entries:
            remaining = min_tok - current_len
            # How many leading draft positions still need stop-token masking.
            n_mask = int(min(max(remaining, 0), num_draft_arr[req_idx]))

            if n_mask > 0:
                offset = cumsum[req_idx]
                row_indices = np.arange(offset, offset + n_mask, dtype=np.int64)
                n_stop = len(stop_toks)
                all_rows.append(np.repeat(row_indices, n_stop))
                all_toks.append(np.tile(stop_toks, n_mask))

        if all_rows:
            rows_arr = np.concatenate(all_rows)
            toks_arr = np.concatenate(all_toks)
            # (row_indices, token_indices) for index_put_ to set -inf.
            logits_slice = (
                async_tensor_h2d(rows_arr, device=self.device),
                async_tensor_h2d(toks_arr, device=self.device),
            )
            logits.index_put_(logits_slice, self.neg_inf_tensor)

        return logits


def process_dict_updates(
    req_entries: dict[int, T],
    batch_update: BatchUpdate | None,
    new_state: Callable[[SamplingParams, list[int] | None, list[int]], T | None],
) -> bool:
    """Utility function to update dict state for sparse LogitsProcessors."""

    if not batch_update:
        # Nothing to do.
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        if (state := new_state(params, prompt_tok_ids, output_tok_ids)) is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # Process removed requests.
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # Process moved requests, unidirectional (a->b) and
        # swapped (a<->b)
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated
