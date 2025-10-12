# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

import torch

from vllm import SamplingParams
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

    def update_state(self, batch_update: Optional[BatchUpdate]):
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
        logits[invalid_token_mask] = -float("inf")
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
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

    def update_state(self, batch_update: Optional[BatchUpdate]):
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
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

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
        self.pin_memory = is_pin_memory
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}

        # (req_idx_tensor,eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

    def is_argmax_invariant(self) -> bool:
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    @staticmethod
    def add_request(
        params: SamplingParams, _: Optional[list[int]], output_tok_ids: list[int]
    ) -> Optional[tuple[int, Sequence[int], set[int]]]:
        min_tokens = params.min_tokens
        if not min_tokens or len(output_tok_ids) >= min_tokens:
            return None
        return min_tokens, output_tok_ids, params.all_stop_token_ids

    def update_state(self, batch_update: Optional[BatchUpdate]):
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
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits[self.logits_slice] = -float("inf")
        return logits


class ThinkingTokenBudgetLogitsProcessor(LogitsProcessor):
    """Limits the number of tokens allowed inside a 'thinking' section."""

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        """
        Args:
          reasoning_config: Configuration for reasoning, which includes
            the token IDs for thinking start and end.
          pin_memory (bool): Whether to use pinned memory for tensors.
          device (torch.device): Device to use for tensor operations.
        """
        reasoning_config = vllm_config.reasoning_config
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        # Check if thinking is enabled
        self.is_enabled = (
            reasoning_config is not None and reasoning_config.is_thinking_enabled()
        )

        self.think_start_token_ids = getattr(
            reasoning_config, "think_start_token_ids", []
        )
        self.think_end_token_ids = getattr(reasoning_config, "think_end_token_ids", [])

        self.pin_memory = is_pin_memory
        self.device = device
        # Per-request state tracking for thinking token management
        # Key: request_index, Value: state dict containing:
        # "in_think": bool - currently in thinking mode
        # "in_end": bool - currently forcing end tokens output
        # "check_count_down": int - steps remaining until next think
        #                            start/end token parsing
        # "think_count": int - number of thinking tokens generated
        # "end_count": int - number of end tokens forced so far
        # "thinking_token_budget": int - max allowed thinking tokens
        # "output_tok_ids": list[int] - generated output tokens
        # "prev_output_length": int - previous output length for
        #                               incremental processing
        self._state: dict[int, dict[str, Any]] = {}

        # Preallocate reusable tensors
        self.mask = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.force_token_ids = torch.full(
            (max_num_reqs,), -1, dtype=torch.long, device=device
        )

    @staticmethod
    def _find_last_sequence_index(target_list: list[int], token_ids: list[int]) -> int:
        """
        Returns the index of the last occurrence of token_ids in target_list.

        Args:
          target_list (list[int]): The list of token IDs.
          token_ids (list[int]): The sequence of token IDs to find.
        """
        if not token_ids:
            return -1
        for i in range(len(target_list) - len(token_ids), -1, -1):
            if target_list[i : i + len(token_ids)] == token_ids:
                return i
        return -1

    def _init_state_entry(
        self, prompt_tok_ids: Optional[list[int]], thinking_token_budget: int
    ) -> dict[str, Any]:
        """Initializes the tracking state for a given sequence index."""
        if prompt_tok_ids is None:
            last_start = -1
            last_end = -1
            in_think = False
            think_count = 0
        else:
            last_start = self._find_last_sequence_index(
                prompt_tok_ids, self.think_start_token_ids
            )
            last_end = self._find_last_sequence_index(
                prompt_tok_ids, self.think_end_token_ids
            )
            in_think = last_start > last_end
            if in_think:
                think_count = len(prompt_tok_ids) - (
                    last_start + len(self.think_start_token_ids)
                )
            else:
                think_count = 0

        return {
            "in_think": in_think,  # Currently in thinking mode
            "in_end": in_think and thinking_token_budget == 0,
            "check_count_down": thinking_token_budget,
            "think_count": think_count,  # Number of tokens in thinking section
            "end_count": 0,  # Number of end tokens forced so far
            "prompt_tok_ids": prompt_tok_ids,
            "output_tok_ids": [],
            "thinking_token_budget": thinking_token_budget,
            "prev_output_length": 0,
            # Track previous output length for incremental updates
        }

    def _update_think_state(self, state: dict[str, Any]):
        """Updates the state based on newly generated output tokens."""
        if not state.get("in_end", False) and state.get("check_count_down", 0) > 0:
            state["check_count_down"] -= 1
            return

        output = state.get("output_tok_ids", [])
        if not output:
            return

        # Track previous output length for incremental processing
        prev_length = state.get("prev_output_length", 0)
        current_length = len(output)

        if current_length <= prev_length:
            return

        # Process only newly added tokens
        new_tokens = output[prev_length:]
        state["prev_output_length"] = current_length

        # Check if new tokens contain think start or end sequences
        start_len = len(self.think_start_token_ids)
        end_len = len(self.think_end_token_ids)

        # Look for think sequences in recent tokens (including boundary)
        # Check overlapping regions where sequences might span boundaries
        check_start_idx = max(0, prev_length - max(start_len, end_len) + 1)
        recent_tokens = output[check_start_idx:]

        # Find any think start/end sequences in recent tokens
        recent_start_pos = self._find_last_sequence_index(
            recent_tokens, self.think_start_token_ids
        )
        recent_end_pos = self._find_last_sequence_index(
            recent_tokens, self.think_end_token_ids
        )

        # Update state based on recent sequences
        if not state["in_end"]:
            if recent_start_pos >= 0 and recent_end_pos >= 0:
                if recent_start_pos > recent_end_pos:
                    # Case: ...<end>...<start>... - entering think mode
                    absolute_start_pos = check_start_idx + recent_start_pos
                    new_think_count = current_length - (absolute_start_pos + start_len)
                    state["in_think"] = True
                    state["think_count"] = new_think_count
                else:
                    # Case: ...<start>...<end>... - exiting think mode
                    state["in_think"] = False
                    state["think_count"] = 0
            elif recent_start_pos >= 0:
                # Found think start - entering think mode
                absolute_start_pos = check_start_idx + recent_start_pos
                new_think_count = current_length - (absolute_start_pos + start_len)
                state["in_think"] = True
                state["think_count"] = new_think_count
            elif recent_end_pos >= 0:
                # Found think end - exiting think mode
                state["in_think"] = False
                state["think_count"] = 0
            elif state["in_think"]:
                # Continue thinking mode, increment count by new tokens
                state["think_count"] += len(new_tokens)

            # Set countdown based on current state
            if state["in_think"]:
                remaining_budget = max(
                    0, state["thinking_token_budget"] - state["think_count"]
                )
                state["check_count_down"] = remaining_budget
            else:
                state["check_count_down"] = state["thinking_token_budget"]

            # Check if need to transition to end mode
            if (
                state["in_think"]
                and state["think_count"] >= state["thinking_token_budget"]
            ):
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                state["check_count_down"] = state["thinking_token_budget"]
        else:
            # In end mode
            state["end_count"] += 1
            if state["end_count"] >= len(self.think_end_token_ids):
                state.update(
                    {
                        "in_end": False,
                        "end_count": 0,
                        "check_count_down": state["thinking_token_budget"],
                    }
                )

    def is_argmax_invariant(self) -> bool:
        """This logits processor can change the outcome of
        greedy sampling by forcing that the thinking section
        ends after a certain number of tokens."""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not self.is_enabled:
            return
        if batch_update:
            for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
                thinking_token_budget = params.thinking_token_budget

                if thinking_token_budget is not None:
                    self._state[index] = self._init_state_entry(
                        prompt_tok_ids, thinking_token_budget
                    )
                    self._state[index]["output_tok_ids"] = output_tok_ids
                else:
                    # Remove state if no thinking budget
                    self._state.pop(index, None)

            for index in batch_update.removed:
                self._state.pop(index, {})

            for i1, i2, direction in batch_update.moved:
                if direction == MoveDirectionality.SWAP:
                    state1 = self._state.get(i1, {})
                    state2 = self._state.get(i2, {})
                    if state1 or state2:
                        self._state[i1] = state2
                        self._state[i2] = state1
                else:
                    self._state[i2] = self._state.pop(i1, {})

        for state in self._state.values():
            self._update_think_state(state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.is_enabled or not self._state:
            return logits

        batch_size = logits.size(0)
        self.mask[:batch_size] = False

        for i in range(batch_size):
            state = self._state.get(i)
            if state and state["in_end"]:
                self.mask[i] = True
                self.force_token_ids[i] = self.think_end_token_ids[state["end_count"]]

        # Check in CPU first not to sync with GPU
        has_active_thinking = any(
            state.get("in_end", False) for state in self._state.values()
        )

        if has_active_thinking:
            current_mask = self.mask[:batch_size]
            active_indices = current_mask.nonzero(as_tuple=False).view(-1)
            if len(active_indices) > 0:
                force_tokens = self.force_token_ids[active_indices]
                # Apply a large value for the end thinking token id index
                logits[active_indices, force_tokens] = 1e9

        return logits


def process_dict_updates(
    req_entries: dict[int, T],
    batch_update: Optional[BatchUpdate],
    new_state: Callable[[SamplingParams, Optional[list[int]], list[int]], Optional[T]],
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
