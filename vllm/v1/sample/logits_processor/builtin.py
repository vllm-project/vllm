# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

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

    def update_state(
        self, 
        batch_update: BatchUpdate | None,
        spec_token_ids: list[list[int]] | None = None
    ):
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

    def update_state(
        self, 
        batch_update: BatchUpdate | None, 
        spec_token_ids: list[list[int]] | None = None
    ):
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

    def update_state(
        self, 
        batch_update: BatchUpdate | None,
        spec_token_ids: list[list[int]] | None = None
    ):
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
            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
        return logits


class ThinkingTokenBudgetLogitsProcessor(LogitsProcessor):
    """Limits the number of tokens allowed inside a 'thinking' section."""

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        reasoning_config = vllm_config.reasoning_config
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        if vllm_config.speculative_config:
            self.num_spec_tokens = \
            vllm_config.speculative_config.num_speculative_tokens
        else:
            self.num_spec_tokens = 0  # Default to 0 for non-speculative mode

        # Check if thinking is enabled
        self.is_enabled = (
            reasoning_config is not None and reasoning_config.is_thinking_enabled
        )

        self.think_start_token_ids = getattr(
            reasoning_config, "think_start_token_ids", []
        )
        print("This is the start thinking token", self.think_start_token_ids)
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
        if self.num_spec_tokens > 0:
            self.mask = torch.zeros(max_num_reqs * self.num_spec_tokens, dtype=torch.bool, device=device)
        else:
            self.mask = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

        if self.num_spec_tokens > 0:
            self.force_token_ids = torch.full(
                (max_num_reqs*self.num_spec_tokens,), -1, dtype=torch.long, device=device
            )
        else:
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
        self, prompt_tok_ids: list[int] | None, thinking_token_budget: int
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
            "spec_token_ids": [],
            "force_index": 0,
            "start_thinking": -1,
            "end_thinking": -1
            # Track previous output length for incremental updates
        }

    def check_sequence(self, output_token_ids, start_thinking_token_ids) -> int:
        return (len(output_token_ids) - len(start_thinking_token_ids)
                if output_token_ids and start_thinking_token_ids
                and len(start_thinking_token_ids) <= len(output_token_ids)
                and output_token_ids[-len(start_thinking_token_ids):] \
                == start_thinking_token_ids
                else -1)


    def _update_think_state(self, state: dict[str, Any]):
        """Updates the state based on newly generated output tokens."""

        if state["start_thinking"] == -1: 
            start_thinking = self.check_sequence(\
            state.get("output_tok_ids", []), self.think_start_token_ids)
            state["start_thinking"] = start_thinking
        total_tokens = len(state.get("output_tok_ids", [])) + len(state.get("spec_token_ids", []))
        
        if state["in_think"]:
            end_thinking = self.check_sequence(\
            state.get("output_tok_ids", []), self.think_end_token_ids)
            state["end_thinking"] = end_thinking
            return

        if not state.get("in_end", False) and \
        state.get("check_count_down", 0) > 0 and state["start_thinking"] > -1:
            state["check_count_down"] -= (total_tokens - state.get("prev_output_length", 0))
            state["prev_output_length"] = len(state.get("output_tok_ids", []))
            print("Comes here to check count down:", state["check_count_down"])
            return

        output = state.get("output_tok_ids", [])
        if not output:
            return

        # Track previous output length for incremental processing
        prev_length = state.get("prev_output_length", 0)
        current_length = len(output)

        if current_length <= prev_length:
            print("comes here")
            return

        # Process only newly added tokens
        new_tokens = output[prev_length:]
        state["prev_output_length"] = current_length

        # Check if new tokens contain think start or end sequences
        start_len = len(self.think_start_token_ids)
        end_len = len(self.think_end_token_ids)
        absolute_start_pos = state["start_thinking"]
        absolute_end_pos = state["end_thinking"]
        print("Comes here to update think state. Absolute start pos:", absolute_start_pos,)
        # Update state based on recent sequences
        if state["in_end"] and \
        self.think_end_token_ids[0] not in state.get("output_tok_ids", []):
            state["in_think"] = True
        if not state["in_end"]:
            if absolute_start_pos >= 0 and absolute_end_pos >= 0:
                if absolute_start_pos > absolute_end_pos:
                    # Case: ...<end>...<start>... - entering think mode
                    new_think_count = current_length - ((absolute_start_pos -1) + start_len)
                    state["in_think"] = True
                    state["think_count"] = new_think_count
                else:
                    # Case: ...<start>...<end>... - exiting think mode
                    state["in_think"] = False
                    state["think_count"] = 0
            elif absolute_start_pos >= 0:
                # Found think start - entering think mode
                absolute_start_pos = absolute_start_pos
                new_think_count = current_length - ((absolute_start_pos -1) + start_len)
                state["in_think"] = True
                state["think_count"] = new_think_count
                print("Entering think mode. Think count:", state["think_count"])
            elif absolute_end_pos >= 0:
                # Found think end - exiting think mode
                state["in_think"] = False
                state["think_count"] = 0
            elif state["in_think"]:
                # Continue thinking mode, increment count by new tokens
                state["think_count"] += len(new_tokens)

            # Set countdown based on current state
            print("Updating check count down. In think:", state["in_think"], "Think count:", state["think_count"])
            if state["in_think"]:
                remaining_budget = max(
                    0, state["thinking_token_budget"] - state["think_count"]
                )
                state["check_count_down"] = remaining_budget
                print("Comes here to set check count down for think mode:", state["check_count_down"])
            else:
                state["check_count_down"] = state["thinking_token_budget"]

            # Check if need to transition to end mode
            total_thinking_tokens = state["think_count"] + len(state["spec_token_ids"])
            if state["in_think"] and total_thinking_tokens >= state["thinking_token_budget"]:
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0 # make changes here
                state["check_count_down"] = state["thinking_token_budget"]
                print("The code comes here to transition to end mode. Total thinking tokens:", total_thinking_tokens, "Think count:", state["think_count"], "Spec token count:", len(state["spec_token_ids"]))
                # Calculate force_index: position within spec_token_ids where forcing starts
                # If we're already over budget without spec tokens, force from position 0
                if state["think_count"] >= state["thinking_token_budget"]:
                    state["force_index"] = 0
                else:
                    # Force from the position where budget is exceeded
                    remaining_budget = total_thinking_tokens - state["thinking_token_budget"]
                    state["force_index"] = len(state["spec_token_ids"]) - remaining_budget
        else:
            # In end mode
            state["end_count"] += 1
            state["force_index"] = 0
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

    def update_state(
        self, 
        batch_update: BatchUpdate | None, 
        spec_token_ids: list[list[int]] | None = None,
    ):
        if not self.is_enabled:
            return

        # Store the spec token IDs for use in apply()
        self._spec_token_ids = spec_token_ids
        if batch_update:
            for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
                thinking_token_budget = params.thinking_token_budget
                if thinking_token_budget is not None:
                    self._state[index] = self._init_state_entry(
                        prompt_tok_ids, thinking_token_budget
                    )
                    self._state[index]["output_tok_ids"] = output_tok_ids
                    
                    # Set spec_token_ids if available
                    if self._spec_token_ids and len(self._spec_token_ids[index]) > 0:
                        self._state[index]["spec_token_ids"] = self._spec_token_ids[index]
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

        # Reset mask
        self.mask[:] = False
        print(self.mask.shape, self.force_token_ids.shape)
        # Find active sequences and set up masking
        cumulative_offset = 0
        # print("Applying ThinkingTokenBudgetLogitsProcessor. Current state:", self._state)
        for seq_idx in sorted(self._state.keys()):
            state = self._state[seq_idx]
            if state and state.get("in_end", False):
                print("Comes here since state is ready")
                force_index = state.get("force_index", 0)
                if 0 <= force_index:
                    print("Force index:", force_index, "for sequence index:", seq_idx)
                    mask_idx = cumulative_offset + force_index
                    print("This is mask index calculated:", mask_idx, "Cumulative offset:", cumulative_offset)
                    print("This length of self mask:", len(self.mask), "This length of force token ids:", len(self.force_token_ids))
                    if mask_idx < len(self.mask):
                        print(f"Masking index {mask_idx} for sequence {seq_idx} at force index {force_index}")
                        self.mask[mask_idx] = True
                        end_count = state.get("end_count", 0)
                        if end_count < len(self.think_end_token_ids):
                            print(f"Masking index {mask_idx} for sequence {seq_idx} with end_count {end_count}")
                            self.force_token_ids[mask_idx] = self.think_end_token_ids[end_count]
            
            # Update cumulative offset based on spec token count for this sequence
            spec_tokens_count = len(state.get("spec_token_ids", [])) if state else 0
            cumulative_offset += spec_tokens_count + 1

        # Check in CPU first not to sync with GPU
        has_active_thinking = any(
            state.get("in_end", False) for state in self._state.values()
        )

        if has_active_thinking:
            current_mask = self.mask[:cumulative_offset]
            active_indices = current_mask.nonzero(as_tuple=False).view(-1)
            
            if len(active_indices) > 0:
                force_tokens = self.force_token_ids[active_indices]
                
                # Debug prints to diagnose the error
                print("Logits shape:", logits.shape)
                print("Active indices shape:", active_indices.shape)
                print("Active indices values:", active_indices)
                print("Force tokens shape:", force_tokens.shape)
                print("Force tokens values:", force_tokens)
                print("Max active index:", int(active_indices.max()) if len(active_indices) > 0 else "None")
                print("Max force token:", int(force_tokens.max()) if len(force_tokens) > 0 else "None")
                print("Min force token:", int(force_tokens.min()) if len(force_tokens) > 0 else "None")
                
                # Bounds checks
                assert int(active_indices.max()) < logits.shape[0], f"Index {int(active_indices.max())} >= {logits.shape[0]}"
                assert int(force_tokens.min()) >= 0, f"Token ID {int(force_tokens.min())} < 0"
                assert int(force_tokens.max()) < logits.shape[1], f"Token ID {int(force_tokens.max())} >= {logits.shape[1]}"
                
                # Apply a large value for the end thinking token id index
                logits[active_indices, force_tokens] = 1e9

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
