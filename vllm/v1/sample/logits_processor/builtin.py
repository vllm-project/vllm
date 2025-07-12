# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import torch

from vllm.v1.sample.logits_processor.interface import (BatchUpdate,
                                                       LogitsProcessor,
                                                       MoveDirectionality)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MinPLogitsProcessor(LogitsProcessor):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=is_pin_memory)
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            # Pre-allocated device tensor
            self.min_p_device: torch.Tensor = torch.empty((max_num_reqs, ),
                                                          dtype=torch.float32,
                                                          device=device)
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
                self.min_p.copy_(self.min_p_cpu_tensor[:size],
                                 non_blocking=True)
            self.min_p.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values,
                                       dim=-1,
                                       keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits[invalid_token_mask] = -float('inf')
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):

    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (self._device_tensor([], torch.int32),
                             self._device_tensor([], torch.int32))

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        needs_update: bool = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            if lb := params.logit_bias:
                self.biases[index] = lb
                needs_update = True
            else:
                # Drop biases metadata at batch index
                if self.biases.pop(index, None) is not None:
                    # If a new request replaces an old request which
                    # specified biases, we should update processor tensors
                    needs_update = True

        if self.biases:
            # Process removed requests.
            for index in batch_update.removed:
                if self.biases.pop(index, None):
                    needs_update = True

            # Process moved requests, unidirectional (a->b) and swap (a<->b)
            for a_index, b_index, direct in batch_update.moved:
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if (a_entry := self.biases.pop(a_index, None)) is None:
                        if self.biases.pop(b_index, None) is not None:
                            needs_update = True
                    else:
                        self.biases[b_index] = a_entry
                        needs_update = True
                else:
                    a_entry = self.biases.pop(a_index, None)
                    if (b_entry := self.biases.pop(b_index, None)) is not None:
                        self.biases[a_index] = b_entry
                        needs_update = True
                    if a_entry is not None:
                        self.biases[b_index] = a_entry
                        needs_update = True

        # Update tensors if needed.
        if needs_update:
            reqs, tok_ids, biases = [], [], []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (self._device_tensor(reqs, torch.int32),
                                 self._device_tensor(tok_ids, torch.int32))

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        # index -> (min_toks, output_token_ids, stop_token_ids)
        self.device = device
        self.pin_memory = is_pin_memory
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}

        # (req_idx_tensor,eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor,
                                 torch.Tensor] = (self._device_tensor(
                                     [], torch.int32),
                                                  self._device_tensor(
                                                      [], torch.int32))

    def is_argmax_invariant(self) -> bool:
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        needs_update = False

        if batch_update:
            # Process added requests.
            for index, params, _, output_tok_ids in batch_update.added:
                if ((min_tokens := params.min_tokens)
                        and len(output_tok_ids) < min_tokens):
                    # Replace request metadata at batch index
                    self.min_toks[index] = (min_tokens, output_tok_ids,
                                            params.all_stop_token_ids)
                    needs_update = True
                else:
                    # Drop min_toks metadata at batch index
                    if self.min_toks.pop(index, None) is not None:
                        # If a new request replaces an old request which
                        # specified min_toks, we should update processor tensors
                        needs_update = True

            if self.min_toks:
                # Process removed requests.
                for index in batch_update.removed:
                    if self.min_toks.pop(index, None):
                        needs_update = True

                # Process moved requests, unidirectional (a->b) and
                # swapped (a<->b)
                for a_index, b_index, direct in batch_update.moved:
                    if direct == MoveDirectionality.UNIDIRECTIONAL:
                        if (a_entry := self.min_toks.pop(a_index,
                                                         None)) is None:
                            if self.min_toks.pop(b_index, None) is not None:
                                needs_update = True
                        else:
                            self.min_toks[b_index] = a_entry
                            needs_update = True
                    else:
                        a_entry = self.min_toks.pop(a_index, None)
                        if (b_entry := self.min_toks.pop(b_index,
                                                         None)) is not None:
                            self.min_toks[a_index] = b_entry
                            needs_update = True
                        if a_entry is not None:
                            self.min_toks[b_index] = a_entry
                            needs_update = True

        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            to_remove = tuple(index for index, (min_toks, out_tok_ids,
                                                _) in self.min_toks.items()
                              if len(out_tok_ids) >= min_toks)
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

            self.logits_slice = (self._device_tensor(reqs, torch.int32),
                                 self._device_tensor(tok_ids, torch.int32))

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits[self.logits_slice] = -float("inf")
        return logits


class MaxThinkTokensLogitsProcessor(LogitsProcessor):
    """A logits processor that limits the maximum number of thinking tokens."""

    def __init__(self, reasoning_config: ReasoningConfig, pin_memory: bool, device: torch.device):
        """
        Args:
            think_start_token_id (int): Token ID for the start of thinking section.
            think_end_token_id (int): Token ID for the end of thinking section.
            pin_memory (bool): Whether to use pinned memory for tensors.
            device (torch.device): Device to use for tensor operations.
        """
        super().__init__()
        self.think_start_token_id = reasoning_config.think_start_token_id
        self.think_end_token_id = reasoning_config.think_end_token_id
        self.pin_memory = pin_memory
        self.device = device
        self._state = {}

    def _find_last_token_index(self, tokens, token_id):
        try:
            return len(tokens) - tokens[::-1].index(token_id) - 1
        except ValueError:
            return -1

    def is_argmax_invariant(self) -> bool:
        """This logits processor can change the outcome of greedy sampling
        by forcing that the thinking section ends after a certain number of tokens."""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if batch_update is None:
            return

        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            max_think_tokens = params.max_think_tokens if isinstance(params, SamplingParams) else None

            if max_think_tokens is None:
                continue

            last_think_start_idx = self._find_last_token_index(prompt_tok_ids, self.think_start_token_id)
            last_think_end_idx = self._find_last_token_index(prompt_tok_ids, self.think_end_token_id)

            in_think = False
            count = 0

            if last_think_start_idx > last_think_end_idx:
                in_think = True
                count = len(prompt_tok_ids) - (last_think_start_idx + 1)

            self._state[index] = {
                "in_think": in_think,
                "count": count,
                "prompt_tok_ids": prompt_tok_ids,
                "output_tok_ids": output_tok_ids,
                "max_think_tokens": max_think_tokens,
            }

        for index in batch_update.removed:
            self._state.pop(index, None)

        for i1, i2, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                self._state[i1], self._state[i2] = self._state[i2], self._state[i1]
            else:
                self._state[i2] = self._state.pop(i1, None)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)
        if batch_size == 0:
            return logits

        mask = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)
        end_token_id = self.think_end_token_id

        for index in range(batch_size):
            state = self._state.get(index, None)
            if not state or not state.get("output_tok_ids"):
                continue

            last_tok = state["output_tok_ids"][-1]
            in_think = state["in_think"]
            count = state["count"]

            if last_tok == self.think_start_token_id:
                in_think = True
                count = 0
            elif last_tok == self.think_end_token_id:
                in_think = False
                count = 0
            elif in_think:
                count += 1

            state["in_think"] = in_think
            state["count"] = count

            if state["in_think"] and state["count"] >= state["max_think_tokens"]:
                mask[index] = True

        if mask.any():
            logits[mask] = -float("inf")
            logits[mask, end_token_id] = 0.0

        return logits
