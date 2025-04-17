# SPDX-License-Identifier: Apache-2.0
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

import torch
from torch._prims_common import DeviceLikeType

from vllm import SamplingParams


@dataclasses.dataclass
class BatchUpdate:
    # The current number of requests in the batch.
    batch_size: int
    # Batch indices of any removed requests.
    removed: Sequence[int] = ()
    # (from, to) batch indices of any requests
    # moved within the batch.
    moved: Sequence[tuple[int, int]] = ()
    # (index, params, output_tok_ids) for new
    # requests added to the batch.
    added: Sequence[tuple[int, SamplingParams, list[int]]] = ()


class LogitsProcessor(ABC):

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def update_states(
        self,
        batch_update: Optional[BatchUpdate] = None,
    ) -> None:
        """Called when there are new output tokens, prior
        to each forward pass.

        Args:
            batch_update is non-None iff there have been
            changes to the batch makeup.
        """
        raise NotImplementedError


###### ----- LogitsProcessor impls below here


class MinPLogitsProcessor(LogitsProcessor):

    def __init__(self, max_num_reqs: int, pin_memory: bool,
                 device: DeviceLikeType):
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()
        # Pre-allocated device tensor
        self.min_p_gpu: torch.Tensor = torch.empty((max_num_reqs, ),
                                                   dtype=torch.float32,
                                                   device=device)
        # Current slice of the device tensor
        self.min_p: torch.Tensor = self.min_p_gpu[:0]

    def update_states(self, batch_update: Optional[BatchUpdate] = None):
        if not batch_update:
            return

        needs_update = False
        if self.min_p_count:
            # Process removed and moved requests.
            for index in batch_update.removed:
                if self.min_p_cpu[index]:
                    self.min_p_count -= 1
                    needs_update = True

            for from_index, to_index in batch_update.moved:
                min_p = self.min_p_cpu[from_index]
                self.min_p_cpu[to_index] = min_p
                if min_p:
                    needs_update = True

        # Process added requests.
        for index, sampling_params, _ in batch_update.added:
            min_p = sampling_params.min_p
            self.min_p_cpu[index] = min_p
            if min_p:
                self.min_p_count += 1
                needs_update = True

        # Update tensors if needed.
        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):

            self.min_p = self.min_p_gpu[:size]
            self.min_p.copy_(self.min_p_cpu_tensor[:size], non_blocking=True)
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

    def __init__(self, pin_memory: bool, device: torch.device):
        self.biases: dict[int, dict[int, float]] = {}
        self.device = device
        self.pin_memory = pin_memory

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (torch.tensor(
            ()), torch.tensor(()))

    def update_states(self, batch_update: Optional[BatchUpdate] = None):
        if not batch_update:
            return

        needs_update = False
        if self.biases:
            # Process removed and moved requests.
            for index in batch_update.removed:
                if self.biases.pop(index, None):
                    needs_update = True

            for from_index, to_index in batch_update.moved:
                if entry := self.biases.pop(from_index, None):
                    self.biases[to_index] = entry
                    needs_update = True

        # Process added requests.
        for index, sampling_params, _ in batch_update.added:
            if lb := sampling_params.logit_bias:
                self.biases[index] = lb
                needs_update = True

        # Update tensors if needed.
        if self.biases and needs_update:
            reqs, tok_ids, biases = [], [], []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._tensor(biases, torch.float32)
            self.logits_slice = (self._tensor(reqs, torch.int32),
                                 self._tensor(tok_ids, torch.int32))

    def _tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
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

    def __init__(self, pin_memory: bool, device: torch.device):
        # index -> (min_toks, output_token_ids, stop_token_ids)
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}
        self.device = device
        self.pin_memory = pin_memory

        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (torch.tensor(
            ()), torch.tensor(()))

    def update_states(self, batch_update: Optional[BatchUpdate] = None):
        needs_update = False
        if batch_update:
            if self.min_toks:
                # Process removed and moved requests.
                for index in batch_update.removed:
                    if self.min_toks.pop(index, None):
                        needs_update = True

                for from_index, to_index in batch_update.moved:
                    if entry := self.min_toks.pop(from_index, None):
                        self.min_toks[to_index] = entry
                        needs_update = True

            # Process added requests.
            for index, sampling_params, output_tok_ids in batch_update.added:
                if ((min_tokens := sampling_params.min_tokens)
                        and len(output_tok_ids) < min_tokens):
                    self.min_toks[index] = (min_tokens, output_tok_ids,
                                            sampling_params.all_stop_token_ids)
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
            if needs_update and self.min_toks:
                reqs: list[int] = []
                tok_ids: list[int] = []
                for req, (_, _, stop_tok_ids) in self.min_toks.items():
                    reqs.extend([req] * len(stop_tok_ids))
                    tok_ids.extend(stop_tok_ids)

                self.logits_slice = (self._tensor(reqs, torch.int32),
                                     self._tensor(tok_ids, torch.int32))

    def _tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            logits[self.logits_slice] = -float("inf")
        return logits
