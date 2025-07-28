# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from typing import Optional, Union

import torch
from torch._prims_common import DeviceLikeType

from vllm import PoolingParams, SamplingParams
from vllm.logger import init_logger

logger = init_logger(__name__)


class MoveDirectionality(Enum):
    # One-way i1->i2 req move within batch
    UNIDIRECTIONAL = 0
    # Two-way i1<->i2 req swap within batch
    SWAP = 1


# (index, params, output_tok_ids) tuples for new
# requests added to the batch.
AddedRequest = tuple[int, Union[SamplingParams, PoolingParams], list[int]]
# (index 1, index 2, directionality) tuples representing
# one-way moves or two-way swaps of requests in batch
MovedRequest = tuple[int, int, MoveDirectionality]
# Batch indices of any removed requests.
RemovedRequest = int


@dataclasses.dataclass(frozen=True)
class BatchUpdate:
    """Persistent batch state change info for logitsprocs"""
    batch_size: int  # Current num reqs in batch

    # Metadata for requests added to, removed from, and moved
    # within the persistent batch.
    #
    # Note: each added request is represented as
    # (index, params, output_tok_ids)
    # Key assumption: output_tok_ids is a reference to the
    # request's running output tokens list; in this way
    # the logits processors always see the latest list of
    # generated tokens
    removed: Sequence[RemovedRequest]
    moved: Sequence[MovedRequest]
    added: Sequence[AddedRequest]


class BatchUpdateBuilder:
    """Helps track persistent batch state changes and build
    a batch update data structure for logitsprocs
    
    Assumptions:
    * All information about requests removed from persistent batch
      during a step is aggregated in self._removed through calls to
      self.removed_append() at the beginning of a step. This must happen
      before the first time that self.removed, self.pop_removed()
      or self.peek_removed() are invoked in a given step
    * After the first time that self.removed, self.pop_removed()
      or self.peek_removed() are read in a step, no new removals
      are registered using self.removed_append()
    * Elements of self._removed are never directly modified, added or
      removed (i.e. modification is only via self.removed_append() and
      self.pop_removed())
    
    Guarantees under above assumptions:
    * self.removed is always sorted in descending order
    * self.pop_removed() and self.peek_removed() both return
      the lowest removed request index in the current step
    """

    _removed: list[RemovedRequest]
    _is_removed_sorted: bool
    moved: list[MovedRequest]
    added: list[AddedRequest]

    def __init__(
        self,
        removed: Optional[list[RemovedRequest]] = None,
        moved: Optional[list[MovedRequest]] = None,
        added: Optional[list[AddedRequest]] = None,
    ) -> None:
        self._removed = removed or []
        self.moved = moved or []
        self.added = added or []
        self._is_removed_sorted = False

    def _ensure_removed_sorted(self) -> None:
        """Sort removed request indices in
        descending order.
        
        Idempotent after first call in a
        given step, until reset.
        """
        if not self._is_removed_sorted:
            self._removed.sort(reverse=True)
            self._is_removed_sorted = True

    @property
    def removed(self) -> list[RemovedRequest]:
        """Removed request indices sorted in
        descending order"""
        self._ensure_removed_sorted()
        return self._removed

    def removed_append(self, index: int) -> None:
        """Register the removal of a request from
        the persistent batch.

        Must not be called after the first time
        self.removed, self.pop_removed() or
        self.peek_removed() are invoked.
        
        Args:
          index: request index
        """
        if self._is_removed_sorted:
            raise RuntimeError("Cannot register new removed request after"
                               " self.removed has been read.")
        self._removed.append(index)

    def has_removed(self) -> bool:
        return bool(self._removed)

    def peek_removed(self) -> Optional[int]:
        """Return lowest removed request index"""
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed[-1]
        return None

    def pop_removed(self) -> Optional[int]:
        """Pop lowest removed request index"""
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed.pop()
        return None

    def get_and_reset(self, batch_size: int) -> Optional[BatchUpdate]:
        """Generate a logitsprocs batch update data structure
        and reset internal batch update builder state.
        
        Args:
          batch_size: current persistent batch size

        Returns:
          Frozen logitsprocs batch update instance; `None` if no updates
        """
        # Reset removal-sorting logic
        self._is_removed_sorted = False
        if not any((self._removed, self.moved, self.added)):
            # No update; short-circuit
            return None
        # Build batch state update
        batch_update = BatchUpdate(
            batch_size=batch_size,
            removed=self._removed,
            moved=self.moved,
            added=self.added,
        )
        # Reset removed/moved/added update lists
        self._removed = []
        self.moved = []
        self.added = []
        return batch_update


class LogitsProcessor(ABC):

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        """True if logits processor has no impact on the
        argmax computation in greedy sampling.
        NOTE: may or may not have the same value for all
        instances of a given LogitsProcessor subclass,
        depending on subclass implementation.
        TODO(andy): won't be utilized until logits
        processors are user-extensible
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self,
        batch_update: Optional[BatchUpdate],
    ) -> None:
        """Called when there are new output tokens, prior
        to each forward pass.

        Args:
            batch_update is non-None iff there have been
            changes to the batch makeup.
        """
        raise NotImplementedError


@dataclass
class LogitsProcessorManager:
    """Encapsulates initialized logitsproc objects."""
    argmax_invariant: list[LogitsProcessor] = field(
        default_factory=list)  # argmax-invariant logitsprocs
    non_argmax_invariant: list[LogitsProcessor] = field(
        default_factory=list)  # non-argmax-invariant logitsprocs

    @property
    def all(self) -> Iterator[LogitsProcessor]:
        """Iterator over all logits processors."""
        return chain(self.argmax_invariant, self.non_argmax_invariant)


###### ----- Built-in LogitsProcessor impls below here


class MinPLogitsProcessor(LogitsProcessor):

    def __init__(self, max_num_reqs: int, pin_memory: bool,
                 device: DeviceLikeType):
        super().__init__()
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        self.use_double_tensor = torch.device("cpu") != torch.device(device)

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
        for index, params, _ in batch_update.added:
            min_p = params.min_p if isinstance(params, SamplingParams) else 0.0
            if self.min_p_cpu[index] != min_p:
                needs_update = True
                self.min_p_cpu[index] = min_p
            if min_p:
                self.min_p_count += 1

        if self.min_p_count:
            # Process removed requests.
            needs_update |= bool(batch_update.removed)
            for index in batch_update.removed:
                if self.min_p_cpu[index]:
                    self.min_p_count -= 1

            # Process moved requests, unidirectional (a->b) and swap (a<->b)
            for adx, bdx, direct in batch_update.moved:
                change = (min_p_a :=
                          self.min_p_cpu[adx]) != (min_p_b :=
                                                   self.min_p_cpu[bdx])
                needs_update |= change
                if change:
                    self.min_p_cpu[bdx] = min_p_a
                    if direct == MoveDirectionality.SWAP:
                        self.min_p_cpu[adx] = min_p_b

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

    def __init__(self, pin_memory: bool, device: torch.device):
        super().__init__()
        self.biases: dict[int, dict[int, float]] = {}
        self.device = device
        self.pin_memory = pin_memory

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
        for index, params, _ in batch_update.added:
            if isinstance(params, SamplingParams) and (lb :=
                                                       params.logit_bias):
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

    def __init__(self, pin_memory: bool, device: torch.device):
        # index -> (min_toks, output_token_ids, stop_token_ids)
        super().__init__()
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}
        self.device = device
        self.pin_memory = pin_memory

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
            for index, params, output_tok_ids in batch_update.added:
                if (isinstance(params, SamplingParams)
                        and (min_tokens := params.min_tokens)
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


def init_builtin_logitsprocs(pin_memory_available: bool, max_num_reqs: int,
                             device: torch.device) -> LogitsProcessorManager:
    """Construct 'builtin' vLLM logitsprocs which the engine
    loads by default.

    Args:
      pin_memory_available: pinned memory is available for use
                            for use by logitsproc
      max_num_reqs: ceiling on request count in persistent batch
      device: inference device

    Returns:
      Data structure encapsulating loaded logitsprocs
    """
    min_tokens_logitproc = MinTokensLogitsProcessor(
        pin_memory=pin_memory_available, device=device)
    logit_bias_logitproc = LogitBiasLogitsProcessor(
        pin_memory=pin_memory_available, device=device)
    min_p_logitproc = MinPLogitsProcessor(
        pin_memory=pin_memory_available,
        device=device,
        # +1 for temporary swap space
        max_num_reqs=max_num_reqs + 1)
    return LogitsProcessorManager(
        non_argmax_invariant=[
            min_tokens_logitproc,
            logit_bias_logitproc,
        ],
        argmax_invariant=[min_p_logitproc],
    )
