# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Generic, Optional, TypeVar

import torch

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig

T = TypeVar("T")


class MoveDirectionality(Enum):
    # One-way i1->i2 req move within batch
    UNIDIRECTIONAL = auto()
    # Two-way i1<->i2 req swap within batch
    SWAP = auto()


# Batch indices of any removed requests.
RemovedRequest = int

# (index, params, prompt_tok_ids, output_tok_ids) tuples for new
# requests added to the batch.
AddedRequest = tuple[int, SamplingParams, Optional[list[int]], list[int]]

# (index 1, index 2, directionality) tuples representing
# one-way moves or two-way swaps of requests in batch
MovedRequest = tuple[int, int, MoveDirectionality]


@dataclass(frozen=True)
class BatchUpdate:
    """Persistent batch state change info for logitsprocs"""
    batch_size: int  # Current num reqs in batch

    # Metadata for requests added to, removed from, and moved
    # within the persistent batch.
    #
    # Key assumption: the `output_tok_ids` list (which is an element of each
    # tuple in `added`) is a reference to the request's running output tokens
    # list; via this reference, the logits processors always see the latest
    # list of generated output tokens.
    #
    # NOTE:
    # * Added or moved requests may replace existing requests with the same
    #   index.
    # * Operations should be processed in the following order:
    #   - removed, added, moved
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]


class LogitsProcessor(ABC, Generic[T]):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool) -> None:

        # Per-request logits processor state
        self.states: dict[int, T] = {}

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply LogitsProcessor to batch logits tensor.

        The updated tensor must be returned but may be
        modified in-place.
        """
        raise NotImplementedError

    def is_argmax_invariant(self) -> bool:
        """True if logits processor has no impact on the argmax computation in
        greedy sampling; causes logits processor to be optimized away in greedy
        sampling scenarios. Base-class default is false but can be overriden by
        subclass.
        NOTE: may or may not have the same value for all
        instances of a given LogitsProcessor subclass,
        depending on subclass implementation.
        """
        return False

    @abstractmethod
    def get_state_from_params(self, params: SamplingParams,
                              prompt_tok_ids: list[int],
                              out_tok_ids: list[int]) -> Optional[T]:
        """Produce a minimal representation of initial logits processor state
        for a newly-added request
        
        Args:
            params: `SamplingParams` instance for request newly-added to batch
            prompt_tok_ids: list of new request prompt token ids
            out_tok_ids: list of request generated tokens as of current engine
                         step

        Returns:
            `None` if logits processor is not applicable to request; otherwise,
            instance of initial logits processor state representation
        """
        raise NotImplementedError

    def state_update_callback(self) -> None:
        """Override to implement specialized optimizations to logits processor
        state management."""
        pass

    def update_state(
        self,
        batch_update: Optional["BatchUpdate"],
    ) -> None:
        """Called when there are new output tokens, prior
        to each forward pass.

        Args:
            batch_update: Non-None iff there have been changes
                to the batch makeup.
        """
        needs_update = process_dict_updates(self.states, batch_update,
                                            self.get_state_from_params)

        if needs_update:
            # Apply custom
            self.state_update_callback()


def process_dict_updates(
    req_entries: dict[int, T], batch_update: Optional[BatchUpdate],
    new_state: Callable[[SamplingParams, Optional[list[int]], list[int]],
                        Optional[T]]
) -> bool:
    """Utility function to update dict state for sparse LogitsProcessors."""

    if not batch_update:
        # Nothing to do.
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        if (state := new_state(params, prompt_tok_ids,
                               output_tok_ids)) is not None:
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
