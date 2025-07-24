# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

import torch

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MoveDirectionality(Enum):
    # One-way i1->i2 req move within batch
    UNIDIRECTIONAL = 0
    # Two-way i1<->i2 req swap within batch
    SWAP = 1


# (prompt_tok_ids, index, params, output_tok_ids) tuples for new
# requests added to the batch.
AddedRequest = tuple[list[int], int, Union[SamplingParams], list[int]]

# (index 1, index 2, directionality) tuples representing
# one-way moves or two-way swaps of requests in batch
MovedRequest = tuple[int, int, MoveDirectionality]

# Batch indices of any removed requests.
RemovedRequest = int


@dataclass(frozen=True)
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


class LogitsProcessor(ABC):

    @abstractmethod
    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool) -> None:
        raise NotImplementedError

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
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self,
        batch_update: Optional["BatchUpdate"],
    ) -> None:
        """Called when there are new output tokens, prior
        to each forward pass.

        Args:
            batch_update is non-None iff there have been
            changes to the batch makeup.
        """
        raise NotImplementedError
