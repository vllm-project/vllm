# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from typing import TYPE_CHECKING, Optional, Union

from vllm import PoolingParams, SamplingParams

if TYPE_CHECKING:
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class MoveDirectionality(Enum):
    # One-way i1->i2 req move within batch
    UNIDIRECTIONAL = 0
    # Two-way i1<->i2 req swap within batch
    SWAP = 1


# (index, params, output_tok_ids, prompt_tok_ids) tuples for new
# requests added to the batch.
AddedRequest = tuple[int, Union[SamplingParams, PoolingParams], list[int],
                     list[int]]
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


class LogitsProcessors:
    """Encapsulates initialized logitsproc objects."""
    argmax_invariant: list["LogitsProcessor"] = field(
        default_factory=list, init=False)  # argmax-invariant logitsprocs
    non_argmax_invariant: list["LogitsProcessor"] = field(
        default_factory=list, init=False)  # non-argmax-invariant logitsprocs

    def __init__(
            self,
            logitsprocs: Optional[Iterator["LogitsProcessor"]] = None) -> None:
        self.argmax_invariant = []
        self.non_argmax_invariant = []
        if logitsprocs:
            for logitproc in logitsprocs:
                (self.argmax_invariant if logitproc.is_argmax_invariant() else
                 self.non_argmax_invariant).append(logitproc)

    @property
    def all(self) -> Iterator["LogitsProcessor"]:
        """Iterator over all logits processors."""
        return chain(self.argmax_invariant, self.non_argmax_invariant)
