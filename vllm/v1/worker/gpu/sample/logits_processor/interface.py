# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class LogitsProcessor(ABC):
    """Custom logits processor for Model Runner V2.

    This is a standalone contract, separate from the V1
    ``vllm.v1.sample.logits_processor.LogitsProcessor``: the two are not
    interchangeable, because V2 reorders logits rows every step and expands
    them under speculative decoding, and its persistent batch has a different
    state lifecycle (no ``BatchUpdate`` ledger).

    The V2 persistent batch keeps per-request state in fixed buffers keyed by
    the request slot index, and slots are never moved or compacted. The
    lifecycle is:

    * ``add_request()`` initializes per-slot state when a request enters
      the batch.
    * ``remove_request()`` tears it down when a request leaves. Slots are
      recycled through a free list, so without this hook state keyed by slot
      would leak.
    * ``update_state()`` is an optional per-step hook called before each
      logits processing pass.

    ``apply()`` receives the row-to-slot mapping because a logits row is
    not, in general, one per request: rows are reordered per step, and with
    speculative decoding a request owns one row per draft token
    (``expanded_idx_mapping`` maps each row to its slot).

    Under speculative decoding, processors observe the expanded draft rows,
    unlike V1 where custom processors were disabled.

    Grammar bitmasks are applied to the logits before custom processors, so
    processors must not re-inflate grammar-masked tokens.
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        # Concrete default; subclasses may override to capture the config.
        pass

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        """Whether ``apply()`` leaves the per-row argmax unchanged.

        The sampler applies argmax-invariant processors only on the sampling
        path (after temperature, before top-k/top-p) and skips them on the
        greedy path.
        """

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        """Initialize per-slot state for a request entering the batch."""

    def remove_request(self, req_idx: int) -> None:
        """Tear down per-slot state for a request leaving the batch."""

    def should_apply(self, sampling_params: SamplingParams) -> bool:
        """Whether this processor modifies logits for requests with these
        params. Called once when a request enters the batch; the result
        feeds the per-request ``needs_logits_processing`` flag, which lets
        the sampler skip the logits-processing pipeline for batches where
        no request needs it. Defaults to True (all requests).

        Note: this gates pipeline admission only. When the pipeline runs
        for other requests, ``apply()`` still sees every row, so processors
        must filter rows themselves via ``expanded_idx_mapping``."""
        return True

    def update_state(self) -> None:
        """Optional per-step hook, called before each logits processing
        pass when any request in the batch needs logits processing."""

    @abstractmethod
    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        """Modify logits in place or return a new tensor.

        Args:
            logits: [num_logits_rows, vocab_size] float32 tensor.
            expanded_idx_mapping: [num_logits_rows] int32 device tensor
                mapping each logits row to its persistent request slot.
            idx_mapping_np: [num_reqs] CPU array mapping each request in
                the batch to its persistent slot index.
        """
        raise NotImplementedError
