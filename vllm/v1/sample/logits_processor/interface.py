# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.sample.logits_processor.state import BatchUpdate


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
