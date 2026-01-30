# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import issues
if TYPE_CHECKING:
    # EWSJF MODIFICATION: We now expect a Request object, not a SequenceGroup.
    from vllm.v1.request import Request


class AbstractScoreCalculator(ABC):
    def __init__(self, weighting_factor: float = 0.5):
        self.weighting_factor: float = weighting_factor

    @abstractmethod
    def get_partial_score(self, request: "Request", boundary_step: float) -> float:
        pass

    @abstractmethod
    def complete_score(
        self, request: "Request", partial_score: float, current_time: float
    ) -> float:
        pass


class SimpleScoreCalculator(AbstractScoreCalculator):
    def get_partial_score(self, request: "Request", boundary_step: float) -> float:
        """
        Calculates the part of the score dependent on request length and queue
        position.
        """
        # EWSJF MODIFICATION: Get prompt token IDs directly from the Request.
        # Fix: Handle potential None type for prompt_token_ids
        length = len(request.prompt_token_ids or [])

        # Avoid division by zero if boundary_step is 0
        if boundary_step == 0:
            queue_index = 0.0
            queue_index_factor = 1.0
        else:
            queue_index = length // boundary_step
            queue_index_factor = boundary_step / (queue_index + 1)

        fairness_factor: float = 1 + self.weighting_factor * math.log(length + 1)

        return fairness_factor * queue_index_factor

    def complete_score(
        self, request: "Request", partial_score: float, current_time: float
    ) -> float:
        """
        Calculates the final score by combining the partial score with the
        dynamic wait time.
        """
        normalized_cost: float = 1.0  # Placeholder

        # EWSJF MODIFICATION: Get arrival_time directly from the Request object
        wait_time: float = current_time - request.arrival_time
        base_score: float = wait_time / normalized_cost

        return base_score * partial_score
