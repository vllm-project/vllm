import math
from abc import abstractmethod, ABC
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
    def complete_score(self, request: "Request", partial_score: float, current_time: float) -> float:
        pass


class SimpleScoreCalculator(AbstractScoreCalculator):
    def get_partial_score(self, request: "Request", boundary_step: float) -> float:
        """
        Calculates the part of the score dependent on request length and queue position.
        """
        # EWSJF MODIFICATION: Get prompt token IDs directly from the Request object.
        length = len(request.prompt_token_ids)
        queue_index = length // boundary_step
        fairness_factor: float = 1 + self.weighting_factor * math.log(length + 1)
        queue_index_factor: float = boundary_step / (queue_index + 1)
        # print(
        #     f'calculate partial score. length: {length}, fairness factor: {fairness_factor}, queue_index_factor: {queue_index_factor}')

        return fairness_factor * queue_index_factor

    def complete_score(self, request: "Request", partial_score: float, current_time: float) -> float:
        """
        Calculates the final score by combining the partial score with the dynamic wait time.
        """
        normalized_cost: float = 1.0  # Placeholder

        # EWSJF MODIFICATION: Get arrival_time directly from the Request object.
        wait_time: float = current_time - request.arrival_time
        base_score: float = wait_time / normalized_cost
        # print(f'length: {len(request.prompt_token_ids)}, partial_score: {partial_score}, base_score: {base_score}')

        return base_score * partial_score