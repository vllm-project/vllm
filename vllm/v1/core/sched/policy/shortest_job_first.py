# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import time
from functools import total_ordering

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)


class ScoreDim:
    """
    Normalized scoring dimension.
    """

    def __init__(
        self, name: str, median: float, norm_scale=0.0, weight=0.5, reverse=False
    ):
        self.name = name
        self.median = median
        if norm_scale != 0.0:
            self.norm_scale = norm_scale
        else:
            self.norm_scale = 1 / median
        self.weight = weight
        self.reverse = reverse


class NormalizedScorer:
    """
    Normalize unbounded N-dimensional values into a composite score using the Sigmoid
    function.
    """

    def __init__(self, dim_list: list[ScoreDim]) -> None:
        """
        Initialize the scorer with a list of scoring dimensions.

        Args:
            dim_list: A list of `ScoreDim` objects. Each dimension must define a
                      median reference point, scaling factor, and weight.
        """
        self.dim_list = dim_list
        self.dim_count = len(dim_list)

    @staticmethod
    def _sigmoid_normalize(value, median, norm_scale):
        """Sigmoid function: Maps value to (0, 1)."""
        return 1 / (1 + math.exp(-norm_scale * (value - median)))

    @staticmethod
    def _inv_sigmoid_normalize(value, median, norm_scale):
        """Inverse Sigmoid: Used for dimensions where a larger value yields a lower
        score.
        """
        # Equivalent to sigmoid(-x), but more numerically stable.
        return 1 / (1 + math.exp(norm_scale * (value - median)))

    def score(self, *dims: float) -> float:
        """
        Compute the composite score.
        Larger value → higher score → use forward Sigmoid.
        Smaller value → higher score → use inverse Sigmoid.
        """
        if len(dims) > self.dim_count:
            raise ValueError(
                f"Dim num({len(dims)}) exceeds max num dim({self.dim_count})"
            )

        final_score = 0.0
        for idx, dim_value in enumerate(dims):
            dim_info = self.dim_list[idx]
            if dim_info.reverse:
                score = self._inv_sigmoid_normalize(
                    dim_value, dim_info.median, dim_info.norm_scale
                )
            else:
                score = self._sigmoid_normalize(
                    dim_value, dim_info.median, dim_info.norm_scale
                )
            logger.debug("%s(%s) : %.10f", dim_info.name, dim_info.reverse, score)

            # Weighted summation.
            final_score += score * dim_info.weight
        return max(0.0, min(1.0, final_score))  # Clamp to [0, 1].


class TimeAndLengthScorer(NormalizedScorer):
    """
    Scorer for time and length dimensions; defaults to forward scoring with equal
    weights (0.5 each).
    """

    def __init__(
        self,
        time_median=5,
        length_median=1024 * 32,
        time_scale=0.0,
        length_scale=0.0,
        time_weight=0.5,
        length_weight=0.5,
        reverse_time=False,
        reverse_len=True,
    ) -> None:
        dim_list = [
            ScoreDim("time", time_median, time_scale, time_weight, reverse_time),
            ScoreDim("length", length_median, length_scale, length_weight, reverse_len),
        ]
        super().__init__(dim_list)

    def score(self, *dims: float) -> float:
        assert len(dims) == 2
        return super().score(*dims)


@total_ordering
class WeightedScoreSorter:
    def __init__(self, request: Request, scorer: TimeAndLengthScorer):
        self.request = request
        self.scorer = scorer
        assert request.prompt_token_ids is not None
        self.request_length = len(request.prompt_token_ids)
        self.request_arrival_time = request.arrival_time
        self.__update_stats()

    def __lt__(self, other_request_weighted_score: "WeightedScoreSorter") -> bool:
        self.__update_stats()
        return self.weighted_score > other_request_weighted_score.weighted_score

    def __eq__(self, other_request_weighted_score: object) -> bool:
        if not isinstance(other_request_weighted_score, WeightedScoreSorter):
            return NotImplemented
        return self.request.request_id == other_request_weighted_score.request.request_id

    def __update_stats(self):
        self.wait_time = time.time() - self.request_arrival_time
        self.weighted_score = self.scorer.score(self.wait_time, self.request_length)
