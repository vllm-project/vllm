# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from functools import total_ordering
from typing import Any

from vllm.v1.core.sched.policy.normalized_scorer import TimeAndLengthScorer

TimeAndLengthScorer_Instance = None

if TimeAndLengthScorer_Instance == None:
    TimeAndLengthScorer_Instance = TimeAndLengthScorer(
        time_median=5,
        time_weight=0.5,
        length_median=32 * 1024,
        length_weight=0.5,
        reverse_len=True,
    )


@total_ordering
class WeightedScoreSorter:
    def __init__(
        self,
        request_length: int,
        request_arrival_time: float,
        request_slo_requirement: list[Any] | None = None,
    ):
        self.request_length = request_length
        self.request_arrival_time = request_arrival_time
        self.request_slo_requirement = request_slo_requirement
        self.__update_stats()

    def __lt__(self, other_request_weighted_score: "WeightedScoreSorter") -> bool:
        self.__update_stats()
        return self.weighted_score > other_request_weighted_score.weighted_score

    def __eq__(self, other_request_weighted_score: object) -> bool:
        if not isinstance(other_request_weighted_score, WeightedScoreSorter):
            return NotImplemented
        return self.weighted_score == other_request_weighted_score.weighted_score

    def __update_stats(self):
        self.wait_time = time.time() - self.request_arrival_time
        assert TimeAndLengthScorer_Instance is not None
        self.weighted_score = TimeAndLengthScorer_Instance.score(
            self.wait_time, self.request_length
        )
