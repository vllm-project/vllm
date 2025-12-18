# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from functools import total_ordering

from vllm.v1.core.sched.policy.normalized_scorer import TimeAndLengthScorer
from vllm.v1.request import Request

TimeAndLengthScorer_Instance = None

if TimeAndLengthScorer_Instance is None:
    TimeAndLengthScorer_Instance = TimeAndLengthScorer(
        time_median=5,
        time_weight=0.5,
        length_median=32 * 1024,
        length_weight=0.5,
        reverse_len=True,
    )


@total_ordering
class WeightedScoreSorter:
    def __init__(self, request: Request):
        self.request = request
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
        return self.weighted_score == other_request_weighted_score.weighted_score

    def __update_stats(self):
        self.wait_time = time.time() - self.request_arrival_time
        assert TimeAndLengthScorer_Instance is not None
        self.weighted_score = TimeAndLengthScorer_Instance.score(
            self.wait_time, self.request_length
        )
