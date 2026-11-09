# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Oracle:
    token_ids: tuple[int, ...]
    text: str
    sampled_token_logprob: float

    def __post_init__(self) -> None:
        if len(self.token_ids) != 1:
            raise ValueError("snapshot oracle must contain exactly one token")
        value = self.sampled_token_logprob
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("sampled token logprob must be finite")
        object.__setattr__(self, "sampled_token_logprob", float(value))


def oracles_match(expected: Oracle, actual: Oracle) -> bool:
    return (
        expected.token_ids == actual.token_ids
        and expected.text == actual.text
        and math.isclose(
            expected.sampled_token_logprob,
            actual.sampled_token_logprob,
            rel_tol=0.0,
            abs_tol=1e-3,
        )
    )
