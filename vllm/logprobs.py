# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass


# We use dataclass for now because it is used for
# openai server output, and msgspec is not serializable.
# TODO(sang): Fix it.
@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = list[dict[int, Logprob] | None]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = list[dict[int, Logprob]]
