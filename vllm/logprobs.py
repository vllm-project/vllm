# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Iterable
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


class FlattenLogprobs:
    """
    Flatten logprobs of a request into multiple primitive type lists.

    Compared to list[dict[int, Logprob]], this data structure reduced GC
    overhead significantly. As it flattened logprob information for
    all positions and ranks in to multiple primitive type lists (i.e.
    logprobs, token_ids, ranks per token_ids, decoded_tokens).
    So regardless of the sequence length and top_logprobs setup,
    FlattenLogprobs would only introduce a constant amount of objects.

    As each position might contains different amount of ranks,
    start_indices_per_position would be used to access the logprob ranges
    for different positions.
    """

    def __init__(self, for_prompt: bool) -> None:
        # The start indices of logprobs for each position.
        self.start_indices_per_position: list[int] = [0, 0] if for_prompt else [0]

        # Flatten Logprob information for (each position, rank).
        # For position <i>, the logprobs are ranged
        # from self.start_indices_per_position[i]
        # to self.start_indices_per_position[i+1] (exclusive).
        self.logprobs: list[float] = []
        self.token_ids: list[int] = []
        self.ranks: list[int] = []
        self.decoded_tokens: list[str | None] = []

    def append_logprobs_for_next_position(
        self,
        logprobs: list[float],
        token_ids: list[int],
        decoded_tokens: Iterable[str | None],
        rank: int,
        num_logprobs: int,
    ) -> None:
        if num_logprobs == -1:
            num_logprobs = len(logprobs)
        topk_ranks = range(1, num_logprobs + 1)
        ranks = itertools.chain((rank,), topk_ranks)

        for logprob, token_id, rank, decoded_token in zip(
            logprobs, token_ids, ranks, decoded_tokens
        ):
            self.logprobs.append(logprob)
            self.token_ids.append(token_id)
            self.ranks.append(rank)
            self.decoded_tokens.append(decoded_token)

        self.start_indices_per_position.append(len(self.logprobs))

    def get_logprob(self, position: int, token_id: int) -> float:
        for i in range(
            self.start_indices_per_position[position],
            self.start_indices_per_position[position + 1],
        ):
            if self.token_ids[i] == token_id:
                return self.logprobs[i]
        raise ValueError(f"{token_id=} not found in {position=}.")

    # TODO(Jialin): add more helper functions here to extract logprob info
    # from this flatten structure.

    # TODO(Jialin): Maybe add a helper function to convert the structure
    # back to the original PromptLogprobs or SampleLogprobs structure.


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = FlattenLogprobs | list[dict[int, Logprob] | None]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = FlattenLogprobs | list[dict[int, Logprob]]
