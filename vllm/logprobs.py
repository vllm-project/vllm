# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import vllm.envs as envs


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
        # NOTE: There's no logprob of first prompt token, so if the
        # FlattenLogprobs is for prompt, we directly set
        # the start index for the second prompt token.
        self.start_indices_per_position: list[int] = [0, 0] if for_prompt else [0]

        # Flatten Logprob information for (each position, rank).
        # For position <i>, the logprobs are ranged
        # from self.start_indices_per_position[i]
        # to self.start_indices_per_position[i+1] (exclusive).
        self.logprobs: list[float] = []
        self.token_ids: list[int] = []
        self.ranks: list[int] = []
        self.decoded_tokens: list[str | None] = []

    def num_positions(self) -> None:
        """Gets number of positions stored in the container"""
        return len(self.start_indices_per_position) - 1

    def num_logprobs_per_position(self, position: int) -> int:
        """Gets the number of logprobs of a given position"""
        return (
            self.start_indices_per_position[position + 1]
            - self.start_indices_per_position[position]
        )

    def logprob(self, position: int, token_id: int) -> float | None:
        """Gets logprob value of a given position and token ID."""
        idx = self._idx(position, token_id)
        if idx is not None:
            return self.logprobs[idx]
        return None

    def rank(self, position: int, token_id: int) -> int | None:
        """Gets rank of a given position and token ID."""
        idx = self._idx(position, token_id)
        if idx is not None:
            return self.ranks[idx]
        return None

    def _idx(self, position: int, token_id: int) -> int | None:
        """Locates the index of a given position and token ID."""
        return next(
            (
                idx
                for idx in range(
                    self.start_indices_per_position[position],
                    self.start_indices_per_position[position + 1],
                )
                if self.token_ids[idx] == token_id
            ),
            None,
        )


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = FlattenLogprobs | list[dict[int, Logprob] | None]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = FlattenLogprobs | list[dict[int, Logprob]]


def create_prompt_logprobs() -> PromptLogprobs:
    """Creates a container to store prompt logprobs for a request"""
    # NOTE: logprob of first prompt token is None.
    return FlattenLogprobs(for_prompt=True) if envs.VLLM_FLATTEN_LOGPROBS else [None]


def create_sample_logprobs() -> SampleLogprobs:
    """Creates a container to store decode logprobs for a request"""
    return FlattenLogprobs(for_prompt=False) if envs.VLLM_FLATTEN_LOGPROBS else []


def num_positions(logprobs: PromptLogprobs | SampleLogprobs) -> int:
    """Gets the number of positions stored in the logprobs"""
    return (
        logprobs.num_positions()
        if isinstance(logprobs, FlattenLogprobs)
        else len(logprobs)
    )


def num_logprobs_per_position(
    logprobs: PromptLogprobs | SampleLogprobs, position: int
) -> int:
    """Gets the number of logprobs of a given position"""
    if isinstance(logprobs, FlattenLogprobs):
        return logprobs.num_logprobs_per_position(position)
    return len(logprobs[position]) if logprobs[position] is not None else 0


def get_logprob(
    logprobs: PromptLogprobs | SampleLogprobs, position: int, token_id: int
) -> int | None:
    """Gets the logprob value of a given position and token ID"""
    if isinstance(logprobs, FlattenLogprobs):
        return logprobs.logprob(position, token_id)
    if (single_position_logprobs := logprobs[position]) and (
        logprob := single_position_logprobs.get(token_id)
    ):
        return logprob.logprob
    return None


def get_rank(
    logprobs: PromptLogprobs | SampleLogprobs, position: int, token_id: int
) -> int | None:
    """Gets the rank of a given position and token ID"""
    if isinstance(logprobs, FlattenLogprobs):
        return logprobs.rank(position, token_id)
    if (single_position_logprobs := logprobs[position]) and (
        logprob := single_position_logprobs.get(token_id)
    ):
        return logprob.rank
    return None


def append_logprobs_for_next_position(
    request_logprobs: PromptLogprobs | SampleLogprobs,
    logprobs: list[float],
    logprob_token_ids: list[int],
    decoded_tokens: Iterable[str | None],
    rank: int,
    num_logprobs: int,
) -> None:
    """Appends logprobs for the next position.

    Args:
        request_logprobs: target request-level logprobs container
        logprobs: list of log probabilities
        logprob_token_ids: list of top token ids
        decoded_tokens: list of decoded top tokens
        rank: rank of the sampled token
        num_logprobs: number of logprobs requested by the user
                      (in addition to sampled logprob)
    """
    if num_logprobs == -1:
        num_logprobs = len(logprobs)
    # We do not need a special case for the sampled token
    # being in the topk, since inserting duplicated data
    # into a dictionary twice is the same as doing it once.
    topk_ranks = range(1, num_logprobs + 1)
    ranks = itertools.chain((rank,), topk_ranks)

    if isinstance(request_logprobs, FlattenLogprobs):
        for token_id, logprob, rank, token in zip(
            logprob_token_ids, logprobs, ranks, decoded_tokens
        ):
            request_logprobs.logprobs.append(logprob)
            request_logprobs.token_ids.append(token_id)
            request_logprobs.ranks.append(rank)
            request_logprobs.decoded_tokens.append(token)

        request_logprobs.start_indices_per_position.append(
            len(request_logprobs.logprobs)
        )
    else:
        request_logprobs.append(
            {
                token_id: Logprob(
                    logprob=logprob,
                    rank=rank,
                    decoded_token=token,
                )
                for token_id, logprob, rank, token in zip(
                    logprob_token_ids, logprobs, ranks, decoded_tokens
                )
            }
        )
