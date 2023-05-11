from typing import Dict, Set


class SamplingParams:

    def __init__(
        self,
        n: int = 1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        stop_token_ids: Set[int] = set(),
        max_tokens: int = 16,
        logprobs: int = 0,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")
        if not -2.0 <= presence_penalty <= 2.0:
            raise ValueError(
                f"presence_penalty must be in [-2, 2], got {presence_penalty}.")
        if not -2.0 <= frequency_penalty <= 2.0:
            raise ValueError(
                f"frequency_penalty must be in [-2, 2], got {frequency_penalty}.")
        if temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {temperature}.")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
        if top_k < -1 or top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, "
                             f"got {top_k}.")
        if max_tokens < 1:
            raise ValueError(
                f"max_tokens must be at least 1, got {max_tokens}.")
        if logprobs < 0:
            raise ValueError(
                f"logprobs must be non-negative, got {logprobs}.")

        if use_beam_search:
            if n == 1:
                raise ValueError(
                    "n must be greater than 1 when using beam search.")
            if temperature > 0.0:
                raise ValueError(
                    "temperature must be 0 when using beam search.")
            if top_p < 1.0:
                raise ValueError(
                    "top_p must be 1 when using beam search.")
            if top_k != -1:
                raise ValueError(
                    "top_k must be -1 when using beam search.")
        elif temperature == 0.0:
            # Zero temperature means greedy sampling.
            if n > 1:
                raise ValueError(
                    "n must be 1 when using greedy sampling.")
            if top_p < 1.0:
                raise ValueError(
                    "top_p must be 1 when using greedy sampling.")
            if top_k != -1:
                raise ValueError(
                    "top_k must be -1 when using greedy sampling.")

        self.n = n
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = use_beam_search
        self.stop_token_ids = stop_token_ids
        self.max_tokens = max_tokens
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"SamplingParams(n={self.n}, "
                f"presence_penalty={self.presence_penalty}, "
                f"frequency_penalty={self.frequency_penalty}, "
                f"temperature={self.temperature}, "
                f"top_p={self.top_p}, "
                f"top_k={self.top_k},"
                f"use_beam_search={self.use_beam_search}, "
                f"stop_token_ids={self.stop_token_ids}, "
                f"max_tokens={self.max_tokens}, "
                f"logprobs={self.logprobs}")
