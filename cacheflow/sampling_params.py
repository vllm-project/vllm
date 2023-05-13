from typing import Set


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

        self._verify_args()
        if self.use_beam_search:
            self._verity_beam_search()
        elif self.temperature == 0.0:
            # Zero temperature means greedy sampling.
            self._verify_greedy_sampling()

    def _verify_args(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2], got "
                             f"{self.presence_penalty}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2], got "
                             f"{self.frequency_penalty}.")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, "
                             f"got {self.top_k}.")
        if self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.logprobs < 0:
            raise ValueError(
                f"logprobs must be non-negative, got {self.logprobs}.")

    def _verity_beam_search(self) -> None:
        if self.n == 1:
            raise ValueError("n must be greater than 1 when using beam search.")
        if self.temperature > 0.0:
            raise ValueError("temperature must be 0 when using beam search.")
        if self.top_p < 1.0:
            raise ValueError("top_p must be 1 when using beam search.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using beam search.")

    def _verify_greedy_sampling(self) -> None:
        if self.n > 1:
            raise ValueError("n must be 1 when using greedy sampling.")
        if self.top_p < 1.0:
            raise ValueError("top_p must be 1 when using greedy sampling.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using greedy sampling.")

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
