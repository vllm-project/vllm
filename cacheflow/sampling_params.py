from typing import Dict, Set


class SamplingParams:

    def __init__(
        self,
        n: int,
        presence_penalty: float,
        frequency_penalty: float,
        temperature: float,
        top_p: float,
        top_k: int,
        use_beam_search: bool,
        stop_token_ids: Set[int],
        max_num_steps: int,
        num_logprobs: int,
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
        if max_num_steps < 1:
            raise ValueError(
                f"max_num_steps must be at least 1, got {max_num_steps}.")
        if num_logprobs < 0:
            raise ValueError(
                f"num_logprobs must be non-negative, got {num_logprobs}.")

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
        self.max_num_steps = max_num_steps
        self.num_logprobs = num_logprobs

    def __repr__(self) -> str:
        return (f"SamplingParams(n={self.n}, "
                f"presence_penalty={self.presence_penalty}, "
                f"frequency_penalty={self.frequency_penalty}, "
                f"temperature={self.temperature}, "
                f"top_p={self.top_p}, "
                f"top_k={self.top_k},"
                f"use_beam_search={self.use_beam_search}, "
                f"stop_token_ids={self.stop_token_ids}, "
                f"max_num_steps={self.max_num_steps}, "
                f"num_logprobs={self.num_logprobs}")

    @classmethod
    def from_dict(cls, d: Dict) -> "SamplingParams":
        sampling_params = cls(
            n=d.pop("n", 1),
            presence_penalty=d.pop("presence_penalty", 0.0),
            frequency_penalty=d.pop("frequency_penalty", 0.0),
            temperature=d.pop("temperature", 1.0),
            top_p=d.pop("top_p", 1.0),
            top_k=d.pop("top_k", -1),
            use_beam_search=d.pop("use_beam_search", False),
            stop_token_ids=set(d.pop("stop_token_ids", set())),
            max_num_steps=d.pop("max_num_steps", 16),
            num_logprobs=d.pop("num_logprobs", 0),
        )
        if d:
            raise ValueError(f"Unrecognized keys in dict: {d.keys()}")
        return sampling_params
