"""Sampling parameters for text generation."""
from typing import Set


class SamplingParams:
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.

    TODO(woosuk): Support best_of.

    Args:
        n: Number of output sequences to generate from the given prompt. This is
            regarded as the beam width when using beam search.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        use_beam_search: Whether to use beam search instead of sampling.
        stop_token_ids: Set of token IDs that indicate the end of a sequence.
        max_tokens: Maximum number of tokens to generate per output sequence.
        logprobs: Number of log probabilities to return per output token.
    """

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
