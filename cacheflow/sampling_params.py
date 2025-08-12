from typing import Optional, Set, Dict


class SamplingParams:

    def __init__(
        self,
        n: int,
        temperature: float,
        top_p: float,
        use_beam_search: bool,
        stop_token_ids: Set[int],
        max_num_steps: int,
        num_logprobs: int,
        context_window_size: Optional[int],
    ) -> None:
        if n < 1:
            raise ValueError(f'n must be at least 1, got {n}.')
        if temperature < 0.0:
            raise ValueError(
                f'temperature must be non-negative, got {temperature}.')
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f'top_p must be in (0, 1], got {top_p}.')
        if max_num_steps < 1:
            raise ValueError(
                f'max_num_steps must be at least 1, got {max_num_steps}.')
        if num_logprobs < 0:
            raise ValueError(
                f'num_logprobs must be non-negative, got {num_logprobs}.')
        if context_window_size is not None and context_window_size < 0:
            raise ValueError(
                'context_window_size must be non-negative, '
                f'got {context_window_size}.')

        if use_beam_search:
            if n == 1:
                raise ValueError(
                    'n must be greater than 1 when using beam search.')
            if temperature > 0.0:
                raise ValueError(
                    'temperature must be 0 when using beam search.')
            if top_p < 1.0:
                raise ValueError(
                    'top_p must be 1 when using beam search.')
        elif temperature == 0.0:
            # Zero temperature means greedy sampling.
            if n > 1:
                raise ValueError(
                    'n must be 1 when using greedy sampling.')
            if top_p < 1.0:
                raise ValueError(
                    'top_p must be 1 when using greedy sampling.')

        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.use_beam_search = use_beam_search
        self.stop_token_ids = stop_token_ids
        self.max_num_steps = max_num_steps
        self.num_logprobs = num_logprobs
        self.context_window_size = context_window_size

    def __repr__(self) -> str:
        return (f'SamplingParams(n={self.n}, '
                f'temperature={self.temperature}, '
                f'top_p={self.top_p}, '
                f'use_beam_search={self.use_beam_search}, '
                f'stop_token_ids={self.stop_token_ids}, '
                f'max_num_steps={self.max_num_steps}, '
                f'num_logprobs={self.num_logprobs}, '
                f'context_window_size={self.context_window_size})')

    @classmethod
    def from_dict(cls, d: Dict) -> 'SamplingParams':
        return cls(
            n=d.get('n', 1),
            temperature=d.get('temperature', 1.0),
            top_p=d.get('top_p', 1.0),
            use_beam_search=d.get('use_beam_search', False),
            stop_token_ids=set(d.get('stop_token_ids', set())),
            max_num_steps=d.get('max_num_steps', 16),
            num_logprobs=d.get('num_logprobs', 0),
            context_window_size=d.get('context_window_size', None),
        )
