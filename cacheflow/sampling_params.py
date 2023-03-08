from typing import Optional, Set


class SamplingParams:

    def __init__(
        self,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        use_beam_search: bool = False,
        stop_token_ids: Set[int] = [],
        max_num_steps: int = 16,  # From OpenAI API.
        context_window_size: Optional[int] = None,
    ) -> None:
        assert n >= 1
        assert temperature >= 0.0
        assert 0.0 < top_p <= 1.0
        if use_beam_search:
            assert n > 1
            assert temperature > 0.0
            assert top_p == 1.0
        elif temperature == 0.0:
            # Zero temperature means greedy sampling.
            assert n == 1
            assert top_p == 1.0
        assert max_num_steps >= 1
        assert context_window_size is None or context_window_size >= 0

        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.use_beam_search = use_beam_search
        self.stop_token_ids = stop_token_ids
        self.max_num_steps = max_num_steps
        self.context_window_size = context_window_size

    def __repr__(self) -> str:
        return (f'SamplingParams(n={self.n}, '
                f'temperature={self.temperature}, '
                f'top_p={self.top_p}, '
                f'use_beam_search={self.use_beam_search}, '
                f'stop_token_ids={self.stop_token_ids}, '
                f'max_num_steps={self.max_num_steps}, '
                f'context_window_size={self.context_window_size})')
