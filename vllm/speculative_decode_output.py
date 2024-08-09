class SpeculativeDecodeOutput:
    def __init__(
        self,
        draft_token_indices: list[int],
        target_token_indices: list[int],
        accepted_tokens_count: int,
    ) -> None:
        self._draft_token_indices = draft_token_indices
        self._target_token_indices = target_token_indices
        self._accepted_tokens_count = accepted_tokens_count

        assert len(draft_token_indices) == len(target_token_indices)
        assert accepted_tokens_count >= 0
    
    @property
    def draft_token_indices(self) -> list[int]:
        return self._draft_token_indices
    
    @property
    def target_token_indices(self) -> list[int]:
        return self._target_token_indices
    
    @property
    def accepted_tokens_count(self) -> list[int]:
        return self._accepted_tokens_count
