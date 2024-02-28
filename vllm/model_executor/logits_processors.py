"""Logits Processors to use with SamplingParams"""
from typing import List, Union

import torch


class MinNewTokensLogitsProcessor:

    def __init__(self, min_new_tokens: int, stop_token_ids: Union[int,
                                                                  List[int]]):
        self.min_tokens = min_new_tokens
        self.stop_token_ids = torch.tensor(stop_token_ids)

    def __call__(self, token_ids: List[int],
                 logits: torch.tensor) -> torch.tensor:
        # token_ids is only output tokens
        if len(token_ids) < self.min_tokens:
            logits[self.stop_token_ids] = -float("inf")
        return logits
