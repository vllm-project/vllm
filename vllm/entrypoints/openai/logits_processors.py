# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache, partial
from typing import Dict, FrozenSet, Iterable, List, Optional, Union

import torch

from vllm.sampling_params import LogitsProcessor
from vllm.transformers_utils.tokenizer import AnyTokenizer


class AllowedTokenIdsLogitsProcessor:
    """Logits processor for constraining generated tokens to a
    specific set of token ids."""

    def __init__(self, allowed_ids: Iterable[int]):
        self.allowed_ids: Optional[List[int]] = list(allowed_ids)
        self.mask: Optional[torch.Tensor] = None

    def __call__(self, token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            self.mask = torch.ones((logits.shape[-1], ),
                                   dtype=torch.bool,
                                   device=logits.device)
            self.mask[self.allowed_ids] = False
            self.allowed_ids = None
        logits.masked_fill_(self.mask, float("-inf"))
        return logits


@lru_cache(maxsize=32)
def _get_allowed_token_ids_logits_processor(
    allowed_token_ids: FrozenSet[int],
    vocab_size: int,
) -> LogitsProcessor:
    if not allowed_token_ids:
        raise ValueError("Empty allowed_token_ids provided")
    if not all(0 <= tid < vocab_size for tid in allowed_token_ids):
        raise ValueError("allowed_token_ids contains "
                         "out-of-vocab token id")
    return AllowedTokenIdsLogitsProcessor(allowed_token_ids)


def logit_bias_logits_processor(
    logit_bias: Dict[str, torch.Tensor],
    token_ids: List[int],
    logits: torch.Tensor,
) -> torch.Tensor:
    logits.index_add_(0, logit_bias["index"].to(logits.device),
                      logit_bias["value"].to(logits.device))
    return logits


def get_logits_processors(
    logit_bias: Optional[Union[Dict[int, float], Dict[str, float]]],
    allowed_token_ids: Optional[List[int]],
    tokenizer: AnyTokenizer,
    dtype: Union[str, torch.dtype],
) -> List[LogitsProcessor]:
    logits_processors: List[LogitsProcessor] = []
    if logit_bias:
        try:
            # Convert token_id to integer
            # Clamp the bias between -100 and 100 per OpenAI API spec
            logit_bias_index = [int(token_id) for token_id in logit_bias]
            logit_bias_value = [
                min(100.0, max(-100.0, bias)) for bias in logit_bias.values()
            ]
        except ValueError as exc:
            raise ValueError(
                "Found token_id in logit_bias that is not "
                "an integer or string representing an integer") from exc

        # Check if token_id is within the vocab size
        for token_id in logit_bias_index:
            if token_id < 0 or token_id >= len(tokenizer):
                raise ValueError(f"token_id {token_id} in logit_bias contains "
                                 "out-of-vocab token id")

        clamped_logit_bias: Dict[str, torch.Tensor] = {
            "index": torch.tensor(logit_bias_index),
            "value": torch.tensor(logit_bias_value, dtype=dtype)
        }
        logits_processors.append(
            partial(logit_bias_logits_processor, clamped_logit_bias))

    if allowed_token_ids is not None:
        logits_processors.append(
            _get_allowed_token_ids_logits_processor(
                frozenset(allowed_token_ids), len(tokenizer)))

    return logits_processors
