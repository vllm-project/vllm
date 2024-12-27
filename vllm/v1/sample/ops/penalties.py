from typing import List, Set, Tuple

import torch

from vllm.model_executor.layers.utils import apply_penalties
from vllm.utils import is_pin_memory_available, make_tensor_with_pad


def apply_min_token_penalties(logits: torch.Tensor,
                              output_token_ids: List[List[int]],
                              stop_token_ids: List[Set[int]],
                              min_tokens: List[int]) -> None:
    """
    Applies minimum token penalty by setting the logits of the stop tokens
    to -inf.
    """
    min_tokens_logits_to_penalize: List[Tuple[int, int]] = []
    for index, min_token in enumerate(min_tokens):
        if len(output_token_ids[index]) < min_token:
            for stop_token_id in stop_token_ids[index]:
                min_tokens_logits_to_penalize.append((index, stop_token_id))
    if min_tokens_logits_to_penalize:
        logits[tuple(zip(*min_tokens_logits_to_penalize))] = -float("inf")


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: List[List[int]],
) -> torch.Tensor:
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size,
                                          logits.device)
    return apply_penalties(logits, prompt_token_ids, output_tokens_t,
                           presence_penalties, frequency_penalties,
                           repetition_penalties)


def _convert_to_tensors(output_token_ids: List[List[int]], vocab_size: int,
                        device: torch.device) -> torch.Tensor:
    """
    Convert the different list data structures to tensors.
    """
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    )
    return output_tokens_tensor.to(device, non_blocking=True)
