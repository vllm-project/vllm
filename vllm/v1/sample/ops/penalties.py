# SPDX-License-Identifier: Apache-2.0

import torch
from typing import List, Optional, Dict, Any
from vllm.v1.sample.metadata import SamplingMetadata 

from vllm.model_executor.layers.utils import apply_penalties
from vllm.utils import is_pin_memory_available, make_tensor_with_pad


def apply_min_token_penalties(
        logits: torch.Tensor, output_token_ids: list[list[int]],
        min_tokens: dict[int, tuple[int, set[int]]]) -> None:
    """
    Applies minimum token penalty by setting the logits of the stop tokens
    to -inf.
    """
    min_tokens_logits_to_penalize: list[tuple[int, int]] = []
    for index, (min_token, stop_token_ids) in min_tokens.items():
        if len(output_token_ids[index]) < min_token:
            for stop_token_id in stop_token_ids:
                min_tokens_logits_to_penalize.append((index, stop_token_id))
    if min_tokens_logits_to_penalize:
        logits[tuple(zip(*min_tokens_logits_to_penalize))] = -float("inf")


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
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


def _convert_to_tensors(output_token_ids: list[list[int]], vocab_size: int,
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


# Constants for DRY
_DRY_MAX_NGRAM = 12
_DRY_MAX_OCCURRENCES = 8
_DRY_EARLY_EXIT_MATCH_LEN = 8

# Default DRY parameter values
_DRY_DEFAULT_MULTIPLIER = 0.0
_DRY_DEFAULT_BASE = 1.0
_DRY_DEFAULT_ALLOWED_LEN = 3
_DRY_DEFAULT_RANGE = 1500 
_DRY_DEFAULT_BREAKERS: set[int] = set()


def apply_dry(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """
    Applies DRY (Don't Repeat Yourself) penalty to logits based on parameters
    found in sampling_metadata.extra_data for each request.

    Modifies logits in-place or returns a modified tensor.

    Expected keys in extra_data[irow] (if DRY is active):
        - 'dry_multiplier' (float)
        - 'dry_base' (float)
        - 'dry_allowed_len' (int)
        - 'dry_range' (int)
        - 'dry_breakers' (List[int] or Set[int])
    """
    batch_size, vocab_size = logits.shape
    device = logits.device

    # Assume extra_data is a list of dicts, one per request
    # Or potentially accessed via methods on sampling_metadata
    # Adjust access pattern if sampling_metadata structure differs
    if not hasattr(sampling_metadata, 'extra_data') or sampling_metadata.extra_data is None:
         # If no extra_data field exists or is None, cannot apply DRY
         return logits

    # Check if any request might have DRY enabled (basic check)
    # More robust check would involve iterating through extra_data first
    has_potential_dry = any(
        data and data.get('dry_multiplier', _DRY_DEFAULT_MULTIPLIER) > 0
        for data in sampling_metadata.extra_data
    )
    if not has_potential_dry:
         return logits

    # --- Iterate through each request in the batch ---
    for irow in range(batch_size):
        # Ensure sampling_metadata has data for this row index
        if irow >= len(sampling_metadata.extra_data):
             # If metadata doesn't cover this row (shouldn't happen in normal flow), skip
             continue
        extra_data = sampling_metadata.extra_data[irow]
        if not extra_data:
            continue

        # Get DRY parameters for this row, using defaults if missing
        multiplier = float(extra_data.get('dry_multiplier', _DRY_DEFAULT_MULTIPLIER))
        if multiplier <= 0.0:
            continue # DRY not active for this row

        base = float(extra_data.get('dry_base', _DRY_DEFAULT_BASE))
        allowed_length = int(extra_data.get('dry_allowed_len', _DRY_DEFAULT_ALLOWED_LEN))
        dry_range = int(extra_data.get('dry_range', _DRY_DEFAULT_RANGE))
        breakers_input = extra_data.get('dry_breakers', _DRY_DEFAULT_BREAKERS)
        breakers = set(breakers_input) if breakers_input else _DRY_DEFAULT_BREAKERS

        # 1. Construct the token sequence for this row
        # Get prompt tokens (handle potential padding if needed)
        # Assuming prompt_token_ids is available and correctly indexed
        # Need prompt_lens if prompt_token_ids is padded
        prompt_len_attr = getattr(sampling_metadata, 'prompt_lens', None)
        current_prompt_len = (prompt_len_attr[irow]
                              if prompt_len_attr and irow < len(prompt_len_attr)
                              else None)

        # Ensure prompt_token_ids covers this row
        if irow >= sampling_metadata.prompt_token_ids.shape[0]:
             continue # Skip if prompt data isn't available for this row
        prompt_tensor_row = sampling_metadata.prompt_token_ids[irow]

        if current_prompt_len is not None:
             current_prompt_tokens = prompt_tensor_row[:current_prompt_len].tolist()
        else:
             # If prompt_lens is not available, we cannot reliably determine
             # the prompt sequence length. Log a warning or raise an error,
             # or potentially skip DRY for this row if appropriate.
             # For now, let's skip if length is unknown.
             # Consider adding logging: logger.warning("prompt_lens not available...")
             continue # Skip DRY for this row if prompt length is unknown

        # Get output tokens for this row
        # Ensure output_token_ids covers this row
        if irow >= len(sampling_metadata.output_token_ids):
             continue # Skip if output data isn't available
        current_output_tokens = sampling_metadata.output_token_ids[irow]
        token_seq_list = current_prompt_tokens + current_output_tokens

        # 2. Apply range limit
        if dry_range > 0 and len(token_seq_list) > dry_range:
            token_seq_list = token_seq_list[-dry_range:]

        seq_len = len(token_seq_list)
        if seq_len < 2:
            continue # Need at least 2 tokens

        last_token = token_seq_list[-1]
        if last_token in breakers:
            continue

        # Convert to tensor for efficient processing
        token_seq_tensor = torch.tensor(
            token_seq_list,
            dtype=torch.long,
            device=device
        )

        # 3. Build break mask on device
        break_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if breakers:
            breaker_tensor = torch.tensor(list(breakers), dtype=torch.long, device=device)
            # Use broadcasting for efficiency
            break_mask = torch.any(token_seq_tensor.unsqueeze(1) == breaker_tensor, dim=1)


        # 4. Determine max n-gram length possible from the end
        max_ngram = 0
        for offset in range(1, min(seq_len, _DRY_MAX_NGRAM + 1)):
            check_idx = -offset - 1
            if check_idx < -seq_len:
                break
            if break_mask[check_idx]:
                break
            max_ngram = offset

        if max_ngram < allowed_length:
            continue

        # 5. Find previous occurrences of last_token
        endpoint_indices = (token_seq_tensor == last_token).nonzero(as_tuple=True)[0]
        if len(endpoint_indices) < 2:
            continue

        endpoint_indices = endpoint_indices[endpoint_indices != seq_len - 1]
        if len(endpoint_indices) == 0:
            continue

        if len(endpoint_indices) > _DRY_MAX_OCCURRENCES:
            endpoint_indices = endpoint_indices[-_DRY_MAX_OCCURRENCES:]

        # 6. Calculate match lengths for potential next tokens
        ngram_lens = torch.zeros(vocab_size, dtype=torch.int32, device=device)
        found_early_exit_match = False

        for idx_tensor in reversed(endpoint_indices):
            idx = idx_tensor.item()
            match_len = 0
            for unwind in range(1, max_ngram + 1):
                current_idx = idx - unwind
                history_idx = seq_len - 1 - unwind
                if current_idx < 0:
                    break
                # Check breaks using the precomputed mask
                if break_mask[current_idx] or break_mask[history_idx]:
                    break
                if token_seq_tensor[current_idx] != token_seq_tensor[history_idx]:
                    break
                match_len = unwind

            if match_len >= allowed_length: # Match length must meet minimum
                next_tok_idx = idx + 1
                if next_tok_idx < seq_len:
                    next_tok = token_seq_tensor[next_tok_idx].item()
                    # Use match_len as the length of the *matched* sequence
                    new_len = match_len
                    current_max = ngram_lens[next_tok].item()
                    ngram_lens[next_tok] = max(current_max, new_len)
                    if new_len >= _DRY_EARLY_EXIT_MATCH_LEN:
                        found_early_exit_match = True

            if found_early_exit_match:
                 break

        # 7. Apply penalty to logits for this row
        penalty_mask = ngram_lens > 0
        if penalty_mask.any():
            match_lengths_for_penalty = ngram_lens[penalty_mask]
            # Clamp exponent >= 0
            exponents = (match_lengths_for_penalty.float() - allowed_length).clamp_(min=0.0)
            scales = base ** exponents
            logits[irow, penalty_mask] -= multiplier * scales
        # --- End of DRY logic for row ---

    return logits
