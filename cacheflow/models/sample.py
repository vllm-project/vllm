from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from cacheflow.models import InputMetadata


class Sampler(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> Dict[int, Tuple[int, int]]:
        # Get the hidden states of the last tokens.
        start_idx = 0
        last_token_indicies: List[int] = []
        for prompt_len in input_metadata.prompt_lens:
            last_token_indicies.append(start_idx + prompt_len - 1)
            start_idx += prompt_len
        last_token_indicies.extend(
            range(start_idx, start_idx + input_metadata.num_generation_tokens))
        hidden_states = hidden_states[last_token_indicies]

        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())

        # Sample the next tokens.
        # TODO(woosuk): Implement other sampling methods.
        next_token_ids = torch.argmax(logits, dim=-1)
        next_token_ids = next_token_ids.tolist()

        # Return the next tokens.
        next_tokens: Dict[int, Tuple[int, int]] = {}
        for seq_id, token_id in zip(input_metadata.seq_ids, next_token_ids):
            next_tokens[seq_id] = (seq_id, token_id)
        return next_tokens
