from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class PromptAdapterMapping:
    # Per every token in input_ids:
    index_mapping: Tuple[int, ...]
    # Per sampled token:
    prompt_mapping: Tuple[int, ...]

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)


def apply_prompt_adapter(instance, hidden_states: torch.Tensor,
                         positions: torch.Tensor) -> torch.Tensor:
    if instance.prefix_encoder is not None:
        soft_prompt = instance.prefix_encoder.prompt_embedding
        indices = (positions < soft_prompt.shape[0])
        hidden_states[indices] = soft_prompt[positions[indices]]
    return hidden_states
