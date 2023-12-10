"""Utils for model executor."""
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")
        setattr(weight, key, value)


def replace_prompt_embeds(
    inputs_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_indices: List[int],
):
    inputs_embeds[torch.tensor(prompt_embeds_indices)] = torch.index_select(
        prompt_embeds, 0, torch.tensor(prompt_embeds_indices))
    return inputs_embeds
