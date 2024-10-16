"""Utils for model executor."""
from typing import Any, Dict, Optional

import torch

from vllm.utils import seed_everything
from vllm.platforms import current_platform


def set_random_seed(seed: int) -> None:
    seed_everything(seed)


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

        # NOTE(woosuk): For TPU, param.data.copy_(weight) happens lazily,
        # which means that the param and weight tensors co-exist until the param
        # tensor is used by other operations. This causes excessive memory usage
        # during model loading. To avoid this, we sync the param tensor after
        # its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        if current_platform.is_tpu() and key == "weight_loader":
            original_weight_loader = value

            def _synced_weight_loader(param, *args, **kwargs):
                original_weight_loader(param, *args, **kwargs)
                torch._sync(param)

            value = _synced_weight_loader
        setattr(weight, key, value)
