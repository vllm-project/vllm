"""Utils for model executor."""
import random
import importlib
from typing import Any, Dict, Optional

import numpy as np
import torch

from vllm.config import DeviceConfig, ModelConfig

DEVICE_TO_MODEL_LOADER_MAP = {
    "cuda": "model_loader",
    "neuron": "neuron_model_loader",
}


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


def get_model(model_config: ModelConfig, device_config: DeviceConfig,
              **kwargs) -> torch.nn.Module:
    model_loader_module = DEVICE_TO_MODEL_LOADER_MAP[device_config.device_type]
    imported_model_loader = importlib.import_module(
        f"vllm.model_executor.{model_loader_module}")
    get_model_fn = imported_model_loader.get_model
    return get_model_fn(model_config, device_config, **kwargs)
