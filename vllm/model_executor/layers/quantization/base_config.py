from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch

from vllm.model_executor.layers.linear import LinearMethodBase


class QuantizationConfig(ABC):
    """Base class for quantization configs."""

    @abstractmethod
    def get_name(self) -> str:
        """Name of the quantization method."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """List of supported activation dtypes."""
        raise NotImplementedError

    @abstractmethod
    def get_min_capability(self) -> int:
        """Minimum GPU capability to support the quantization method.

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
        This requirement is due to the custom CUDA kernels used by the
        quantization method.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> List[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config."""
        raise NotImplementedError

    @staticmethod
    def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Cannot find any of {keys} in the model's "
                         "quantization config.")

    @abstractmethod
    def get_linear_method(self) -> LinearMethodBase:
        """Get the linear method to use for the quantized linear layer."""
        raise NotImplementedError

    @abstractmethod
    def get_scaled_act_names(self) -> List[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError
