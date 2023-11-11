from typing import Any, Dict, List

import torch

from vllm.model_executor.layers.linear import LinearMethodBase


class QuantizationConfig:

    @classmethod
    def get_name(cls) -> str:
        """Name of the quantization method."""
        raise NotImplementedError

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        """List of supported activation dtypes."""
        raise NotImplementedError

    @classmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU capability to support the quantization method.

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
        This requirement is due to the custom CUDA kernels used by the
        quantization method.
        """
        raise NotImplementedError

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError

    @classmethod
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

    def get_linear_method(self) -> LinearMethodBase:
        raise NotImplementedError
