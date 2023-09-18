from typing import Any, Dict, List

import torch


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

    @classmethod
    def get_packed_tensor_names(cls) -> List[str]:
        raise NotImplementedError

    @classmethod
    def is_packed(cls, tensor_name: str) -> bool:
        """Returns True if a tensor is packed.

        A tensor is considered packed if each element in the tensor is a
        packed representation of multiple elements in the original tensor.
        For example, an INT32 element in the tensor may represent 8 INT4
        elements in the original tensor.
        """
        return any(tag in tensor_name for tag in cls.get_packed_tensor_names())

    @classmethod
    def get_transposed_tensor_names(cls) -> List[str]:
        raise NotImplementedError

    @classmethod
    def is_transposed(cls, tensor_name: str) -> bool:
        """Returns True if a tensor is transposed relative to nn.Linear.weight.
        """
        return any(tag in tensor_name
                   for tag in cls.get_transposed_tensor_names())

    @classmethod
    def get_tp_tensor_names(cls) -> List[str]:
        raise NotImplementedError
