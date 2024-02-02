from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from typing import Type

from vllm.model_executor.layers.linear import LinearMethodBase
from magic_wand import CompressedStorageFormat


class SparsityConfig(ABC):
    """Base class for sparsity configs."""

    @abstractmethod
    def get_storage_format_cls(self) -> Type[CompressedStorageFormat]:
        """Sparse representation format"""
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Name of the sparse method."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """List of supported act_dtypes."""
        raise NotImplementedError

    @abstractmethod
    def get_min_capability(self) -> int:
        """Minimum GPU capability to support the sparsity method."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> List[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "SparsityConfig":
        """Create a config class from the model's sparse config."""
        raise NotImplementedError

    @staticmethod
    def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:
        """Get a value from the model's sparsity config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Cannot find any of {keys} in the model's "
                         "sparsity config.")

    @abstractmethod
    def get_linear_method(self) -> LinearMethodBase:
        """Get the linear method to use for the sparse linear layer."""
        raise NotImplementedError
