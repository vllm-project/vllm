from abc import abstractmethod
from typing import Type

from magic_wand import CompressedStorageFormat

from vllm.model_executor.layers.quantization import QuantizationConfig


class SparsityConfig(QuantizationConfig):
    """Base class for sparsity configs."""

    @abstractmethod
    def get_storage_format_cls(self) -> Type[CompressedStorageFormat]:
        """Sparse representation format"""
        raise NotImplementedError
