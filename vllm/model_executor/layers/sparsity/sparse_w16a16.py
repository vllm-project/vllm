from typing import Any, Dict, List, Type

import torch

from vllm.model_executor.layers.sparsity.base_config import SparsityConfig

from .sparse_w16a16_linear_method import SparseW16A16LinearMethod
from magic_wand import (CompressedStorageFormat, SparseBEGemmStorageFormat)


class SparseW16A16Config(SparsityConfig):
    """Config class for SparseW16A16.

    TODO: Add based on need
    """

    def __init__(self) -> None:
        # TODO: Add new configs here
        pass

    def __repr__(self) -> str:
        return "SparseW16A16Config()"

    @classmethod
    def get_storage_format_cls(cls) -> Type[CompressedStorageFormat]:
        return SparseBEGemmStorageFormat

    @classmethod
    def get_name(cls) -> str:
        return "sparse_w16a16"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # TODO: Update after checks on more GPUs
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["sparsity_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SparseW16A16Config":
        return cls()

    def get_linear_method(self) -> "SparseW16A16LinearMethod":
        return SparseW16A16LinearMethod(self, self.get_storage_format_cls())
