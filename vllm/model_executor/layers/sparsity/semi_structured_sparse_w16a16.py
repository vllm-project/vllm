import torch

from typing import Any, Dict, List, Type
from vllm.model_executor.layers.sparsity.base_config import SparsityConfig
from .sparse_w16a16_linear_method import SparseW16A16LinearMethod
from magic_wand import (CompressedStorageFormat,
                        SparseSemiStructuredStorageFormat)


class SemiStructuredSparseW16A16Config(SparsityConfig):
    """Config class for SemiStructuredSparseW16A16."""

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "SemiStructuredSparseW16A16Config()"

    @classmethod
    def get_storage_format_cls(cls) -> Type[CompressedStorageFormat]:
        return SparseSemiStructuredStorageFormat

    @classmethod
    def get_name(cls) -> str:
        return "semi_structured_sparse_w16a16"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # TODO: Update after checks on more GPUs
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["sparsity_config.json"]

    @classmethod
    def from_config(
            cls, config: Dict[str, Any]) -> "SemiStructuredSparseW16A16Config":
        return cls()

    def get_linear_method(self) -> "SparseW16A16LinearMethod":
        return SparseW16A16LinearMethod(self, self.get_storage_format_cls())
