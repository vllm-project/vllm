from typing import Any, Dict, List, Type

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.sparsity.base_config import SparsityConfig

from .sparse_w16a16_linear_method import SparseW16A16LinearMethod
from magic_wand import (CompressedStorageFormat, SparseBitmaskStorageFormat,
                        SparseBEGemmStorageFormat)

logger = init_logger(__name__)


class SparseW16A16Config(SparsityConfig):
    """Config class for SparseW16A16."""

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "SparseW16A16Config()"

    @classmethod
    def get_storage_format_cls(cls) -> Type[CompressedStorageFormat]:
        cuda_compute_capability = torch.cuda.get_device_capability()
        if cuda_compute_capability >= (8, 0):
            return SparseBEGemmStorageFormat
        else:
            # For NVIDIA SM < 8.0
            logger.warning("Unstructured sparse kernels are not optimized for "
                           "NVIDIA SM < 8.0. Naive decompress kernels will be "
                           "used and can be slower than dense models")
            return SparseBitmaskStorageFormat

    @classmethod
    def get_name(cls) -> str:
        return "sparse_w16a16"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["sparsity_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SparseW16A16Config":
        return cls()

    def get_linear_method(self) -> "SparseW16A16LinearMethod":
        return SparseW16A16LinearMethod(self, self.get_storage_format_cls())
