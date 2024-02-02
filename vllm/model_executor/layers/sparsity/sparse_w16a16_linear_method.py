from typing import Any, Dict, Optional, Type

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.sparsity.base_config import SparsityConfig
from vllm.model_executor.layers.parameters import SparseParameter
from magic_wand import (CompressedStorageFormat,
                        SparseSemiStructuredStorageFormat)


class SparseW16A16LinearMethod(LinearMethodBase):
    """Linear method for Sparse W16A16.

    Args:
        sparsity_config: The sparse config.
    """
    storage_format_cls: Type[CompressedStorageFormat] = None

    def __init__(self, sparsity_config: SparsityConfig,
                 storage_format_cls: Type[CompressedStorageFormat]):
        self.sparsity_config = sparsity_config
        self.storage_format_cls = storage_format_cls

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        weight = SparseParameter(shape=torch.Size(
            (output_size_per_partition, input_size_per_partition)),
                                 dtype=params_dtype,
                                 storage_format_cls=self.storage_format_cls)

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        return {"weight": weight}

    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sparse_weight = weights["weight"]

        if self.storage_format_cls == SparseSemiStructuredStorageFormat:
            output = F.linear(x, sparse_weight, bias)
            return output
        else:

            # Standard matrix multiply
            # Uncompress to dense
            output = F.linear(x, sparse_weight.to_dense(), bias)
            return output
