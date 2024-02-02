from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.sparsity.base_config import SparsityConfig
from vllm.model_executor.layers.parameters import SparseParameter


class SemiStructuredSparseW16A16Config(SparsityConfig):
    """Config class for SemiStructuredSparseW16A16.
    """

    def __init__(self) -> None:
        # TODO: Add new configs here
        pass

    def __repr__(self) -> str:
        return "SemiStructuredSparseW16A16Config()"

    @classmethod
    def get_name(cls) -> str:
        return "semi_structured_sparse_w16a16"

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
    def from_config(cls, config: Dict[str, Any]) -> "SemiStructuredSparseW16A16Config":
        return cls()

    def get_linear_method(self) -> "SemiStructuredSparseW16A16LinearMethod":
        return SemiStructuredSparseW16A16LinearMethod(self)


class SemiStructuredSparseW16A16LinearMethod(LinearMethodBase):
    """Linear method for Semi Structured Sparse W16A16.

    Args:
        sparsity_config: The sparse config.
    """

    def __init__(self, sparsity_config: SemiStructuredSparseW16A16Config):
        self.sparsity_config = sparsity_config

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        weight = SparseParameter(
            shape=torch.Size(
                (output_size_per_partition, input_size_per_partition)),
            dtype=params_dtype,
        )

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        return {"weight": weight}

    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sparse_weight = weights["weight"]

        # Uncompress to dense
        dense_weight = sparse_weight.to_dense()

        # # Uncomment to verify sparsity
        # density = torch.count_nonzero(
        #     dense_weight).item() / dense_weight.numel()
        # print(f"sparsity = {1.0 - density}")

        # Standard matrix multiply
        if bias is not None:
            output = F.linear(x, dense_weight, bias)
        else:
            output = F.linear(x, dense_weight)

        return output
