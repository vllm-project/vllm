from typing import Type
import importlib.util

is_magic_wand_available = importlib.util.find_spec("magic_wand") is not None
if not is_magic_wand_available:
    raise ValueError(
        "magic_wand is not available and required for sparsity "
        "support. Please install it with `pip install nm-magic-wand`")

from vllm.model_executor.layers.sparsity.base_config import SparsityConfig  # noqa: E402
from vllm.model_executor.layers.sparsity.sparse_w16a16 import SparseW16A16Config  # noqa: E402
from vllm.model_executor.layers.sparsity.semi_structured_sparse_w16a16 import SemiStructuredSparseW16A16Config  # noqa: E402

_SPARSITY_CONFIG_REGISTRY = {
    "sparse_w16a16": SparseW16A16Config,
    "semi_structured_sparse_w16a16": SemiStructuredSparseW16A16Config,
}


def get_sparsity_config(sparsity: str) -> Type[SparsityConfig]:
    if sparsity not in _SPARSITY_CONFIG_REGISTRY:
        raise ValueError(f"Invalid sparsity method: {sparsity}")
    return _SPARSITY_CONFIG_REGISTRY[sparsity]


__all__ = [
    "SparsityConfig",
    "get_sparsity_config",
]
