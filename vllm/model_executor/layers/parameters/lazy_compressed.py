import importlib.util
from typing import Optional, Type

import numpy
import torch
from torch.utils._pytree import tree_map

from vllm.logger import init_logger

logger = init_logger(__name__)

is_magic_wand_available = importlib.util.find_spec("magic_wand") is not None

# These are types from magic_wand, but we only want to import if required
CompressedStorageFormat = "CompressedStorageFormat"
SparseBitmaskStorageFormat = "SparseBitmaskStorageFormat"


class LazyCompressedParameter(torch.Tensor):
    uncompressed_data: Optional[torch.Tensor]
    compressed_data: Optional[torch.Tensor]

    @staticmethod
    def __new__(
            cls,
            uncompressed_data: torch.Tensor,
            is_empty: bool = False,
            # Lazy import causes typing issues.
            storage_format_cls:  # type: ignore
        Type[  # type: ignore
            CompressedStorageFormat] = SparseBitmaskStorageFormat,  # type: ignore
            compress_transposed: bool = False):

        if not is_magic_wand_available:
            raise ValueError(
                "magic_wand is not available and required for sparsity "
                "support. Please install it with `pip install nm-magic-wand`")

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            size=uncompressed_data.shape,
            dtype=uncompressed_data.dtype,
            device=uncompressed_data.device,
            requires_grad=False)
        self._is_param = True

        self.storage_format_cls = storage_format_cls
        self.compress_transposed = compress_transposed
        self.compressed_data = None

        self.is_empty = is_empty
        self.uncompressed_data = None if self.is_empty else uncompressed_data

        return self

    @property
    def has_compressed_data(self) -> bool:
        return (self.compressed_data is not None)

    @property
    def has_uncompressed_data(self) -> bool:
        if self.is_empty:
            raise ValueError(
                "has_uncompressed_data() was called with empty data")
        return self.uncompressed_data is not None

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        ret_storage_format_cls = None

        def unwrap(e):
            nonlocal ret_storage_format_cls
            if isinstance(e, LazyCompressedParameter):
                assert (ret_storage_format_cls is None
                        or ret_storage_format_cls == e.storage_format_cls)
                ret_storage_format_cls = e.storage_format_cls

                if e.is_empty:
                    e.is_empty = False
                    e.uncompressed_data = torch.empty(size=e.size(),
                                                      dtype=e.dtype,
                                                      device=e.device)

                return e.uncompressed_data
            else:
                return e

        rs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        def wrap(e):
            if isinstance(e,
                          torch.Tensor) and ret_storage_format_cls is not None:
                return LazyCompressedParameter(
                    e,
                    # Here, "e" is the output of "func" so it is real
                    # data and we store it
                    is_empty=False,
                    storage_format_cls=ret_storage_format_cls)
            return e

        rs = tree_map(wrap, rs)
        return rs

    def compress(self) -> None:
        from magic_wand import SparseSemiStructuredStorageFormat

        if self.storage_format_cls == SparseSemiStructuredStorageFormat:
            # Semi-structured sparsity assumes a 2:4 pattern, where
            # each 4 elements have at minimum 2 zeros. We need to validate
            # this pattern exists, so we check the whole tensor
            # before committing to compression.

            # Count zeros in each group of 4
            if self.uncompressed_data is None:
                raise ValueError(
                    "Uncompressed data must exist if calling .compress(),"
                    "but got self.uncompressed is None")
            reshaped_tensor = self.uncompressed_data.view(-1, 4)
            zeros = reshaped_tensor == 0
            zeros_per_group = zeros.sum(dim=1)

            # Check if each group has exactly 2 zeros
            has_semi_structured_sparsity = torch.all(zeros_per_group == 2)

            if not has_semi_structured_sparsity:
                logger.warning(
                    "Called compress() on tensor of shape %s but "
                    "does not have 2:4 sparsity, skipping compression",
                    self.shape)
                return

        else:
            sparsity = 1 - (torch.count_nonzero(self.uncompressed_data).item()
                            / numpy.prod(self.shape))

            # Only compress if we have sufficient sparsity (>=40%)
            if sparsity < 0.4:
                logger.warning(
                    "Called compress() on tensor of shape %s but "
                    "only has %s sparsity, skipping compression", self.shape,
                    sparsity)
                return

        if self.uncompressed_data is None:
            raise ValueError(
                "Called compress() but uncompressed_data does not exist.")
        self.compressed_data = self.storage_format_cls.compress(
            self.uncompressed_data.t(
            ) if self.compress_transposed else self.uncompressed_data)
        del self.uncompressed_data  # free memory
        self.uncompressed_data = None
