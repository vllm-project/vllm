import numpy
import torch
from torch.utils._pytree import tree_map

from typing import Type
from magic_wand import (CompressedStorageFormat, SparseBitmaskStorageFormat)


class LazyCompressedParameter(torch.Tensor):

    @staticmethod
    def __new__(cls,
                uncompressed_data: torch.Tensor,
                storage_format_cls: Type[
                    CompressedStorageFormat] = SparseBitmaskStorageFormat,
                compress_transposed: bool = False):
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            size=uncompressed_data.shape,
            dtype=uncompressed_data.dtype,
            requires_grad=False)
        self.storage_format_cls = storage_format_cls
        self.compressed_data = None
        self.uncompressed_data = uncompressed_data
        self.compress_transposed = compress_transposed
        self._is_param = True

        return self

    @property
    def has_compressed_data(self) -> bool:
        return (self.compressed_data is not None)

    @property
    def has_uncompressed_data(self) -> bool:
        return (self.uncompressed_data is not None)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        ret_storage_format_cls = None

        def unwrap(e):
            nonlocal ret_storage_format_cls
            if isinstance(e, LazyCompressedParameter):
                assert ret_storage_format_cls is None or ret_storage_format_cls == e.storage_format_cls
                ret_storage_format_cls = e.storage_format_cls
            return e.uncompressed_data if isinstance(
                e, LazyCompressedParameter) else e

        rs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        def wrap(e):
            if isinstance(e,
                          torch.Tensor) and ret_storage_format_cls is not None:
                return LazyCompressedParameter(
                    e, storage_format_cls=ret_storage_format_cls)
            return e

        rs = tree_map(wrap, rs)
        return rs

    def compress(self) -> None:
        density = torch.count_nonzero(
            self.uncompressed_data).item() / numpy.prod(self.shape)

        # only compress if we have sufficient sparsity (>=45%), currently
        # this applies globally across all formats including 2:4
        if (1 - density) < 0.45:
            return

        if self.uncompressed_data is None:
            raise ValueError(
                "Called compress() but uncompressed_data does not exist.")
        self.compressed_data = self.storage_format_cls.compress(
            self.uncompressed_data.t(
            ) if self.compress_transposed else self.uncompressed_data)
        del self.uncompressed_data  # free memory
        self.uncompressed_data = None
