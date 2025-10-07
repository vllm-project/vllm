# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Collection

import torch


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


@contextlib.contextmanager
def set_default_torch_num_threads(num_threads: int):
    """Sets the default number of threads for PyTorch to the given value."""
    old_num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    yield
    torch.set_num_threads(old_num_threads)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


# bool = 0, int = 1, float = 2, complex = 3
def _get_precision_level(dtype: torch.dtype) -> int:
    # NOTE: Complex dtypes return `is_floating_point=False`
    return (dtype != torch.bool) + dtype.is_floating_point + dtype.is_complex * 2


def is_lossless_cast(src_dtype: torch.dtype, tgt_dtype: torch.dtype):
    """
    Test whether it is lossless to cast a tensor from
    `src_dtype` to `tgt_dtype`.
    """
    if src_dtype == tgt_dtype:
        return True

    src_level = _get_precision_level(src_dtype)
    tgt_level = _get_precision_level(tgt_dtype)

    if src_level < tgt_level:
        return True
    if src_level > tgt_level:
        return False

    # Compare integral types
    if not src_dtype.is_floating_point and not src_dtype.is_complex:
        src_info = torch.iinfo(src_dtype)
        tgt_info = torch.iinfo(tgt_dtype)
        return src_info.min >= tgt_info.min and src_info.max <= tgt_info.max

    # Compare floating-point types
    src_info = torch.finfo(src_dtype)
    tgt_info = torch.finfo(tgt_dtype)
    return (
        src_info.min >= tgt_info.min
        and src_info.max <= tgt_info.max
        and src_info.resolution >= tgt_info.resolution
    )


def common_broadcastable_dtype(dtypes: Collection[torch.dtype]):
    """
    Get the common `dtype` where all of the other `dtypes` can be
    cast to it without losing any information.
    """
    return max(
        dtypes,
        key=lambda dtype: sum(is_lossless_cast(dt, dtype) for dt in dtypes),
    )
