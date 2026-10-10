# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import functools

import torch


def token_to_batch_idx(cu_lens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Map each of `num_tokens` flattened tokens to its row in `cu_lens`.

    `cu_lens` is a `[num_reqs + 1]` cumulative-length tensor starting at 0.
    """
    tokens = torch.arange(num_tokens, dtype=cu_lens.dtype)
    return torch.searchsorted(cu_lens[1:], tokens, right=True)


@functools.lru_cache(maxsize=1024)
def ptr_view(addr: int, dtype: torch.dtype, numel: int) -> torch.Tensor:
    """Zero-copy tensor over `numel` elements at host address `addr`.

    Kernels that address several tensors through a pointer array get raw
    addresses; on CPU these are host addresses and can be viewed directly.
    """
    buf = (ctypes.c_byte * (numel * dtype.itemsize)).from_address(addr)
    return torch.frombuffer(buf, dtype=dtype)
