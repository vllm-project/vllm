# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Vendored subset of ``atom.utils`` needed by the ported DeepSeek V4 attention.

Only ``CpuGpuBuffer`` (paired pinned-host + device staging buffer) and the
``envs`` submodule are required by the ported kernels / bridge; the rest of
ATOM's ``utils`` package (zmq/psutil/loader machinery) is intentionally not
vendored to keep this import surface self-contained inside vLLM.
"""

from typing import Any, Callable, Optional, Union

import numpy as np
import torch

from vllm.models.deepseek_v4.amd.atom.utils import envs

__all__ = ["CpuGpuBuffer", "envs", "mark_spliting_op"]


def mark_spliting_op(
    is_custom: bool,
    gen_fake: Optional[Callable[..., Any]] = None,
    mutates_args: Optional[list] = None,
):
    """Register a graph-splitting custom op (verbatim port of ``atom.utils``).

    Used to register ``torch.ops.aiter.v4_attention_with_output`` as a
    Dynamo-opaque splitting op. When ``is_custom`` is False it just tags the
    function; when True it registers a real custom op via
    ``direct_register_custom_op`` and marks the registered op as a splitting op.
    """
    if mutates_args is None:
        mutates_args = []

    from vllm.models.deepseek_v4.amd.atom.utils.custom_register import (
        direct_register_custom_op,
    )

    def decorator(func):
        if not is_custom:
            func.spliting_op = True
            return func

        direct_register_custom_op(
            op_name=func.__name__,
            op_func=func,
            mutates_args=mutates_args,
            fake_impl=gen_fake,
        )
        registered_op = getattr(torch.ops.aiter, func.__name__)
        registered_op.spliting_op = True
        return func

    return decorator


class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU.

    Verbatim port of ``atom.utils.CpuGpuBuffer``: a pinned host tensor with a
    zero-copy ``.np`` numpy view plus a device mirror, with ``copy_to_gpu(n)`` /
    ``copy_to_cpu(n)`` non-blocking length-sliced transfers.
    """

    def __init__(
        self,
        *size: Union[int, "torch.SymInt"],
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = True,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.gpu = torch.zeros_like(self.cpu, device=device)
        self.np: np.ndarray
        # To keep type hints simple (avoiding generics and subclasses), we
        # only conditionally create the numpy array attribute. This can cause
        # AttributeError if `self.np` is accessed when `with_numpy=False`.
        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call CpuGpuBuffer with with_numpy=False"
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: Optional[int] = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    def copy_to_cpu(self, n: Optional[int] = None) -> torch.Tensor:
        """NOTE: Because this method is non-blocking, explicit synchronization
        is needed to ensure the data is copied to CPU."""
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)
