# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placeholder linear kernels for online requantization sources.

A checkpoint quantization method used as an online requantization source only
dequantizes its weights, so it needs no kernel. The `init_*_linear_kernel`
selectors return one of these placeholders instead of dispatching, which keeps
their return types unchanged. Using one for compute raises.
"""

from typing import Any

import torch

from vllm.model_executor.kernels.linear.mxfp4 import MxFp4LinearKernel
from vllm.model_executor.kernels.linear.mxfp6 import MxFp6LinearKernel
from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearKernel
from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearKernel
from vllm.model_executor.kernels.linear.scaled_mm import (
    FP8ScaledMMLinearKernel,
    Int8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

_NO_KERNEL = (
    "Requantization sources only dequantize weights: they have no linear kernel."
)


class _FakeLinearKernel:
    """Satisfies the abstract interface of every kernel family, never computes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def is_supported(cls, *args: Any, **kwargs: Any) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, *args: Any, **kwargs: Any) -> tuple[bool, str | None]:
        return True, None

    def input_quant_key(self) -> QuantKey | None:
        return None

    def process_weights_after_loading(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(_NO_KERNEL)

    def apply_weights(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError(_NO_KERNEL)

    def apply_scaled_mm(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError(_NO_KERNEL)


class FakeFp8LinearKernel(_FakeLinearKernel, FP8ScaledMMLinearKernel):
    pass


class FakeInt8LinearKernel(_FakeLinearKernel, Int8ScaledMMLinearKernel):
    pass


class FakeMxFp8LinearKernel(_FakeLinearKernel, Mxfp8LinearKernel):
    pass


class FakeMxFp4LinearKernel(_FakeLinearKernel, MxFp4LinearKernel):
    pass


class FakeMxFp6LinearKernel(_FakeLinearKernel, MxFp6LinearKernel):
    pass


class FakeNvFp4LinearKernel(_FakeLinearKernel, NvFp4LinearKernel):
    pass
