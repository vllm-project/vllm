# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SP correctness of the unquantized fused and the generic unfused
comm+GEMM kernels against the plain TP linear."""

import pytest
import torch

from vllm.model_executor.kernels.sequence_parallel.base import (
    FusedGemmReduceScatterKernel,
    UnfusedAllGatherGemm,
    UnfusedGemmReduceScatter,
)
from vllm.model_executor.kernels.sequence_parallel.unquantized import (
    FusedAllGatherGemm,
    FusedGemmReduceScatter,
)
from vllm.platforms import current_platform

from .sequence_parallel_test_utils import FakeLinear, FakeQuantMethod, run_sp_matrix


def _build_unquantized(w: torch.Tensor):
    return FakeQuantMethod(), {"weight": w}


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="requires CUDA/ROCm/XPU",
)
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize(
    "kernel_cls,row_parallel",
    [
        (FusedAllGatherGemm, False),
        (FusedGemmReduceScatter, True),
        (UnfusedAllGatherGemm, False),
        (UnfusedGemmReduceScatter, True),
    ],
    ids=["ag", "rs", "unfused-ag", "unfused-rs"],
)
def test_unquantized(kernel_cls, row_parallel, world_size):
    run_sp_matrix(_build_unquantized, kernel_cls, row_parallel, world_size)


class _ScaleByTwoReduceScatter(FusedGemmReduceScatterKernel):
    """Comm-free stand-in to exercise the dispatcher on any device."""

    @classmethod
    def is_supported(cls, layer) -> tuple[bool, str | None]:
        return True, None

    def source_gemm_method(self):
        return self.layer.quant_method.apply

    def target_gemm_method(self, layer, x, bias=None):
        return x * 2


def test_apply_swaps_gemm_method_for_one_call():
    """`apply` must swap the GEMM method for the fused op (adding the RS bias
    once) only for that call, restoring it afterwards even on error."""

    class _Method(FakeQuantMethod):
        def apply(self, layer, x, bias=None):
            return x

    bias = torch.ones(4)
    layer = FakeLinear(_Method(), bias=bias)
    kernel = _ScaleByTwoReduceScatter(layer)
    x = torch.arange(4.0)

    torch.testing.assert_close(kernel.apply(x), x * 2 + bias)
    torch.testing.assert_close(layer(x), x)

    with pytest.raises(TypeError):
        kernel.apply(None)
    torch.testing.assert_close(layer(x), x)
