# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed._symmetric_memory import (
    enable_symm_mem_for_group,
)

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.sequence_parallel.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod


class FusedAllGatherGemm(FusedAllGatherGemmKernel):
    def __init__(self, layer: LinearBase) -> None:
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self.layer.quant_method.apply

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        if type(layer.quant_method) is not UnquantizedLinearMethod:
            return False, "requires UnquantizedLinearMethod"
        return True, None

    def target_gemm_method(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        W = layer.weight.t()
        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x, [W], gather_dim=0, group_name=self._group_name
        )
        output = mm_outputs[0]
        if bias is not None:
            output = output + bias
        return output


class FusedGemmReduceScatter(FusedGemmReduceScatterKernel):
    def __init__(self, layer: LinearBase) -> None:
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self.layer.quant_method.apply

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        if type(layer.quant_method) is not UnquantizedLinearMethod:
            return False, "requires UnquantizedLinearMethod"
        return True, None

    def target_gemm_method(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        W = layer.weight.t()
        output = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x, W, "sum", scatter_dim=0, group_name=self._group_name
        )
        return output
