# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)

class LLFp8BlockScaledMMKernel(DeepGemmFp8BlockScaledMMKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        self._ll_available = False
        try:
            from vllm.model_executor.layers.fused_moe.router.ll_fp8_block_gemm import (
                is_available,
            )
            self._ll_available = is_available()
        except ImportError:
            pass

    @classmethod
    def can_implement(cls, config):
        return super().can_implement(config)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        M = A.shape[0]
        if self._ll_available and M <= 16:
            K_fp8 = A.shape[1]
            N = B.shape[0]
            if K_fp8 <= 4096 and K_fp8 % 256 == 0 and N <= 4096:
                output = torch.empty(
                    (M, N), dtype=self.config.out_dtype, device=A.device
                )
                torch.ops.vllm.ll_fp8_block_gemm_op(A, As, B, Bs, output)
                return output
        return super().apply_block_scaled_mm(A, B, As, Bs)


def _ll_fp8_block_gemm_op(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    from vllm.model_executor.layers.fused_moe.router.ll_fp8_block_gemm import (
        ll_fp8_block_gemm,
    )
    ll_fp8_block_gemm(q_input, input_scale, weight, weight_scale, output)


def _ll_fp8_block_gemm_op_fake(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    return None


from vllm.utils.torch_utils import direct_register_custom_op

direct_register_custom_op(
    "ll_fp8_block_gemm_op",
    _ll_fp8_block_gemm_op,
    mutates_args=["output"],
    fake_impl=_ll_fp8_block_gemm_op_fake,
)
