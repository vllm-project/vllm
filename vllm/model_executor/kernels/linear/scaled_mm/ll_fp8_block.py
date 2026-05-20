# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)

_ll_available = False
try:
    from vllm.model_executor.layers.fused_moe.router.ll_fp8_block_gemm import (
        is_available,
    )
    _ll_available = is_available()
except ImportError:
    pass


class LLFp8BlockScaledMMKernel(DeepGemmFp8BlockScaledMMKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)

    @classmethod
    def can_implement(cls, config):
        return _ll_available and super().can_implement(config)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        output = torch.empty(
            (A.shape[0], B.shape[0]),
            dtype=out_dtype,
            device=A.device,
        )
        torch.ops.vllm.ll_fp8_block_dispatch_op(
            A, As, B, Bs, output, self.use_deep_gemm_e8m0
        )
        return output


def _ll_fp8_block_dispatch(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    M = q_input.shape[0]
    K_fp8 = q_input.shape[1]
    N = weight.shape[0]
    if M <= 16 and K_fp8 <= 4096 and K_fp8 % 256 == 0 and N <= 4096:
        from vllm.model_executor.layers.fused_moe.router.ll_fp8_block_gemm import (
            ll_fp8_block_gemm,
        )
        ll_fp8_block_gemm(q_input, input_scale, weight, weight_scale, output)
    else:
        from vllm.utils.deep_gemm import fp8_gemm_nt
        fp8_gemm_nt(
            (q_input, input_scale),
            (weight, weight_scale),
            output,
            is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
        )


def _ll_fp8_block_dispatch_fake(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    return None


from vllm.utils.torch_utils import direct_register_custom_op

direct_register_custom_op(
    "ll_fp8_block_dispatch_op",
    _ll_fp8_block_dispatch,
    mutates_args=["output"],
    fake_impl=_ll_fp8_block_dispatch_fake,
)
