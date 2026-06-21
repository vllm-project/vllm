# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel


class TritonFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "only CUDA-alike and XPU devices are supported."
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return w8a8_triton_block_scaled_mm(
            A,
            B,
            As,
            Bs,
            list(self.weight_group_shape),
            self.config.out_dtype,
        )
