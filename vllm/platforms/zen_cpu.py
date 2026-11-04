# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform

logger = init_logger(__name__)


class ZenCpuPlatform(CpuPlatform):
    """CPU platform with AMD Zen (ZenDNN/zentorch) optimizations.

    Model-load time (dispatch_cpu_unquantized_gemm in layers/utils.py):
      - Routes linear ops to zentorch_linear_unary.
      - When VLLM_ZENTORCH_WEIGHT_PREPACK=1 (default), eagerly prepacks
        weights via zentorch_weight_prepack_for_linear.
    """

    device_name: str = "cpu"
    device_type: str = "cpu"

    def is_zen_cpu(self) -> bool:
        # is_cpu() also returns True for this platform (inherited from CpuPlatform).
        return True

    # fp16 compute needs AVX512-FP16, which AMD ships from Venice (6th Gen
    # EPYC) onwards; Turin (5th Gen EPYC) and earlier have no fp16 kernels.
    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        if torch.ops.zentorch.zentorch_is_fp16_supported():
            return [torch.bfloat16, torch.float32, torch.float16]
        return [torch.bfloat16, torch.float32]
