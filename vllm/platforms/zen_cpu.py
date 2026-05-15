# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


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

    # Currently, AMD CPUs do not support float16 compute.
    # Hence explicitly return bfloat16 and float32.
    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float32]

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        super().check_and_update_config(vllm_config)

        import zentorch

        zentorch_version = getattr(zentorch, "__version__", "unknown")
        avx512 = torch.cpu._is_avx512_supported()
        avx512_bf16 = torch.cpu._is_avx512_bf16_supported()

        logger.info_once(
            "ZenCpuPlatform activated | zentorch=%s | "
            "VLLM_ZENTORCH_WEIGHT_PREPACK=%d | "
            "AVX-512=%s | AVX-512_BF16=%s",
            zentorch_version,
            int(envs.VLLM_ZENTORCH_WEIGHT_PREPACK),
            avx512,
            avx512_bf16,
        )

        zentorch_config = getattr(zentorch, "__config__", None)
        if zentorch_config:
            logger.info_once("zentorch build config: %s", zentorch_config)
