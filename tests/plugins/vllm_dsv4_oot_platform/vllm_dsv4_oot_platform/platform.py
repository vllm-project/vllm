# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms.cuda import NvmlCudaPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class DSv4OOTPlatform(NvmlCudaPlatform):
    """Test-only OOT platform that piggybacks on CUDA infrastructure."""

    def is_out_of_tree(self) -> bool:
        return True

    @classmethod
    def support_deep_gemm(cls) -> bool:
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        super().check_and_update_config(vllm_config)

        if vllm_config.kernel_config.moe_backend == "auto":
            vllm_config.kernel_config.moe_backend = "triton"

