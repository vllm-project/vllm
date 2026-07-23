# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up CuTeDSL mHC prenorm GEMM compile keys."""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def mhc_cutedsl_warmup(worker: "Worker") -> None:
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
        or not has_cutedsl()
    ):
        return

    from vllm.model_executor.kernels.mhc.cutedsl import (
        HC_PRENORM_GEMM_KERNEL,
    )

    HC_PRENORM_GEMM_KERNEL.warmup(worker.vllm_config)
