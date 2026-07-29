# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup DFlash/DSpark spec-decode Triton kernels via VllmJitKernel wrapper.

Reads deployment-fixed scalars from the live speculator so Triton's
divisibility tags match runtime exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

logger = init_logger(__name__)


def dflash_kernel_warmup(speculator: DFlashSpeculator) -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        _PREPARE_DFLASH_INPUTS_KERNEL,
    )

    _PREPARE_DFLASH_INPUTS_KERNEL.warmup(speculator.vllm_config, speculator=speculator)
