# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

logger = init_logger(__name__)


class MxFp8MoeBackend(Enum):
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"


def select_mxfp8_moe_backend(
    config: FusedMoEConfig,
) -> MxFp8MoeBackend:
    if config.is_lora_enabled:
        raise NotImplementedError("LoRA is not supported for MXFP8 MoE.")

    AVAILABLE_BACKENDS = [
        MxFp8MoeBackend.FLASHINFER_TRTLLM,
    ]

    runner_backend = config.moe_backend
    if runner_backend != "auto":
        mapping = {
            "flashinfer_trtllm": MxFp8MoeBackend.FLASHINFER_TRTLLM,
        }
        if backend := mapping.get(runner_backend):
            logger.info_once(
                "Using '%s' MxFp8 MoE backend (user-requested).",
                backend.value,
            )
            return backend
        raise ValueError(
            f"moe_backend='{runner_backend}' is not supported for MXFP8 MoE. "
            f"Expected one of {list(mapping.keys())}."
        )

    # Auto-select: only one backend available for now.
    backend = AVAILABLE_BACKENDS[0]
    logger.info_once("Using '%s' MxFp8 MoE backend.", backend.value)
    return backend
