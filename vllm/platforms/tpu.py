# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger

logger = init_logger(__name__)


try:
    from vllm_jax.platforms import (
        TpuPlatform as TpuInferencePlatform,
    )

    TpuPlatform = TpuInferencePlatform  # type: ignore
    USE_TPU_INFERENCE = True
except ImportError:
    try:
        from tpu_inference.platforms import (
            TpuPlatform as TpuInferencePlatform,
        )

        TpuPlatform = TpuInferencePlatform  # type: ignore
        USE_TPU_INFERENCE = True
    except ImportError:
        logger.error(
            "vllm_jax not found, please install vllm-jax-core to run vLLM on TPU"
        )
