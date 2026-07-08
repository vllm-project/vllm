# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger

logger = init_logger(__name__)


try:
    from tpu_inference.platforms import (
        TpuPlatform as TpuInferencePlatform,
    )

    TpuPlatform = TpuInferencePlatform  # type: ignore
    USE_TPU_INFERENCE = True
except ImportError:
    logger.error(
        "tpu_inference not found, please install tpu_inference to run vllm on TPU"
    )
    pass
