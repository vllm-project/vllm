# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger

logger = init_logger(__name__)


try:
    from vllm_jax.platforms import TpuPlatform  # noqa: F401
except ImportError:
    try:
        from tpu_inference.platforms import TpuPlatform  # noqa: F401
    except ImportError:
        logger.error(
            "vllm_jax not found, please install vllm-jax-core to run vLLM on TPU"
        )
