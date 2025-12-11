# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from typing import TYPE_CHECKING, Optional, cast

import torch
from tpu_info import device

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.inputs import ProcessorInputs, PromptType
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from typing import TypeAlias

    from vllm.attention.selector import AttentionSelectorConfig
    from vllm.config import VllmConfig
    from vllm.config.cache import BlockSize
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams

    ParamsType: TypeAlias = SamplingParams | PoolingParams
else:
    BlockSize = None
    VllmConfig = None
    PoolingParams = None
    ParamsType = None

logger = init_logger(__name__)


try:
    from tpu_inference.platforms import (
        TpuPlatform as TpuInferencePlatform,
    )

    TpuPlatform = TpuInferencePlatform  # type: ignore
    USE_TPU_INFERENCE = True
except ImportError:
    logger.error("tpu_inference not found, please install tpu_inference to run vllm on TPU")
    pass
