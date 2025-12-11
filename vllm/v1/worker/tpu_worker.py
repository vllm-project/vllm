# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A TPU worker class."""

import os
from collections.abc import Callable
from typing import Any, TypeVar

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.platforms.tpu import USE_TPU_INFERENCE
from vllm.tasks import SupportedTask
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.utils import bind_kv_cache

logger = init_logger(__name__)

_R = TypeVar("_R")

if USE_TPU_INFERENCE:
    from tpu_inference.worker.tpu_worker import TPUWorker as TpuInferenceWorker

    TPUWorker = TpuInferenceWorker  # type: ignore
