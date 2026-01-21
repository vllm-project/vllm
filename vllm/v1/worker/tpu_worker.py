# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A TPU worker class."""

from typing import TypeVar

from vllm.logger import init_logger
from vllm.platforms.tpu import USE_TPU_INFERENCE

logger = init_logger(__name__)

_R = TypeVar("_R")

# TODO(weiyulin) Remove this file after adding an official way to use hardware plugin
if USE_TPU_INFERENCE:
    from tpu_inference.worker.tpu_worker import TPUWorker as TpuInferenceWorker

    TPUWorker = TpuInferenceWorker  # type: ignore
