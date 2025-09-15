# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils import DeviceMemoryProfiler, GiB_bytes

logger = init_logger(__name__)


def load_model(vllm_config: VllmConfig):
    time_before_load = time.perf_counter()

    with DeviceMemoryProfiler() as m:
        model_loader = get_model_loader(vllm_config.load_config)
        logger.info("Loading model from scratch...")
        model = model_loader.load_model(vllm_config=vllm_config,
                                        model_config=vllm_config.model_config)

    time_after_load = time.perf_counter()
    logger.info("Model loading took %.4f GiB and %.6f seconds",
                m.consumed_memory / GiB_bytes,
                time_after_load - time_before_load)
    return model
