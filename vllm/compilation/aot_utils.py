# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def uses_unwrapped_native_dp_ep(vllm_config: VllmConfig) -> bool:
    parallel_config = vllm_config.parallel_config
    effective_dp_size = (
        parallel_config.data_parallel_size
        if vllm_config.model_config is None or vllm_config.model_config.is_moe
        else 1
    )
    return (
        envs.VLLM_FUSED_MOE_WRAP_MODE == "unwrapped"
        and effective_dp_size > 1
        and parallel_config.enable_expert_parallel
    )


def should_use_aot_compile(vllm_config: VllmConfig) -> bool:
    if not envs.VLLM_USE_AOT_COMPILE:
        return False

    if uses_unwrapped_native_dp_ep(vllm_config):
        logger.info_once(
            "Disabling AOT compile for unwrapped MoE with native DP+EP "
            "because AOT Inductor artifacts are bound to a CUDA device index."
        )
        return False

    return True
