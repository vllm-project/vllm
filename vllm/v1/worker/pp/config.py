# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping

from vllm.config import VllmConfig
from vllm.v1.worker.pp.boundary import DynamicPPConfig


def get_pp_config(config: VllmConfig) -> tuple[int, DynamicPPConfig]:
    additional = config.additional_config
    if not isinstance(additional, Mapping):
        return 0, DynamicPPConfig()
    size = additional.get("pp_send_buffer_size", 0)
    if type(size) is not int or size < 0:
        raise ValueError("pp_send_buffer_size must be a non-negative integer")
    if (
        size
        and config.model_config is not None
        and config.model_config.enable_sleep_mode
    ):
        raise ValueError("PP buffering does not yet support sleep mode")
    dynamic = DynamicPPConfig.from_mapping(additional.get("dynamic_pp_config"))
    dynamic.validate_runtime(config, size)
    if (
        size
        and config.parallel_config.distributed_executor_backend == "external_launcher"
    ):
        raise ValueError("PP buffering does not support external_launcher")
    return size, dynamic
