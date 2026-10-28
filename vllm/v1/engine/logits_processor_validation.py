# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from vllm.config import VllmConfig

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams


def build_logits_processors_params_validator(
    vllm_config: VllmConfig,
) -> Callable[["SamplingParams"], None]:
    """Load the custom logits processor classes and return the per-request
    params validator for the active model runner."""
    model_config = vllm_config.model_config
    custom_logitsprocs = model_config.logits_processors
    if vllm_config.use_v2_model_runner:
        if model_config.runner_type == "pooling" and not custom_logitsprocs:
            return lambda _: None

        from vllm.v1.worker.gpu.sample.logits_processor import (
            build_custom_logits_processors_params_validator,
        )

        return build_custom_logits_processors_params_validator(custom_logitsprocs)

    from vllm.v1.sample.logits_processor import (
        validate_logits_processors_parameters,
    )

    return partial(validate_logits_processors_parameters, custom_logitsprocs)
