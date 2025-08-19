# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.plugins import load_plugins_by_group
from vllm.plugins.multimodal_data_processors.interface import (
    MultimodalDataProcessor)
from vllm.plugins.multimodal_data_processors.types import (
    MultiModalRequestOutput)
from vllm.utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)


def get_multimodal_data_processor(
        vllm_config: VllmConfig) -> MultimodalDataProcessor | None:
    # Multimodal processors are loaded as plugins under the
    # 'vllm.multimodal_data_processor_plugins' group. Similar to platform
    # plugins, these plugins register a function that returns the class
    # name for the processor to install.

    # Retrieve the model specific plugin if available
    # This is using a custom field in the hf_config for the model

    if envs.VLLM_USE_MULTIMODAL_DATA_PROCESSOR_PLUGIN:
        # A plugin is specified ad startup via env variable
        model_plugin = envs.VLLM_USE_MULTIMODAL_DATA_PROCESSOR_PLUGIN
    else:
        # A plugin is specified via the model config
        hf_config = vllm_config.model_config.hf_config.to_dict()
        model_plugin = hf_config.get("multimodal_processor_plugin")

    logger.debug("MultiModalProcessor plugin to be loaded %s", model_plugin)

    if not model_plugin:
        logger.info("No MultiModalProcessor plugins installed")
        return None

    # Load all installed plugin in the group
    multimodal_data_processor_plugins = \
        load_plugins_by_group('vllm.multimodal_data_processor_plugins')

    loadable_plugins = {}
    for name, func in multimodal_data_processor_plugins.items():
        try:
            assert callable(func)
            processor_cls_qualname = func()
            if processor_cls_qualname is not None:
                loadable_plugins[name] = processor_cls_qualname
        except Exception:
            logger.warning("Failed to load plugin %s.", name, exc_info=True)

    num_available_plugins = len(loadable_plugins.keys())
    if num_available_plugins == 0:
        raise ValueError("No MultimodalProcessor plugins installed"
                         " but one is required.")

    if model_plugin not in loadable_plugins:
        raise ValueError(
            f"The model requires the '{model_plugin}' plugin but it"
            " is not installed. "
            f"Available plugins: {list(loadable_plugins.keys())}")

    activated_plugin_cls = loadable_plugins[model_plugin]

    return resolve_obj_by_qualname(activated_plugin_cls)(vllm_config)


def multimodal_plugin_outputs_to_pooling_output(
    plugin_out: Sequence[MultiModalRequestOutput], ):

    # Here I am assuming that the only field we care about is the
    # task out and the request+id. This is beause we are not really pooling
    # but rather wrapping non text-generating models as pooling ones.
    # In the future we can also think of aggregating the pooler output
    # for all the sub requests into one single PoolerRequestOutput object.
    outputs = []
    for output in plugin_out:
        outputs.append(
            PoolingRequestOutput(
                request_id=output.request_id,
                finished=True,
                outputs=PoolingOutput(data=torch.empty([])),
                prompt_token_ids=[1],
                task_output=output,
            ))

    return outputs
