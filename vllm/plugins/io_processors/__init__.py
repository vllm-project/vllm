# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from typing import Optional

from vllm.config import VllmConfig
from vllm.plugins import load_plugins_by_group
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)


def get_io_processor(
        vllm_config: VllmConfig,
        plugin_from_init: Optional[str] = None) -> IOProcessor | None:
    # Input.Output processors are loaded as plugins under the
    # 'vllm.io_processor_plugins' group. Similar to platform
    # plugins, these plugins register a function that returns the class
    # name for the processor to install.

    if plugin_from_init:
        model_plugin = plugin_from_init
    else:
        # A plugin can be specified via the model config
        # Retrieve the model specific plugin if available
        # This is using a custom field in the hf_config for the model
        hf_config = vllm_config.model_config.hf_config.to_dict()
        config_plugin = hf_config.get("io_processor_plugin")
        model_plugin = config_plugin

    if model_plugin is None:
        logger.debug("No IOProcessor plugins requested by the model")
        return None

    logger.debug("IOProcessor plugin to be loaded %s", model_plugin)

    # Load all installed plugin in the group
    multimodal_data_processor_plugins = \
        load_plugins_by_group('vllm.io_processor_plugins')

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
        raise ValueError("No IOProcessor plugins installed"
                         f" but one is required ({model_plugin}).")

    if model_plugin not in loadable_plugins:
        raise ValueError(
            f"The model requires the '{model_plugin}' IO Processor plugin "
            "but it is not installed. "
            f"Available plugins: {list(loadable_plugins.keys())}")

    activated_plugin_cls = loadable_plugins[model_plugin]

    return resolve_obj_by_qualname(activated_plugin_cls)(vllm_config)
