# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Optional

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.plugins import load_plugins_by_group
from vllm.plugins.hidden_states_processors.interface import (
    HiddenStatesProcessor)
from vllm.utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)


def identity_hidden_states_processor() -> str:
    return ("vllm.plugins.hidden_states_processors."
            "default.IdentityHiddenStatesProcessor")


default_hidden_states_processors = {
    "identity": identity_hidden_states_processor
}


def get_hidden_states_processor(
        vllm_config: VllmConfig) -> Optional["HiddenStatesProcessor"]:
    # hidden states processors are loaded as plugins under the
    # 'vllm.hidden_state_processor_plugins group. Similar to platform
    # plugins, these plugins register a function that returns the class
    # name for the processor to install.
    # All hidden state plugins implement the HiddenStatesProcessor class

    hidden_states_processor_plugins = \
        load_plugins_by_group('vllm.hidden_states_processor_plugins')

    available_plugins = {
        **default_hidden_states_processors,
        **hidden_states_processor_plugins
    }

    loadable_plugins = {}
    for name, func in available_plugins.items():
        try:
            assert callable(func)
            processor_cls_qualname = func()
            if processor_cls_qualname is not None:
                loadable_plugins[name] = processor_cls_qualname
        except Exception:
            pass

    num_available_plugins = len(loadable_plugins.keys())

    # Just a sanity check to make sure we are not
    # messing up with the available plugins
    assert num_available_plugins > 0

    if num_available_plugins > 1 and envs.VLLM_USE_HIDDEN_STATES_PROCESSOR:
        activated_plugin_cls = loadable_plugins[
            envs.VLLM_USE_HIDDEN_STATES_PROCESSOR]
        activated_plugin_name = envs.VLLM_USE_HIDDEN_STATES_PROCESSOR
    else:
        activated_plugin_name = list(loadable_plugins.keys())[0]
        activated_plugin_cls = loadable_plugins[activated_plugin_name]
        if (num_available_plugins > 1
                and not envs.VLLM_USE_HIDDEN_STATES_PROCESSOR):
            logger.info(
                "Multiple hidden states processor plugins available "
                "but VLLM_USE_HIDDEN_STATES_PROCESSOR is not pointing "
                "to any specific plugins. Loading the first available one.\n"
                "Available hidden states "
                "processor plugins %s", str(loadable_plugins.keys()))

    logger.info("Loaded hidden states processor plugin: %s",
                activated_plugin_name)
    return resolve_obj_by_qualname(activated_plugin_cls)(vllm_config)
