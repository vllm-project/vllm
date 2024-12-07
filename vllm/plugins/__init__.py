import logging
import os

import torch

import vllm.envs as envs
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

# make sure one process only loads plugins once
plugins_loaded = False


def load_general_plugins():
    """WARNING: plugins can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """

    # all processes created by vllm will load plugins,
    # and here we can inject some common environment variables
    # for all processes.

    # see https://github.com/vllm-project/vllm/issues/10480
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    # see https://github.com/vllm-project/vllm/issues/10619
    torch._inductor.config.compile_threads = 1
    if current_platform.is_xpu():
        # see https://github.com/pytorch/pytorch/blob/8cada5cbe5450e17c26fb8b358116785324537b2/torch/_dynamo/config.py#L158  # noqa
        os.environ['TORCH_COMPILE_DISABLE'] = 'True'
    if current_platform.is_hpu():
        # NOTE(kzawora): PT HPU lazy backend (PT_HPU_LAZY_MODE = 1)
        # does not support torch.compile
        # Eager backend (PT_HPU_LAZY_MODE = 0) must be selected for
        # torch.compile support
        is_lazy = os.environ.get('PT_HPU_LAZY_MODE', '1') == '1'
        if is_lazy:
            # see https://github.com/pytorch/pytorch/blob/43c5f59/torch/_dynamo/config.py#L158
            torch._dynamo.config.disable = True
            # NOTE(kzawora) multi-HPU inference with HPUGraphs (lazy-only)
            # requires enabling lazy collectives
            # see https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html # noqa: E501
            os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = 'true'

    global plugins_loaded
    if plugins_loaded:
        return
    plugins_loaded = True
    import sys
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    allowed_plugins = envs.VLLM_PLUGINS

    discovered_plugins = entry_points(group='vllm.general_plugins')
    if len(discovered_plugins) == 0:
        logger.debug("No plugins found.")
        return
    logger.info("Available plugins:")
    for plugin in discovered_plugins:
        logger.info("name=%s, value=%s, group=%s", plugin.name, plugin.value,
                    plugin.group)
    if allowed_plugins is None:
        logger.info("all available plugins will be loaded.")
        logger.info("set environment variable VLLM_PLUGINS to control"
                    " which plugins to load.")
    else:
        logger.info("plugins to load: %s", allowed_plugins)
    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            try:
                func = plugin.load()
                func()
                logger.info("plugin %s loaded.", plugin.name)
            except Exception:
                logger.exception("Failed to load plugin %s", plugin.name)
