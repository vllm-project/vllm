import vllm.envs as envs
from vllm.logger import init_logger

plugins = envs.VLLM_PLUGINS

logger = init_logger(__name__)

for plugin in plugins:
    if not plugin.startswith("vllm_"):
        logger.info("Ignore invalid plugin: %s", plugin)
    else:
        logger.info("Importing plugin %s", plugin)
        __import__(plugin)
        logger.info("Plugin %s imported", plugin)
