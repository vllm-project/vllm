# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)
ENGINE_NAME = "sglang-instance"


def is_false(value: str) -> bool:
    """Check if the given string value is equivalent to 'false'."""
    return value.lower() in ("false", "0", "no", "n", "off")


def lmcache_get_config(config_file: str = "") -> LMCacheEngineConfig:
    """Load the LMCache configuration.

    Args:
        config_file: Path to a YAML configuration file, or empty string
            to build the config from ``LMCACHE_*`` environment variables.

    Returns:
        A validated ``LMCacheEngineConfig``.
    """
    if config_file:
        logger.info("Loading LMCache config file %s", config_file)
        config = LMCacheEngineConfig.from_file(config_file)
    else:
        config = LMCacheEngineConfig.from_defaults()
    config.validate()
    return config
