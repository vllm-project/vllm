# SPDX-License-Identifier: Apache-2.0
"""
LMCache Controller Configuration

Configuration system for LMCache Controller that:
- Loads configuration from YAML file or environment variables
- Supports command-line parameter overrides
- Provides thread-safe singleton pattern for global access
"""

# Standard
from typing import Any, Dict, Optional
import json

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config_base import (
    create_config_class,
    create_singleton_config,
    load_config_with_overrides,
)

logger = init_logger(__name__)


# Controller-specific configuration definitions
_CONTROLLER_CONFIG_DEFINITIONS: dict[str, dict[str, Any]] = {
    # Basic controller configurations
    "controller_monitor_ports": {
        "type": Optional[dict],
        "default": '{"pull": 8300, "reply": 8400}',
        "env_converter": lambda x: (
            x if isinstance(x, dict) else json.loads(x) if x else None
        ),
        "description": "JSON string of monitor ports",
    },
    "controller_host": {
        "type": str,
        "default": "0.0.0.0",
        "env_converter": str,
        "description": "Controller host address",
    },
    "controller_port": {
        "type": int,
        "default": 9000,
        "env_converter": int,
        "description": "Controller API server port",
    },
    "health_check_interval": {
        "type": int,
        "default": -1,
        "env_converter": int,
        "description": "Health check interval in seconds (-1 = disabled)",
    },
    "lmcache_worker_timeout": {
        "type": int,
        "default": 300,
        "env_converter": int,
        "description": "LMCache worker timeout in seconds",
    },
    # Extra configurations
    "extra_config": {
        "type": Optional[dict],
        "default": None,
        "env_converter": lambda x: (
            x if isinstance(x, dict) else json.loads(x) if x else None
        ),
        "description": "Extra configuration parameters",
    },
}


# Specialized methods that are unique to ControllerConfig
def _validate_config(self):
    """Validate configuration parameters"""
    # Validate timeouts
    if self.health_check_interval != -1 and self.health_check_interval < 1:
        raise ValueError(f"Invalid health_check_interval: {self.health_check_interval}")
    return self


def _log_config(self):
    """Log configuration"""
    config_dict = {}
    for name in _CONTROLLER_CONFIG_DEFINITIONS:
        value = getattr(self, name)
        config_dict[name] = value

    logger.info("Controller Configuration: %s", config_dict)
    return self


def _post_init(self):
    """Post-initialization setup"""
    pass


# Create configuration class using the base utility
ControllerConfig = create_config_class(
    config_name="ControllerConfig",
    config_definitions=_CONTROLLER_CONFIG_DEFINITIONS,
    namespace_extras={
        "validate": _validate_config,
        "log_config": _log_config,
        "__post_init__": _post_init,
    },
    env_prefix="LMCACHE_CONTROLLER_",
)


# Create singleton getter using the base utility
controller_get_or_create_config = create_singleton_config(
    getter_func_name="controller_get_or_create_config",
    config_class=ControllerConfig,
    config_env_var="LMCACHE_CONTROLLER_CONFIG_FILE",
)


def override_controller_config_from_dict(
    config: "ControllerConfig",  # type: ignore[valid-type]
    overrides: dict[str, Any],
):
    """Override configuration with dictionary"""
    for key, value in overrides.items():
        if hasattr(config, key):
            old_value = getattr(config, key)

            # Check if this field has an env_converter in the definitions
            if key in _CONTROLLER_CONFIG_DEFINITIONS:
                env_converter = _CONTROLLER_CONFIG_DEFINITIONS[key].get("env_converter")
                if env_converter:
                    try:
                        # Apply the env_converter to the value
                        converted_value = env_converter(value)
                        setattr(config, key, converted_value)
                    except (ValueError, json.JSONDecodeError) as e:
                        logger.warning("Failed to convert %s=%r: %s", key, value, e)
                        # Keep the original value if conversion fails
                        setattr(config, key, value)
                else:
                    setattr(config, key, value)
            else:
                setattr(config, key, value)

            new_value = getattr(config, key)
            if old_value != new_value:
                logger.info(
                    "Override controller config: %s = %s (was %s)",
                    key,
                    new_value,
                    old_value,
                )
        else:
            logger.warning("Unknown controller config key: %s, ignoring", key)


def load_controller_config_with_overrides(
    config_file_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> "ControllerConfig":  # type: ignore[valid-type]
    """
    Load controller configuration with support for file, env vars, and overrides.

    This function uses the generic load_config_with_overrides utility from
    config_base.py to reduce code duplication.

    Args:
        config_file_path: Optional direct path to config file
        overrides: Optional dictionary of configuration overrides

    Returns:
        Loaded and validated ControllerConfig instance
    """
    return load_config_with_overrides(
        config_class=ControllerConfig,
        config_file_env_var="LMCACHE_CONTROLLER_CONFIG_FILE",
        config_file_path=config_file_path,
        overrides=overrides,
    )
