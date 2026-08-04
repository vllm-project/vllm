# SPDX-License-Identifier: Apache-2.0
"""
LMCache Configuration Base Module

This module provides common configuration utilities and base classes
for all LMCache configuration systems to avoid code duplication.
"""

# Standard
from dataclasses import make_dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Union
import ast
import json
import os
import threading
import uuid

# Third Party
import requests
import yaml

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _apply_env_converter_safely(config_definitions, name, value):
    """Apply env_converter to a value safely."""
    if name not in config_definitions:
        return value

    config = config_definitions[name]
    env_converter = config.get("env_converter")
    if env_converter:
        try:
            # Don't apply converter if value is None
            if value is None:
                return None
            return env_converter(value)
        except (ValueError, json.JSONDecodeError) as e:
            log_message = f"Failed to convert value for {name}={value!r}: {e}"
            logger.warning(log_message)
            # Return None if conversion fails
            return None
    return value


# Common configuration parsing utilities
def _parse_local_disk(local_disk) -> Optional[str]:
    """Parse local disk path configuration.

    Accepts a single path or comma-separated paths, each optionally
    prefixed with ``file://``.  Returns a comma-joined string of bare
    directory paths suitable for ``LocalDiskBackend``.

    Examples::

        "file:///mnt/nvme0/"              -> "/mnt/nvme0/"
        "/mnt/nvme0/,/mnt/nvme1/"         -> "/mnt/nvme0/,/mnt/nvme1/"
        "file:///mnt/nvme0/,file:///mnt/nvme1/"
                                           -> "/mnt/nvme0/,/mnt/nvme1/"

    Args:
        local_disk: Raw config value — ``None``, a single path, or
            comma-separated paths.

    Returns:
        Comma-joined bare directory paths, or ``None`` if disabled.
    """
    if local_disk is None:
        return None

    raw_parts = [p.strip() for p in str(local_disk).split(",") if p.strip()]
    if not raw_parts:
        return None

    parsed: list[str] = []
    for part in raw_parts:
        if part.startswith("file://"):
            parsed.append(part[7:])
        else:
            parsed.append(part)

    return ",".join(parsed)


def _to_int_list(
    value: Optional[Union[str, int, list[Any]]],
) -> Optional[list[int]]:
    """Convert value to list of integers"""
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, int):
        return [value]
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [int(p) for p in parts]


def _to_float_list(
    value: Optional[Union[str, float, list[Any]]],
) -> Optional[list[float]]:
    """Convert value to list of floats"""
    if value is None:
        return None
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, float):
        return [value]
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [float(p) for p in parts]


def _to_str_list(
    value: Optional[Union[str, list[str]]],
) -> Optional[list[str]]:
    """Convert value to list of strings"""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [p for p in parts]


def _to_bool(
    value: Optional[Union[bool, int, str]],
) -> bool:
    """Convert value to boolean"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ["true", "1"]


def _parse_quoted_string(value: str) -> str:
    """Parse a string that may be surrounded by quotes and handle escape characters.

    Args:
        value: The input string that may be quoted

    Returns:
        The unquoted string with escape characters properly handled
    """
    if not value:
        return value

    value = value.strip()

    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        try:
            evaluated = ast.literal_eval(value)
            if isinstance(evaluated, str):
                return evaluated
        except (ValueError, SyntaxError):
            # If ast.literal_eval fails, it's not a valid Python literal.
            # Fall back to simply stripping the outer quotes.
            return value[1:-1]

    return value


def _to_json(obj: Any) -> str:
    """Convert object to JSON string"""
    # If object has to_dict method, use it to convert to dict first
    if hasattr(obj, "to_dict"):
        return json.dumps(obj.to_dict(), indent=2)
    # Otherwise try to serialize directly
    return json.dumps(obj, indent=2)


def _from_json(cls, json_str: str):
    """Deserialize a JSON string into a configuration object."""
    try:
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON input: %s", e)
        raise


# Configuration aliases and deprecated mappings utility
def _resolve_config_aliases(
    config_dict: dict,
    source: str,
    config_definitions: dict,
    config_aliases: dict,
    deprecated_configs: dict,
) -> dict:
    """Resolve configuration aliases and handle deprecated configurations."""
    resolved = {}

    # Process each key in the input
    for key, value in config_dict.items():
        if key in deprecated_configs:
            # Log deprecation warning
            logger.warning("%s (source: %s)", deprecated_configs[key], source)

            # Map to new key if alias exists
            if key in config_aliases:
                new_key = config_aliases[key]
                resolved[new_key] = value
            else:
                # Keep deprecated key for backward compatibility
                resolved[key] = value
        elif key in config_definitions:
            # Valid configuration key
            resolved[key] = value
        else:
            # Unknown configuration key
            logger.warning("Unknown configuration key: %s (source: %s)", key, source)

    return resolved


# Base configuration class creator
def create_config_class(
    config_name: str,
    config_definitions: dict[str, dict[str, Any]],
    config_aliases: Optional[dict[str, str]] = None,
    deprecated_configs: Optional[dict[str, str]] = None,
    namespace_extras: Optional[dict[str, Any]] = None,
    env_prefix: str = "LMCACHE_",
):
    """Create a configuration class dynamically with common functionality.

    Args:
        config_name: Name of the configuration class
        config_definitions: Dictionary of configuration definitions
        config_aliases: Optional mapping of deprecated names to current names
        deprecated_configs: Optional mapping of deprecated names to warning messages
        namespace_extras: Optional additional namespace items for the class
        env_prefix: Environment variable prefix (default: "LMCACHE_")

    Returns:
        A dynamically created dataclass with configuration functionality
    """
    # Default values
    config_aliases = config_aliases or {}
    deprecated_configs = deprecated_configs or {}
    namespace_extras = namespace_extras or {}

    # Extract fields from configuration definitions
    fields_dict = {}
    for name, config in config_definitions.items():
        fields_dict[name] = (config["type"], config["default"])

    def _post_init(self):
        """Post-initialization setup"""
        # Initialize user-set keys tracking set
        # This tracks which config keys were explicitly set by user
        # (via file, env vars, or overrides) vs. using default values
        if not hasattr(self, "_user_set_keys"):
            object.__setattr__(self, "_user_set_keys", set())
        # Generate instance ID if not set
        if not getattr(self, "lmcache_instance_id", None):
            self.lmcache_instance_id = f"{config_name.lower()}_{uuid.uuid4().hex}"

    def _from_env(cls):
        """Load configuration from environment variables"""

        def get_env_name(attr_name: str) -> str:
            return f"{env_prefix}{attr_name.upper()}"

        # Collect all defined and deprecated env vars
        all_keys = list(config_definitions.keys()) + list(config_aliases.keys())
        env_config = {}
        for name in all_keys:
            env_name = get_env_name(name)
            env_value = os.getenv(env_name)
            if env_value is not None:
                env_config[name] = env_value

        # Resolve aliases and handle deprecated configurations
        resolved_config = _resolve_config_aliases(
            env_config,
            "environment variables",
            config_definitions,
            config_aliases,
            deprecated_configs,
        )

        config_values = {}
        user_set_keys = set()  # Track keys explicitly set by user
        for name, config in config_definitions.items():
            if name in resolved_config:
                try:
                    raw_value = resolved_config[name]
                    value = _parse_quoted_string(raw_value)
                    # Apply env_converter safely
                    config_values[name] = _apply_env_converter_safely(
                        config_definitions, name, value
                    )
                    user_set_keys.add(name)  # Mark as user-set
                except (ValueError, json.JSONDecodeError) as e:
                    raw_value_for_log = resolved_config.get(name, "unknown value")
                    log_message = (
                        f"Failed to parse {get_env_name(name)}"
                        f"={raw_value_for_log!r}: {e}"
                    )
                    logger.warning(log_message)
                    # Use default value with conversion
                    config_values[name] = _apply_env_converter_safely(
                        config_definitions, name, config["default"]
                    )
            else:
                # Use default value with conversion
                config_values[name] = _apply_env_converter_safely(
                    config_definitions, name, config["default"]
                )

        instance = cls(**config_values)
        # Store user-set keys in the instance
        object.__setattr__(instance, "_user_set_keys", user_set_keys)
        return instance

    def _from_file(cls, file_path: str):
        """Load configuration from file"""
        with open(file_path, "r") as fin:
            file_config = yaml.safe_load(fin) or {}

        # Resolve aliases and handle deprecated configurations
        resolved_config = _resolve_config_aliases(
            file_config,
            f"file: {file_path}",
            config_definitions,
            config_aliases,
            deprecated_configs,
        )

        config_values = {}
        user_set_keys = set()  # Track keys explicitly set by user
        for name, config in config_definitions.items():
            if name in resolved_config:
                value = resolved_config[name]
                user_set_keys.add(name)  # Mark as user-set
            else:
                value = config["default"]
            # Apply env_converter safely regardless of whether value is None or not
            config_values[name] = _apply_env_converter_safely(
                config_definitions, name, value
            )

        instance = cls(**config_values)
        # Store user-set keys in the instance
        object.__setattr__(instance, "_user_set_keys", user_set_keys)
        return instance

    def _from_defaults(cls, **kwargs):
        """Create configuration from defaults"""
        config_values = {}
        user_set_keys = set()  # Track keys explicitly set by user
        for name, config in config_definitions.items():
            if name in kwargs:
                value = kwargs[name]
                user_set_keys.add(name)  # Mark as user-set
            else:
                value = config["default"]
            # Apply env_converter safely regardless of whether value is None or not
            config_values[name] = _apply_env_converter_safely(
                config_definitions, name, value
            )

        instance = cls(**config_values)
        # Store user-set keys in the instance
        object.__setattr__(instance, "_user_set_keys", user_set_keys)
        return instance

    def _update_config_from_env(self):
        """Update an existing config object with environment variable configurations."""

        def get_env_name(attr_name: str) -> str:
            return f"{env_prefix}{attr_name.upper()}"

        env_config = {}
        # Collect all defined and deprecated env vars
        all_keys = list(config_definitions.keys()) + list(config_aliases.keys())
        for name in all_keys:
            env_name = get_env_name(name)
            env_value = os.getenv(env_name)
            if env_value is not None:
                env_config[name] = env_value

        # Resolve aliases
        resolved_config = _resolve_config_aliases(
            env_config,
            "environment variables",
            config_definitions,
            config_aliases,
            deprecated_configs,
        )

        # Ensure _user_set_keys exists
        if not hasattr(self, "_user_set_keys"):
            object.__setattr__(self, "_user_set_keys", set())

        # Update config object
        for name, config in config_definitions.items():
            if name in resolved_config:
                try:
                    raw_value = resolved_config[name]
                    value = _parse_quoted_string(raw_value)
                    converted_value = config["env_converter"](value)
                    setattr(self, name, converted_value)
                    # Mark as user-set
                    self._user_set_keys.add(name)
                except (ValueError, json.JSONDecodeError) as e:
                    raw_value_for_log = resolved_config.get(name, "unknown value")
                    log_message = (
                        f"Failed to parse {get_env_name(name)}"
                        f"={raw_value_for_log!r}: {e}"
                    )
                    logger.warning(log_message)

        return self

    def _from_dict(cls, config_dict: dict):
        """Create configuration from a dictionary."""
        resolved_config = _resolve_config_aliases(
            config_dict,
            "dictionary input",
            config_definitions,
            config_aliases,
            deprecated_configs,
        )
        config_values = {}
        user_set_keys = set()  # Track keys explicitly set by user
        for name, config in config_definitions.items():
            if name in resolved_config:
                value = resolved_config[name]
                user_set_keys.add(name)  # Mark as user-set
            else:
                value = config["default"]
            if value is not None:
                value = config["env_converter"](value)
            config_values[name] = value
        instance = cls(**config_values)
        # Store user-set keys in the instance
        object.__setattr__(instance, "_user_set_keys", user_set_keys)
        return instance

    def _to_dict(self):
        """Convert the configuration object into a dictionary."""
        return {name: getattr(self, name) for name in config_definitions}

    # Build namespace
    namespace = {
        "__post_init__": _post_init,
        "from_defaults": classmethod(_from_defaults),
        "from_file": classmethod(_from_file),
        "from_env": classmethod(_from_env),
        "update_config_from_env": _update_config_from_env,
        "from_dict": classmethod(_from_dict),
        "to_dict": _to_dict,
        "to_json": _to_json,
        "from_json": classmethod(_from_json),
        "__str__": lambda self: str(
            {name: getattr(self, name) for name in config_definitions}
        ),
    }

    # Add extra namespace items
    namespace.update(namespace_extras)

    # Create class
    cls = make_dataclass(
        config_name,
        [(name, type_, default) for name, (type_, default) in fields_dict.items()],
        namespace=namespace,
    )

    # Add config_definitions as a class attribute for accessing converters
    cls._config_definitions = config_definitions  # type: ignore[attr-defined]

    return cls


# Thread-safe singleton utility
class SingletonGetter(Protocol):
    """Protocol for singleton getter functions"""

    def __call__(self) -> Any: ...

    reset: Callable[[], None]


def create_singleton_config(
    getter_func_name: str,
    config_class,
    config_env_var: str = "LMCACHE_CONFIG_FILE",
) -> SingletonGetter:
    """Create thread-safe singleton configuration access pattern.

    Args:
        getter_func_name: Name for the singleton getter function
        config_class: The configuration class to create singleton for
        config_env_var: Environment variable name for configuration file path
    """

    _config_instance = None
    _config_lock = threading.Lock()

    def get_or_create_config() -> config_class:
        """Get the configuration singleton"""
        nonlocal _config_instance

        # Double-checked locking for thread-safe singleton
        if _config_instance is None:
            with _config_lock:
                if _config_instance is None:  # Check again within lock
                    if config_env_var not in os.environ:
                        logger.warning(
                            "No configuration file is set. Trying to read "
                            "configurations from the environment variables."
                        )
                        logger.warning(
                            f"You can set the configuration file through "
                            f"the environment variable: {config_env_var}"
                        )
                        _config_instance = config_class.from_env()
                    else:
                        config_file = os.environ[config_env_var]
                        logger.info("Loading config file %s", config_file)
                        _config_instance = config_class.from_file(config_file)
                        # Update config from environment variables
                        _config_instance.update_config_from_env()

        return _config_instance

    def reset_config_instance() -> None:
        """Reset the configuration singleton for testing"""
        nonlocal _config_instance
        with _config_lock:
            _config_instance = None

    # Set the function name for better debugging
    get_or_create_config.__name__ = getter_func_name
    get_or_create_config.reset = reset_config_instance  # type: ignore[attr-defined]

    return get_or_create_config  # type: ignore[return-value]


def load_config_with_overrides(
    config_class,
    config_file_env_var: str = "LMCACHE_CONFIG_FILE",
    config_file_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Load configuration with support for file, environment variables, and overrides.

    This is a generic utility function that can be reused across different
    configuration classes (LMCacheEngineConfig, ControllerConfig, etc.)

    Args:
        config_class: The configuration class to instantiate
        config_file_env_var: Environment variable name for config file path
        config_file_path: Optional direct config file path (overrides env var)
        overrides: Optional dictionary of configuration overrides

    Returns:
        Loaded and validated configuration instance
    """
    # Load configuration from file or environment
    actual_config_path = config_file_path or os.getenv(config_file_env_var)

    if actual_config_path:
        logger.info("Loading config file: %s", actual_config_path)
        config = config_class.from_file(actual_config_path)
        # Allow environment variables to override file settings
        config.update_config_from_env()
    else:
        logger.info("No config file specified, loading from environment variables.")
        config = config_class.from_env()

    # Apply any overrides
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                old_value = getattr(config, key)

                # Check if this configuration class has definitions with converters
                if (
                    hasattr(config, "_config_definitions")
                    and key in config._config_definitions
                ):
                    # Use the global helper function to safely apply env_converter
                    new_value = _apply_env_converter_safely(
                        config._config_definitions, key, value
                    )
                    setattr(config, key, new_value)
                else:
                    setattr(config, key, value)

                new_value = getattr(config, key)
                if old_value != new_value:
                    logger.info(
                        "Override config: %s = %s (was %s)", key, new_value, old_value
                    )
            else:
                logger.warning("Unknown config key: %s, ignoring", key)

    # Validate configuration
    if hasattr(config, "validate"):
        config.validate()

    # Log configuration
    if hasattr(config, "log_config"):
        config.log_config()

    return config


def parse_command_line_extra_params(extra_args: list[str]) -> dict[str, Any]:
    """
    Parse extra command-line parameters in key=value format.

    Args:
        extra_args: List of strings in format "key=value"

    Returns:
        Dictionary of parsed parameters
    """
    params = {}
    for arg in extra_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lstrip("-")
            try:
                if value.lower() in ("true", "false"):
                    params[key] = value.lower() == "true"
                elif value.isdigit():
                    params[key] = int(value)  # type: ignore[assignment]
                elif value.replace(".", "", 1).isdigit():
                    params[key] = float(value)  # type: ignore[assignment]
                else:
                    params[key] = value  # type: ignore[assignment]
            except ValueError:
                params[key] = value  # type: ignore[assignment]
            logger.info("Extra parameter: %s = %s", key, params[key])
    return params


def validate_and_set_config_value(config, config_key, value, override: bool = True):
    """Validate and set configuration value.

    Args:
        config: Configuration object to update.
        config_key: The configuration key to set.
        value: The value to set.
        override: If True, completely replace the value. If False:
            - For 'extra_config': merge with existing dict (new values take
              precedence for conflicting keys).
            - For other keys: skip if key was user-set.
            Default is True.

    Returns:
        True if the value was set successfully, False otherwise.
    """
    if not hasattr(config, config_key):
        logger.warning("Config key '%s' does not exist in configuration", config_key)
        return False

    try:
        # Apply type conversion using env_converter if available
        # Skip for extra_config which has special handling below
        if hasattr(config, "_config_definitions") and config_key != "extra_config":
            original_value = value
            value = _apply_env_converter_safely(
                config._config_definitions, config_key, value
            )
            # If original value was not None but conversion returned None,
            # it means conversion failed
            if original_value is not None and value is None:
                # _apply_env_converter_safely already logged warning
                return False

        # Convert string to dict for extra_config
        if config_key == "extra_config" and isinstance(value, str):
            value = json.loads(value) if value else None

        # Handle extra_config special logic
        if config_key == "extra_config":
            current_value = getattr(config, config_key, None)

            # Handle None or empty value when override=False
            if not override and (value is None or value == ""):
                # Keep current value
                logger.info(
                    "Keeping current extra_config (override=False, value is None/empty)"
                )
                return True

            if override:
                # Full override: replace current value with new value
                setattr(config, config_key, value)
                logger.info("Overridden extra_config")
                return True
            else:
                # Merge mode (override=False)
                if value is not None and isinstance(value, dict):
                    if current_value is not None and isinstance(current_value, dict):
                        # Merge: current values preserved, new values override
                        merged_value = {**current_value, **value}
                        setattr(config, config_key, merged_value)
                        logger.info("Merged extra_config")
                    else:
                        # Current value is None or not a dict, set new value
                        setattr(config, config_key, value)
                        logger.info("Set extra_config")
                return True

        # For non-extra_config keys: skip if override=False and key user-set
        if not override:
            user_set_keys: set[str] = getattr(config, "_user_set_keys", set())
            if config_key in user_set_keys:
                current_value = getattr(config, config_key, None)
                logger.info(
                    "Skipping config %s (override=False, user-set value=%s)",
                    config_key,
                    current_value,
                )
                # Return True to indicate operation completed (kept existing)
                return True

        setattr(config, config_key, value)
        logger.info("Set config item '%s'", config_key)
        return True
    except Exception as e:
        logger.error(
            "Failed to set config item '%s' with value %s: %s",
            config_key,
            value,
            e,
        )
        return False


def fetch_remote_config(
    remote_config_url: str,
    app_id: Optional[str],
    config: Any,
    timeout: int = 10,
) -> Optional[dict]:
    """Fetch configuration from remote config service.

    The config server protocol:
    - Request: POST with JSON body containing 'current_config' and 'env_variables'
    - Query parameter: 'appId' if app_id is provided
    - Response: JSON with 'configs' array, each item has 'key', 'value', 'override'

    See examples/remote_config_server/ for a reference implementation.

    Args:
        remote_config_url: URL of the remote config service.
        app_id: Optional app ID to send to the config service.
        config: Current LMCacheEngineConfig to send to the config service.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response from the config service, or None if failed.
    """
    try:
        # Build request payload with current config and env variables
        payload: dict[str, Any] = {
            "current_config": config.to_dict(),
            "env_variables": {k: v for k, v in os.environ.items()},
        }

        # Build URL with app_id query parameter if provided
        params = {"appId": app_id} if app_id else None

        # Send POST request with JSON payload
        response = requests.get(
            remote_config_url,
            json=payload,
            params=params,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        logger.warning(
            "Failed to fetch remote config from %s: %s", remote_config_url, e
        )
        return None
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse remote config response: %s", e)
        return None
    except Exception as e:
        logger.warning("Unexpected error fetching remote config: %s", e)
        return None


def apply_remote_configs(config: Any, remote_response: dict) -> Any:
    """Apply remote configuration to LMCacheEngineConfig.

    This function extracts the 'configs' field from the remote response
    and applies each config item to the LMCacheEngineConfig instance.

    The expected format of remote_response['configs'] is:
    [
        {"override": true, "key": "config_key", "value": "config_value"},
        ...
    ]

    Args:
        config: LMCacheEngineConfig instance to update.
        remote_response: Response from the remote config service.

    Returns:
        Updated LMCacheEngineConfig instance.
    """
    configs = remote_response.get("configs", [])
    if not configs:
        logger.info("No configs found in remote response")
        return config

    applied_count = 0
    for config_item in configs:
        if not isinstance(config_item, dict):
            logger.warning("Invalid config item format: %s", config_item)
            continue

        key = config_item.get("key")
        value = config_item.get("value")
        override = config_item.get("override", True)

        if not key:
            logger.warning("Config item missing 'key': %s", config_item)
            continue

        # Try to convert value to appropriate type
        if validate_and_set_config_value(config, key, value, override=override):
            logger.info("Applied remote config: %s=%s", key, value)
            applied_count += 1
        else:
            logger.warning(
                "Failed to apply remote config %s=%s. Using default value.",
                key,
                value,
            )

    logger.info("Applied %d remote configuration items", applied_count)
    return config
