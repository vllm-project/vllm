# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration management for Helion kernels.

This module provides centralized configuration file management for Helion custom
operations, including naming conventions, directory resolution, and file I/O.

Config File Structure
---------------------
Each kernel has a single JSON config file: {kernel_name}.json

The file uses a simplified 2-layer hierarchical structure:
{
    "h100": {                             # GPU platform
        "default": { ... },               # Fallback configuration
        "batch_32_hidden_4096": { ... },
        "batch_64_hidden_8192": { ... }
    },
    "a100": {
        "default": { ... },
        "batch_16_hidden_2048": { ... }
    }
}

Example file: silu_mul_fp8.json

Config keys should be structured strings that encode the relevant
parameters (e.g., "batch_32_hidden_4096", "seq_512_heads_16", "fp8_batch_64", etc.).

Classes
-------
- ConfigSet: In-memory collection of configs for a kernel with lookup/query APIs.
- ConfigManager: File-level operations for config persistence.
"""

import json
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "ConfigManager requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion

logger = init_logger(__name__)


class ConfigSet:
    """In-memory collection of Helion configs with lookup/query capabilities."""

    # Type alias for nested config structure:
    # platform -> config_key -> helion.Config
    _ConfigDict = dict[str, dict[str, "helion.Config"]]

    def __init__(self, kernel_name: str):
        self._kernel_name = kernel_name
        self._configs: ConfigSet._ConfigDict = {}

    @property
    def kernel_name(self) -> str:
        return self._kernel_name

    def get_config(self, platform: str, config_key: str) -> helion.Config:
        platform_dict = self._configs.get(platform)
        if platform_dict is None:
            avail_platforms = self.get_platforms()
            raise KeyError(
                f"Config not found for kernel '{self._kernel_name}': "
                f"platform '{platform}' not found. "
                f"Available platforms: {avail_platforms or '(none)'}"
            )

        config = platform_dict.get(config_key)
        if config is None:
            avail_keys = self.get_config_keys(platform)
            raise KeyError(
                f"Config not found for kernel '{self._kernel_name}': "
                f"config_key '{config_key}' not found for platform '{platform}'. "
                f"Available config_keys: {avail_keys or '(none)'}"
            )

        return config

    def get_platforms(self) -> list[str]:
        return sorted(self._configs.keys())

    def get_config_keys(self, platform: str) -> list[str]:
        platform_dict = self._configs.get(platform.lower())
        if platform_dict is None:
            return []
        return sorted(platform_dict.keys())

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for platform, config_keys_dict in self._configs.items():
            result[platform] = {}

            for config_key, config in config_keys_dict.items():
                # Convert helion.Config to dict using to_json() + json.loads()
                import json

                result[platform][config_key] = json.loads(config.to_json())

        return result

    @classmethod
    def from_dict(cls, kernel_name: str, data: dict[str, Any]) -> "ConfigSet":
        config_set = cls(kernel_name)
        count = 0

        for platform, platform_data in data.items():
            if platform not in config_set._configs:
                config_set._configs[platform] = {}

            for config_key, config_data in platform_data.items():
                config = helion.Config(**config_data)
                config_set._configs[platform][config_key] = config
                count += 1

        if count > 0:
            logger.debug(
                "Loaded %d configs for kernel '%s'",
                count,
                kernel_name,
            )

        return config_set


class ConfigManager:
    """File-level configuration management for Helion kernels (global singleton)."""

    _instance: "ConfigManager | None" = None
    _instance_base_dir: Path | None = None

    def __new__(cls, base_dir: str | Path | None = None) -> "ConfigManager":
        resolved_base_dir = cls._resolve_base_dir(base_dir)

        if cls._instance is not None:
            # Instance already exists - check for base_dir mismatch
            if cls._instance_base_dir != resolved_base_dir:
                raise ValueError(
                    f"ConfigManager singleton already exists with base_dir "
                    f"'{cls._instance_base_dir}', cannot create with different "
                    f"base_dir '{resolved_base_dir}'"
                )
            return cls._instance

        # Create new instance
        instance = super().__new__(cls)
        cls._instance = instance
        cls._instance_base_dir = resolved_base_dir
        return instance

    def __init__(self, base_dir: str | Path | None = None):
        # Only initialize if not already initialized
        if hasattr(self, "_base_dir"):
            return

        self._base_dir = self._resolve_base_dir(base_dir)
        logger.debug("ConfigManager initialized with base_dir: %s", self._base_dir)

    @staticmethod
    def _resolve_base_dir(base_dir: str | Path | None) -> Path:
        if base_dir is not None:
            return Path(base_dir).resolve()
        return (Path(__file__).parent / "configs").resolve()

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        if cls._instance is None:
            raise RuntimeError(
                "ConfigManager instance has not been created. "
                "Call ConfigManager(base_dir=...) first to initialize."
            )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """For testing purposes only."""
        cls._instance = None
        cls._instance_base_dir = None

    def get_config_file_path(self, kernel_name: str) -> Path:
        return self._base_dir / f"{kernel_name}.json"

    def ensure_base_dir_exists(self) -> Path:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        return self._base_dir

    def load_config_set(self, kernel_name: str) -> ConfigSet:
        config_path = self.get_config_file_path(kernel_name)
        if not config_path.exists():
            return ConfigSet.from_dict(kernel_name, {})

        try:
            with open(config_path) as f:
                data = json.load(f)
            return ConfigSet.from_dict(kernel_name, data)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load config file %s: %s", config_path, e)
            return ConfigSet.from_dict(kernel_name, {})

    def get_platform_configs(
        self, kernel_name: str, platform: str
    ) -> dict[str, helion.Config]:
        config_set = self.load_config_set(kernel_name)
        config_keys = config_set.get_config_keys(platform)

        return {
            config_key: config_set.get_config(platform, config_key)
            for config_key in config_keys
        }

    def save_config_set(self, config_set: ConfigSet) -> Path:
        config_path = self.get_config_file_path(config_set.kernel_name)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config_set.to_dict(), f, indent=2)

        logger.info("Saved config to: %s", config_path)
        return config_path
