# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration management for Helion kernels.

This module provides centralized configuration file management for Helion custom
operations, including naming conventions, directory resolution, and file I/O.

Config File Structure
---------------------
Each kernel has a directory: {kernel_name}/
Inside, each GPU platform has its own JSON file: {kernel_name}/{platform}.json

For example:
    silu_mul_fp8/
        nvidia_h100.json    # { "default": {...}, "batch_32_hidden_4096": {...} }
        nvidia_h200.json    # { "batch_16_hidden_2048": {...} }

Each platform file maps config keys to Helion config objects.
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
            # TODO(@gmagogsfm): add a CLI/env override flag so users can
            # directly specify a platform name instead of relying on
            # auto-detection, and suggest it in this error message.
            raise KeyError(
                f"Config not found for kernel '{self._kernel_name}': "
                f"platform '{platform}' not found. "
                f"Available platforms: {avail_platforms or '(none)'}. "
                f"If your GPU is a variant of a supported platform, "
                f"consider adding a mapping in _GPU_NAME_ALIASES in "
                f"vllm/kernels/helion/utils.py, or run "
                f"scripts/autotune_helion_kernels.py to generate configs "
                f"for your platform."
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

    def set_config(
        self, platform: str, config_key: str, config: "helion.Config"
    ) -> None:
        platform = platform.lower()
        if platform not in self._configs:
            self._configs[platform] = {}
        self._configs[platform][config_key] = config
        logger.debug(
            "Set config for kernel '%s': platform='%s', key='%s'",
            self._kernel_name,
            platform,
            config_key,
        )

    def has_config(self, platform: str, config_key: str) -> bool:
        platform = platform.lower()
        platform_dict = self._configs.get(platform)
        if platform_dict is None:
            return False
        return config_key in platform_dict


class ConfigManager:
    """File-level configuration management for Helion kernels (global singleton)."""

    _instance: "ConfigManager | None" = None
    _instance_base_dir: Path | None = None

    def __new__(cls, base_dir: str | Path | None = None) -> "ConfigManager":
        resolved_base_dir = cls._resolve_base_dir(base_dir)

        if cls._instance is not None:
            if cls._instance_base_dir != resolved_base_dir:
                raise ValueError(
                    f"ConfigManager singleton already exists with base_dir "
                    f"'{cls._instance_base_dir}', cannot create with different "
                    f"base_dir '{resolved_base_dir}'"
                )
            return cls._instance

        instance = super().__new__(cls)
        cls._instance = instance
        cls._instance_base_dir = resolved_base_dir
        return instance

    def __init__(self, base_dir: str | Path | None = None):
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

    def get_kernel_dir(self, kernel_name: str) -> Path:
        return self._base_dir / kernel_name

    def get_config_file_path(
        self, kernel_name: str, platform: str | None = None
    ) -> Path:
        if platform is not None:
            return self.get_kernel_dir(kernel_name) / f"{platform}.json"
        return self.get_kernel_dir(kernel_name)

    def ensure_base_dir_exists(self) -> Path:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        return self._base_dir

    def ensure_base_dir_writable(self) -> None:
        self.ensure_base_dir_exists()
        test_file = self._base_dir / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except OSError as e:
            raise OSError(
                f"Config directory '{self._base_dir}' is not writable: {e}"
            ) from e

    def _load_platform_file(self, kernel_name: str, platform: str) -> dict[str, Any]:
        config_path = self.get_config_file_path(kernel_name, platform)
        if not config_path.exists():
            return {}
        try:
            with open(config_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load config file %s: %s", config_path, e)
            return {}

    def load_config_set(self, kernel_name: str) -> ConfigSet:
        kernel_dir = self.get_kernel_dir(kernel_name)
        if not kernel_dir.is_dir():
            return ConfigSet.from_dict(kernel_name, {})

        data: dict[str, Any] = {}
        for platform_file in sorted(kernel_dir.glob("*.json")):
            platform = platform_file.stem
            try:
                with open(platform_file) as f:
                    platform_data = json.load(f)
                data[platform] = platform_data
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Failed to load config file %s: %s", platform_file, e)

        return ConfigSet.from_dict(kernel_name, data)

    def get_platform_configs(
        self, kernel_name: str, platform: str
    ) -> dict[str, helion.Config]:
        platform_data = self._load_platform_file(kernel_name, platform)
        if not platform_data:
            return {}
        config_set = ConfigSet.from_dict(kernel_name, {platform: platform_data})
        config_keys = config_set.get_config_keys(platform)
        return {
            config_key: config_set.get_config(platform, config_key)
            for config_key in config_keys
        }

    def save_config_set(self, config_set: ConfigSet) -> Path:
        kernel_dir = self.get_kernel_dir(config_set.kernel_name)
        kernel_dir.mkdir(parents=True, exist_ok=True)

        full_data = config_set.to_dict()
        for platform, platform_data in full_data.items():
            platform_path = kernel_dir / f"{platform}.json"
            with open(platform_path, "w") as f:
                json.dump(platform_data, f, indent=2)
            logger.info("Saved config to: %s", platform_path)

        return kernel_dir

    def save_configs(
        self,
        kernel_name: str,
        platform: str,
        configs: dict[str, "helion.Config"],
    ) -> Path:
        """Save configs for a kernel/platform, merging with existing."""
        platform_data = self._load_platform_file(kernel_name, platform)
        for config_key, config in configs.items():
            platform_data[config_key] = json.loads(config.to_json())

        platform_path = self.get_config_file_path(kernel_name, platform)
        platform_path.parent.mkdir(parents=True, exist_ok=True)
        with open(platform_path, "w") as f:
            json.dump(platform_data, f, indent=2)

        logger.info("Saved config to: %s", platform_path)
        return platform_path

    def config_exists(self, kernel_name: str, platform: str, config_key: str) -> bool:
        platform_data = self._load_platform_file(kernel_name, platform)
        return config_key in platform_data
