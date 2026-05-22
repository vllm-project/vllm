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

Platform files store config entries as a JSON array::

    [
        {"key": {}, "config": {...}},
        {"key": {"intermediate": 2048, "numtokens": 256}, "config": {...}},
        ...,
    ]

Config keys are ``CaseKey`` instances mapping parameter names to
values.  The default config uses ``CaseKey.default()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vllm.kernels.helion.case_key import CaseKey
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
    """In-memory collection of Helion configs with lookup/query capabilities.

    Configs are stored keyed by ``CaseKey``.  The default config
    uses ``CaseKey.default()`` as its key.
    """

    _ConfigDict = dict[str, dict[CaseKey, "helion.Config"]]

    def __init__(self, kernel_name: str):
        self._kernel_name = kernel_name
        self._configs: ConfigSet._ConfigDict = {}

    @property
    def kernel_name(self) -> str:
        return self._kernel_name

    def get_config(self, platform: str, config_key: CaseKey) -> helion.Config:
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
                f"config_key '{config_key}' not found for "
                f"platform '{platform}'. "
                f"Available config_keys: {avail_keys or '(none)'}"
            )

        return config

    def get_platforms(self) -> list[str]:
        return sorted(self._configs.keys())

    def get_config_keys(self, platform: str) -> list[CaseKey]:
        platform_dict = self._configs.get(platform.lower())
        if platform_dict is None:
            return []
        return sorted(platform_dict.keys(), key=str)

    def to_config_entries(self) -> dict[str, list[dict[str, Any]]]:
        """Serialize to config entries format for JSON output."""
        result: dict[str, list[dict[str, Any]]] = {}
        for platform, config_dict in self._configs.items():
            pairs: list[dict[str, Any]] = []
            for config_key, config in config_dict.items():
                config_data = json.loads(config.to_json())
                pairs.append({"key": dict(config_key), "config": config_data})
            result[platform] = pairs
        return result

    def to_dict(self) -> dict[str, dict[CaseKey, Any]]:
        """Return configs as a nested dict (platform -> key -> config)."""
        result: dict[str, dict[CaseKey, Any]] = {}
        for platform, config_dict in self._configs.items():
            result[platform] = {
                k: json.loads(v.to_json()) for k, v in config_dict.items()
            }
        return result

    @classmethod
    def from_dict(cls, kernel_name: str, data: dict[str, Any]) -> ConfigSet:
        config_set = cls(kernel_name)
        count = 0

        for platform, platform_data in data.items():
            if platform not in config_set._configs:
                config_set._configs[platform] = {}

            for entry in platform_data:
                raw_key = entry["key"]
                key = CaseKey.default() if not raw_key else CaseKey(raw_key)
                config = helion.Config(**entry["config"])
                config_set._configs[platform][key] = config
                count += 1

        if count > 0:
            logger.debug(
                "Loaded %d configs for kernel '%s'",
                count,
                kernel_name,
            )

        return config_set

    def set_config(
        self,
        platform: str,
        config_key: CaseKey,
        config: helion.Config,
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

    def has_config(self, platform: str, config_key: CaseKey) -> bool:
        platform = platform.lower()
        platform_dict = self._configs.get(platform)
        if platform_dict is None:
            return False
        return config_key in platform_dict


class ConfigManager:
    """File-level configuration management for Helion kernels (global singleton)."""

    _instance: ConfigManager | None = None
    _instance_base_dir: Path | None = None

    def __new__(cls, base_dir: str | Path | None = None) -> ConfigManager:
        resolved_base_dir = cls._resolve_base_dir(base_dir)

        if cls._instance is not None:
            if cls._instance_base_dir != resolved_base_dir:
                raise ValueError(
                    f"ConfigManager singleton already exists with base_dir "
                    f"'{cls._instance_base_dir}', cannot create with "
                    f"different base_dir '{resolved_base_dir}'"
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
    def get_instance(cls) -> ConfigManager:
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

    def _load_platform_file(self, kernel_name: str, platform: str) -> Any:
        config_path = self.get_config_file_path(kernel_name, platform)
        if not config_path.exists():
            return []
        try:
            with open(config_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load config file %s: %s", config_path, e)
            return []

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
                logger.error(
                    "Failed to load config file %s: %s",
                    platform_file,
                    e,
                )

        return ConfigSet.from_dict(kernel_name, data)

    def get_platform_configs(
        self, kernel_name: str, platform: str
    ) -> dict[CaseKey, helion.Config]:
        platform_data = self._load_platform_file(kernel_name, platform)
        if not platform_data:
            return {}
        config_set = ConfigSet.from_dict(kernel_name, {platform: platform_data})
        return {
            k: config_set.get_config(platform, k)
            for k in config_set.get_config_keys(platform)
        }

    def save_config_set(self, config_set: ConfigSet) -> Path:
        kernel_dir = self.get_kernel_dir(config_set.kernel_name)
        kernel_dir.mkdir(parents=True, exist_ok=True)

        full_data = config_set.to_config_entries()
        for platform, pairs in full_data.items():
            platform_path = kernel_dir / f"{platform}.json"
            with open(platform_path, "w") as f:
                json.dump(pairs, f, indent=2)
                f.write("\n")
            logger.info("Saved config to: %s", platform_path)

        return kernel_dir

    def save_configs(
        self,
        kernel_name: str,
        platform: str,
        configs: dict[CaseKey, helion.Config],
    ) -> Path:
        """Save configs for a kernel/platform, merging with existing."""
        config_set = ConfigSet.from_dict(
            kernel_name,
            {platform: self._load_platform_file(kernel_name, platform)},
        )
        for key, config in configs.items():
            config_set.set_config(platform, key, config)

        pairs = config_set.to_config_entries().get(platform, [])
        platform_path = self.get_config_file_path(kernel_name, platform)
        platform_path.parent.mkdir(parents=True, exist_ok=True)
        with open(platform_path, "w") as f:
            json.dump(pairs, f, indent=2)
            f.write("\n")

        logger.info("Saved config to: %s", platform_path)
        return platform_path

    def config_exists(
        self,
        kernel_name: str,
        platform: str,
        config_key: CaseKey,
    ) -> bool:
        platform_data = self._load_platform_file(kernel_name, platform)
        if not platform_data:
            return False
        target = dict(config_key)
        return any(entry["key"] == target for entry in platform_data)
