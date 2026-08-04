# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
# Authors:
#   Ruyi Zhang <ruyi.zhang@samsung.com>
#   Wenwen Chen <wenwen.chen@samsung.com>

# Standard
from pathlib import Path

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    extract_plugin_type,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)
PLUGIN_TYPE = "hf3fs"


class HF3fsConnectorAdapter(ConnectorAdapter):
    """Adapter for HF3fsconnector which access data by 3FS Usrbio interfaces."""

    def __init__(self) -> None:
        super().__init__("hf3fs://")

    def can_parse(self, url: str) -> bool:
        if url.startswith(self.schema):
            return True
        if url.startswith("plugin://"):
            pname = url[len("plugin://") :]
            return extract_plugin_type(pname) == PLUGIN_TYPE
        return False

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .hf3fs_connector import HF3fsConnector

        if context.config is None:
            raise ValueError("The config is not set.")
        if context.config.extra_config is None:
            raise ValueError("The extra config is not set.")

        # Get config from extra_config
        self.extra_config = context.config.extra_config

        self.base_paths_str = None
        # Legacy URL mode: extract base_paths_str from URL
        if context.plugin_name is None:
            self.base_paths_str = context.url[len(self.schema) :]
            logger.info(f"Extract base_path from URL: {context.url}")
        # Plugin mode: extract base_paths_str from extra_config
        else:
            key_prefix = context.plugin_name or PLUGIN_TYPE
            item = f"remote_storage_plugin.{key_prefix}.base_path"
            self.base_paths_str = self.extra_config.get(item)
            if self.base_paths_str is not None:
                logger.info(f"Extract base_path from extra_config by {item}")
            else:
                self.base_paths_str = self.extra_config.get("hf3fs_base_path")
                logger.info("Extract base_path from extra_config by hf3fs_base_path")
            if self.base_paths_str is None:
                raise ValueError(
                    "HF3fSConnector requires base_path via URL or extra_config"
                )

        self.hf3fs_mount_point = self.extra_config.get("hf3fs_mount_point", None)
        self.hf3fs_iov_size = self._get_positive_int(
            "hf3fs_iov_size", 209715200
        )  # default 200M
        self.hf3fs_ior_entries = self._get_positive_int("hf3fs_ior_entries", 256)
        self.hf3fs_io_depth = self._get_int("hf3fs_io_depth", 0)
        self.hf3fs_numa_id = self._get_int("hf3fs_numa_id", -1)
        self.hf3fs_io_thread_num = self._get_positive_int("hf3fs_io_thread_num", 4)

        logger.info(
            f"HF3fSConnector Config:\n"
            f"base_paths_str = {self.base_paths_str}\n"
            f"hf3fs_mount_point = {self.hf3fs_mount_point}\n"
            f"hf3fs_iov_size = {self.hf3fs_iov_size}\n"
            f"hf3fs_ior_entries = {self.hf3fs_ior_entries}\n"
            f"hf3fs_io_depth = {self.hf3fs_io_depth}\n"
            f"hf3fs_numa_id = {self.hf3fs_numa_id}\n"
            f"hf3fs_io_thread_num = {self.hf3fs_io_thread_num}\n"
        )

        # check config
        if not self._validate_config():
            logger.error("Invalid configuration of HF3fsConnector")
            raise ValueError("Invalid configuration of hf3fsConnector")

        logger.info(f"Creating HF3fsConnector by base_paths: {self.base_paths_str}")
        return HF3fsConnector(
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            mount_point=self.hf3fs_mount_point,
            iov_size=self.hf3fs_iov_size,
            ior_entries=self.hf3fs_ior_entries,
            io_depth=self.hf3fs_io_depth,
            numa_id=self.hf3fs_numa_id,
            io_thread_num=self.hf3fs_io_thread_num,
            plugin_name=context.plugin_name,
            base_paths_str=self.base_paths_str,
        )

    def _get_positive_int(self, key: str, default: int) -> int:
        """
        Extract a positive integer from config with validation.
        Args:
            key: Configuration key
            default: Default value if key not found
        Returns:
            Validated positive integer
        Raises:
            ValueError: If value is not a positive integer
        """
        value = self.extra_config.get(key, default)
        try:
            int_value = int(value)
            if int_value <= 0:
                logger.error(f"{key} must be positive, got {int_value}")
                raise ValueError(f"{key} must be positive, got {int_value}")
            return int_value
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid value for {key}: {value}, must be positive integer")
            raise ValueError(
                f"Invalid value for {key}: {value}, must be positive integer"
            ) from e

    def _get_int(self, key: str, default: int) -> int:
        """
        Extract a positive integer from config
        Args:
            key: Configuration key
            default: Default value if key not found
        Returns:
            Validated integer
        Raises:
            ValueError: If value is not an integer
        """
        value = self.extra_config.get(key, default)
        try:
            int_value = int(value)
            return int_value
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid value for {key}: {value}, must be integer"
            ) from e

    def _validate_config(self) -> bool:
        """
        Validate the configuration.
        Returns:
            True if valid, False otherwise
        """
        result = True
        if not 128 <= self.hf3fs_ior_entries <= 1024:
            logger.error("hf3fs_ior_entries must in range [128, 1024]")
            result = False

        if not (100 * 1024 * 1024) <= self.hf3fs_iov_size <= (2 * 1024 * 1024 * 1024):
            logger.error(
                f"hf3fs_iov_size must in range [{100 * 1024 * 1024}(100MB), "
                f"{2 * 1024 * 1024 * 1024}(2GB)]"
            )
            result = False

        if not -128 <= self.hf3fs_io_depth <= 128:
            logger.error("hf3fs_io_depth must in range [-128, 128]")
            result = False

        max_numa_id = self._get_numa_node_count() - 1
        if not -1 <= self.hf3fs_numa_id <= max_numa_id:
            logger.error(
                f"hf3fs_numa_id must in range [-1, {max_numa_id}] in current server"
            )
            result = False

        if not 2 <= self.hf3fs_io_thread_num <= 16:
            logger.error("hf3fs_io_thread_num must in range [2, 16]")
            result = False

        if not self._validate_paths(self.base_paths_str, self.hf3fs_mount_point):
            result = False
        return result

    def _validate_paths(self, base_paths_str, mount_point_path_str) -> bool:
        """parse the base_paths_str which separated by comma, and validate
        the paths are subdirectory of mount point path"""
        try:
            parent_path = Path(mount_point_path_str).resolve()
            if not parent_path.exists():
                logger.error(
                    f"Invalid mount point:{mount_point_path_str} which is not exist"
                )
                return False

            if not parent_path.is_dir():
                logger.error(
                    f"Invalid mount point:{mount_point_path_str} which is not dir"
                )
                return False
        except Exception as e:
            # invalidate path
            logger.error(f"Invalid mount point:{mount_point_path_str}, {e}")
            return False

        try:
            if "," in base_paths_str:
                # multiple paths
                paths = base_paths_str.split(",")
                self.base_paths = [Path(p.strip()) for p in paths]
            else:
                # single path
                self.base_paths = [Path(base_paths_str.strip())]
        except Exception as e:
            # invalidate path
            logger.error(f"Invalid base_paths_str:{base_paths_str}, {e}")
            return False

        for path in self.base_paths:
            try:
                # convert relative path to absolute
                resolved = path.resolve()
                if not resolved.is_relative_to(parent_path):
                    logger.error(
                        f"Invalid path:{str(path)}, is not subdirectory of "
                        f"mount point:{mount_point_path_str}"
                    )
                    return False
            except Exception as e:
                logger.error(f"Invalid path {str(path)}, {e}")
                return False
        return True

    def _get_numa_node_count(self) -> int:
        # Standard
        import glob
        import os
        import subprocess

        # method 1:  /sys/devices/system/node/
        node_paths = glob.glob("/sys/devices/system/node/node[0-9]*")
        count = len([p for p in node_paths if os.path.isdir(p)])
        if count > 0:
            return count

        # method 2: numactl command
        try:
            result = subprocess.run(
                ["numactl", "--hardware"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split("\n"):
                if line.startswith("available:"):
                    return int(line.split()[1])
        except (
            FileNotFoundError,
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
        ):
            pass

        # method 3: psutil
        try:
            # Third Party
            import psutil

            if hasattr(psutil, "sensors_numa"):
                numa_info = psutil.sensors_numa()
                return max(len(numa_info), 1)
        except ImportError:
            pass
        return 1
