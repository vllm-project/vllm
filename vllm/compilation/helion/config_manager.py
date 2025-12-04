# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration management for Helion kernels.

This module provides centralized configuration file management for Helion custom operations,
including naming conventions, directory resolution, and file I/O operations.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.compilation.helion.custom_op import HelionCustomOp

# Type alias for kernel identifier - can be string name or HelionCustomOp class type
KernelIdentifier = Union[str, "type[HelionCustomOp]"]

logger = init_logger(__name__)

# Import Helion types conditionally
try:
    import helion

    HELION_AVAILABLE = True
except ImportError:
    helion = None
    HELION_AVAILABLE = False


class ConfigManager:
    """
    Centralized configuration management for Helion kernels.

    This class handles all config-related operations including:
    - File naming conventions
    - Directory resolution and creation
    - Config loading and saving
    - Kernel name normalization

    File naming convention: helion_{kernel_name}_{config_key}.json

    Singleton pattern: Use ConfigManager.get_instance() to get the shared instance.
    """

    _instance: Optional["ConfigManager"] = None
    _initialized: bool = False

    def __new__(cls, base_dir: str | Path | None = None):
        """Singleton pattern: return existing instance or create new one."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_dir: str | Path | None = None):
        """
        Initialize ConfigManager (singleton pattern).

        Args:
            base_dir: Base directory for configs. If None, uses smart detection
                     to find vllm_repo_root/vllm/compilation/helion/configs

        Note: After first initialization, subsequent calls ignore base_dir parameter.
        """
        # Only initialize once (singleton pattern)
        if not ConfigManager._initialized:
            self.base_dir = self._resolve_base_dir(base_dir)
            ConfigManager._initialized = True
            logger.debug(
                f"ConfigManager singleton initialized with base_dir: {self.base_dir}"
            )

    @classmethod
    def get_instance(cls, base_dir: str | Path | None = None) -> "ConfigManager":
        """
        Get the singleton instance of ConfigManager.

        Args:
            base_dir: Base directory for configs (only used on first call)

        Returns:
            The singleton ConfigManager instance
        """
        if cls._instance is None:
            cls._instance = cls(base_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance (mainly for testing).

        Warning: This should only be used in tests or special circumstances.
        """
        cls._instance = None
        cls._initialized = False

    def _resolve_base_dir(self, base_dir: str | Path | None) -> Path:
        """
        Resolve the base directory for configs.

        Args:
            base_dir: User-provided base directory, or None for auto-detection

        Returns:
            Path to the config directory
        """
        if base_dir is not None:
            return Path(base_dir)

        # Smart detection: find vLLM repo root
        repo_root = self.find_vllm_repo_root()
        if repo_root:
            config_dir = repo_root / "vllm" / "compilation" / "helion" / "configs"
            return config_dir
        else:
            # Fallback to relative path
            logger.warning("Could not find vLLM repo root, using relative path")
            return Path("configs/helion")

    @staticmethod
    def find_vllm_repo_root() -> Path | None:
        """
        Find the vLLM repository root by looking for setup.py.

        Returns:
            Path to repo root, or None if not found
        """
        current = Path.cwd()

        # Search upwards for setup.py
        for parent in [current] + list(current.parents):
            if (parent / "setup.py").exists():
                # Additional validation: check for vllm directory
                if (parent / "vllm").is_dir():
                    return parent

        return None

    def _get_kernel_name(self, kernel_name_or_class: KernelIdentifier) -> str:
        """
        Get kernel name from string or class, using the exact registered name.

        Args:
            kernel_name_or_class: Either a kernel name string or a HelionCustomOp class

        Returns:
            Kernel name as registered in the CustomOp registry

        Examples:
            "silu_mul_fp8_helion" -> "silu_mul_fp8_helion"
            SiluMulFp8HelionClass -> "silu_mul_fp8_helion" (looks up registry name)
        """
        if isinstance(kernel_name_or_class, type):
            # If it's a HelionCustomOp class, try to get kernel_name from a temporary instance
            try:
                from vllm.compilation.helion.custom_op import HelionCustomOp

                if issubclass(kernel_name_or_class, HelionCustomOp):
                    # Create temporary instance to get kernel_name
                    temp_instance = kernel_name_or_class()
                    return temp_instance.kernel_name
            except Exception:
                pass

            # Look up the registered name in CustomOp registry
            from vllm.model_executor.custom_op import CustomOp

            for name, op_cls in CustomOp.op_registry.items():
                if op_cls == kernel_name_or_class:
                    return name
            # Fallback to class name if not found in registry
            return kernel_name_or_class.__name__
        else:
            return str(kernel_name_or_class)

    def get_config_filename(
        self, kernel_name: KernelIdentifier, config_key: str
    ) -> str:
        """
        Generate config filename following the standard convention.

        Args:
            kernel_name: Kernel name string or HelionCustomOp class
            config_key: Configuration key (e.g., "4096", "h4096_s8")

        Returns:
            Config filename: helion_{kernel_name}_{config_key}.json

        Examples:
            get_config_filename("silu_mul_fp8_helion", "4096")
            -> "helion_silu_mul_fp8_helion_4096.json"
        """
        kernel_name_str = self._get_kernel_name(kernel_name)
        return f"helion_{kernel_name_str}_{config_key}.json"

    def get_config_path(self, kernel_name: KernelIdentifier, config_key: str) -> Path:
        """
        Get full path to config file.

        Args:
            kernel_name: Kernel name string or HelionCustomOp class
            config_key: Configuration key

        Returns:
            Full path to config file
        """
        filename = self.get_config_filename(kernel_name, config_key)
        return self.base_dir / filename

    def config_exists(self, kernel_name: KernelIdentifier, config_key: str) -> bool:
        """
        Check if config file exists.

        Args:
            kernel_name: Kernel name string or HelionCustomOp class
            config_key: Configuration key

        Returns:
            True if config file exists
        """
        config_path = self.get_config_path(kernel_name, config_key)
        return config_path.exists()

    def load_config(
        self, kernel_name: KernelIdentifier, config_key: str
    ) -> Optional["helion.Config"]:
        """
        Load Helion config from file.

        Args:
            kernel_name: Kernel name string or HelionCustomOp class
            config_key: Configuration key

        Returns:
            Loaded Helion config, or None if file doesn't exist or Helion unavailable

        Note: If you need all configs for a kernel, use load_all_configs() instead
              as it's more efficient than multiple load_config() calls.
        """
        if not HELION_AVAILABLE:
            logger.warning("Helion not available, cannot load configs")
            return None

        config_path = self.get_config_path(kernel_name, config_key)

        if not config_path.exists():
            logger.debug(f"Config file not found: {config_path}")
            return None

        try:
            return helion.Config.load(str(config_path))
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return None

    def save_config(
        self, kernel_name: KernelIdentifier, config_key: str, config: "helion.Config"
    ) -> Path:
        """
        Save Helion config to file.

        Args:
            kernel_name: Kernel name string or HelionCustomOp class
            config_key: Configuration key
            config: Helion config to save

        Returns:
            Path where config was saved

        Raises:
            ImportError: If Helion is not available
        """
        if not HELION_AVAILABLE:
            raise ImportError("Helion not available, cannot save configs")

        config_path = self.get_config_path(kernel_name, config_key)

        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(str(config_path))
        logger.info(f"Saved config to: {config_path}")

        return config_path

    def load_all_configs(
        self, kernel_name: KernelIdentifier
    ) -> dict[str, "helion.Config"]:
        """
        Load all available configs for a kernel.

        This is more efficient than list_configs() + multiple load_config() calls
        since we typically need all configs anyway.

        Args:
            kernel_name: Kernel name string or HelionCustomOp class

        Returns:
            Dictionary mapping config keys to loaded Helion configs.
            Empty dict if no configs found or Helion unavailable.

        Example:
            configs = config_manager.load_all_configs("silu_mul_fp8_helion")
            # Returns: {"4096": Config(...), "8192": Config(...)}
        """
        if not HELION_AVAILABLE:
            logger.debug("Helion not available, cannot load configs")
            return {}

        kernel_name_str = self._get_kernel_name(kernel_name)
        configs: dict[str, helion.Config] = {}

        if not self.base_dir.exists():
            logger.debug(f"Config directory does not exist: {self.base_dir}")
            return {}

        # Find all config files for this kernel
        pattern = f"helion_{kernel_name_str}_*.json"
        config_files = list(self.base_dir.glob(pattern))

        if not config_files:
            logger.debug(
                f"No config files found for kernel '{kernel_name_str}' with pattern '{pattern}'"
            )
            return {}

        # Load all configs at once
        for config_file in config_files:
            try:
                # Parse config key from filename: helion_{kernel}_{config_key}.json
                # Remove the "helion_{kernel}_" prefix to get just the config_key
                filename_stem = config_file.stem
                prefix = f"helion_{kernel_name_str}_"

                if filename_stem.startswith(prefix):
                    config_key = filename_stem[len(prefix) :]

                    # Load the config
                    config = helion.Config.load(str(config_file))
                    if config is not None:
                        configs[config_key] = config
                        logger.debug(
                            f"Loaded config '{config_key}' for kernel '{kernel_name_str}'"
                        )

            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
                continue

        if configs:
            logger.debug(
                f"Loaded {len(configs)} configs for kernel '{kernel_name_str}': {list(configs.keys())}"
            )
        else:
            logger.debug(f"No configs could be loaded for kernel '{kernel_name_str}'")

        return configs

    def list_configs(
        self, kernel_name: KernelIdentifier | None = None
    ) -> dict[str, dict[str, Path]]:
        """
        List all available configs.

        Args:
            kernel_name: If provided, only list configs for this kernel name or class

        Returns:
            Dictionary mapping kernel names to their configs:
            {
                "silumulfp8": {"4096": Path(...), "8192": Path(...)},
                "rmsnormfp8": {"2048": Path(...)}
            }

        Note: Consider using load_all_configs() instead if you need the actual config objects,
              as it's more efficient than list_configs() + multiple load_config() calls.
        """
        if not self.base_dir.exists():
            return {}

        configs: dict[str, dict[str, Path]] = {}
        pattern = "helion_*.json"

        for config_file in self.base_dir.glob(pattern):
            try:
                # Parse filename: helion_{kernel}_{config_key}.json
                name_parts = config_file.stem.split("_")
                if len(name_parts) < 3 or name_parts[0] != "helion":
                    continue

                file_kernel_name = "_".join(name_parts[1:-1])
                config_key = name_parts[-1]

                # Filter by kernel if requested
                if kernel_name is not None:
                    target_kernel_name = self._get_kernel_name(kernel_name)
                    if file_kernel_name != target_kernel_name:
                        continue

                if file_kernel_name not in configs:
                    configs[file_kernel_name] = {}
                configs[file_kernel_name][config_key] = config_file

            except Exception as e:
                logger.warning(f"Failed to parse config filename {config_file}: {e}")

        return configs

    def get_base_dir(self) -> Path:
        """Get the base directory for configs."""
        return self.base_dir

    def ensure_base_dir_exists(self) -> Path:
        """Ensure base directory exists and return it."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        return self.base_dir
