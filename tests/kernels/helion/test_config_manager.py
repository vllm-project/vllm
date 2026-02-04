# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Helion ConfigManager and ConfigSet.

Tests the simplified configuration management system for Helion custom kernels.
"""

import json
import tempfile
from pathlib import Path

import pytest

from vllm.utils.import_utils import has_helion

# Skip entire module if helion is not available
if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

import helion

from vllm.kernels.helion.config_manager import (
    ConfigManager,
    ConfigSet,
)


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    """Reset ConfigManager singleton before each test."""
    ConfigManager.reset_instance()
    yield
    ConfigManager.reset_instance()


class TestConfigSet:
    """Test suite for ConfigSet class."""

    def test_config_set_creation(self):
        """Test creating an empty ConfigSet."""
        config_set = ConfigSet("test_kernel")

        assert config_set.kernel_name == "test_kernel"
        assert config_set.get_platforms() == []

    def test_config_set_from_dict(self):
        """Test creating ConfigSet from dictionary data."""
        # Use realistic config data that helion.Config can handle
        config_data = {
            "block_sizes": [32, 16],
            "num_warps": 4,
            "num_stages": 3,
            "pid_type": "persistent_interleaved",
        }
        data = {"h100": {"batch_32_hidden_4096": config_data}}

        config_set = ConfigSet.from_dict("test_kernel", data)

        assert config_set.kernel_name == "test_kernel"
        assert config_set.get_platforms() == ["h100"]

        # Verify the config was created correctly
        config = config_set.get_config("h100", "batch_32_hidden_4096")
        assert isinstance(config, helion.Config)
        assert config.block_sizes == [32, 16]
        assert config.num_warps == 4
        assert config.num_stages == 3
        assert config.pid_type == "persistent_interleaved"

    def test_config_set_get_config_keyerror(self):
        """Test that accessing non-existent configs raises informative KeyErrors."""
        config_set = ConfigSet("test_kernel")

        with pytest.raises(KeyError, match="platform 'h100' not found"):
            config_set.get_config("h100", "batch_32_hidden_4096")

        # Use realistic config data
        config_data = {"num_warps": 8, "num_stages": 4}
        data = {"h100": {"batch_64_hidden_2048": config_data}}
        config_set = ConfigSet.from_dict("test_kernel", data)

        with pytest.raises(
            KeyError, match="config_key 'batch_32_hidden_4096' not found"
        ):
            config_set.get_config("h100", "batch_32_hidden_4096")

    def test_config_set_get_platforms(self):
        """Test get_platforms method."""
        # Use realistic config data
        config1 = {"num_warps": 4, "num_stages": 3}
        config2 = {"num_warps": 8, "num_stages": 5}

        data = {
            "h100": {"batch_32_hidden_4096": config1},
            "a100": {"batch_16_hidden_2048": config2},
        }
        config_set = ConfigSet.from_dict("test_kernel", data)

        platforms = config_set.get_platforms()
        assert platforms == ["a100", "h100"]  # Should be sorted

    def test_config_set_get_config_keys(self):
        """Test get_config_keys method."""
        # Use realistic config data
        config1 = {"num_warps": 4, "num_stages": 3}
        config2 = {"num_warps": 8, "num_stages": 5}

        data = {
            "h100": {
                "batch_32_hidden_4096": config1,
                "batch_64_hidden_2048": config2,
            }
        }
        config_set = ConfigSet.from_dict("test_kernel", data)

        config_keys = config_set.get_config_keys("h100")
        assert config_keys == ["batch_32_hidden_4096", "batch_64_hidden_2048"]

        assert config_set.get_config_keys("v100") == []

    def test_config_set_to_dict(self):
        """Test converting ConfigSet to dictionary."""
        # Use realistic config data
        original_config = {
            "block_sizes": [64, 32],
            "num_warps": 16,
            "num_stages": 4,
            "pid_type": "persistent_blocked",
        }
        original_data = {"h100": {"batch_32_hidden_4096": original_config}}

        config_set = ConfigSet.from_dict("test_kernel", original_data)
        result_data = config_set.to_dict()

        # The result should match the original (Config roundtrip should work)
        assert result_data == original_data


class TestConfigManager:
    """Test suite for ConfigManager class."""

    def test_config_manager_creation_default_base_dir(self):
        """Test creating ConfigManager with default base directory."""
        manager = ConfigManager()
        assert manager._base_dir.name == "configs"

    def test_config_manager_creation_custom_base_dir(self):
        """Test creating ConfigManager with custom base directory."""
        custom_dir = "/tmp/custom_configs"
        manager = ConfigManager(base_dir=custom_dir)

        # Paths are resolved, so compare with resolved path
        assert manager._base_dir == Path(custom_dir).resolve()

    def test_get_config_file_path(self):
        """Test getting config file path for a kernel."""
        manager = ConfigManager(base_dir="/tmp")

        file_path = manager.get_config_file_path("silu_mul_fp8")

        expected_path = Path("/tmp/silu_mul_fp8.json")
        assert file_path == expected_path

    def test_ensure_base_dir_exists(self):
        """Test ensuring base directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "non_existent" / "configs"
            manager = ConfigManager(base_dir=base_dir)
            assert not base_dir.exists()

            returned_path = manager.ensure_base_dir_exists()

            assert base_dir.exists()
            assert base_dir.is_dir()
            assert returned_path == base_dir

    def test_load_config_set_file_not_exists(self):
        """Test loading config set when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(base_dir=temp_dir)
            config_set = manager.load_config_set("non_existent_kernel")

            assert isinstance(config_set, ConfigSet)
            assert config_set.kernel_name == "non_existent_kernel"
            assert config_set.get_platforms() == []

    def test_load_config_set_valid_file(self):
        """Test loading config set from valid file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use realistic config data
            kernel_config = {
                "block_sizes": [128, 64],
                "num_warps": 8,
                "num_stages": 6,
                "pid_type": "persistent_interleaved",
            }
            config_data = {"h100": {"batch_32_hidden_4096": kernel_config}}
            config_file = Path(temp_dir) / "test_kernel.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            manager = ConfigManager(base_dir=temp_dir)
            config_set = manager.load_config_set("test_kernel")

            assert isinstance(config_set, ConfigSet)
            assert config_set.kernel_name == "test_kernel"
            assert config_set.get_platforms() == ["h100"]

            # Verify the config was loaded correctly
            config = config_set.get_config("h100", "batch_32_hidden_4096")
            assert isinstance(config, helion.Config)
            assert config.block_sizes == [128, 64]
            assert config.num_warps == 8

    def test_load_config_set_invalid_json(self):
        """Test loading config set from file with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_kernel.json"
            with open(config_file, "w") as f:
                f.write("invalid json content {")

            manager = ConfigManager(base_dir=temp_dir)
            config_set = manager.load_config_set("test_kernel")

            assert isinstance(config_set, ConfigSet)
            assert config_set.kernel_name == "test_kernel"
            assert config_set.get_platforms() == []

    def test_save_config_set(self):
        """Test saving ConfigSet to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use realistic config data
            kernel_config = {
                "block_sizes": [256, 128],
                "num_warps": 16,
                "num_stages": 8,
                "pid_type": "persistent_blocked",
            }
            data = {"h100": {"batch_32_hidden_4096": kernel_config}}
            config_set = ConfigSet.from_dict("test_kernel", data)

            manager = ConfigManager(base_dir=temp_dir)
            saved_path = manager.save_config_set(config_set)

            expected_path = Path(temp_dir) / "test_kernel.json"
            assert saved_path == expected_path
            assert saved_path.exists()

            with open(saved_path) as f:
                loaded_data = json.load(f)
            assert loaded_data == data

    def test_save_config_set_creates_directory(self):
        """Test that save_config_set creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "nested" / "configs"
            config_set = ConfigSet("test_kernel")

            manager = ConfigManager(base_dir=nested_dir)
            saved_path = manager.save_config_set(config_set)

            assert nested_dir.exists()
            assert nested_dir.is_dir()
            assert saved_path.exists()

    def test_get_platform_configs(self):
        """Test getting all configs for a specific platform."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use realistic config data
            config_1 = {"num_warps": 4, "num_stages": 3, "block_sizes": [64, 32]}
            config_2 = {"num_warps": 8, "num_stages": 5, "block_sizes": [128, 64]}
            default_config = {
                "num_warps": 16,
                "num_stages": 7,
                "block_sizes": [256, 128],
            }
            config_3 = {"num_warps": 2, "num_stages": 2, "block_sizes": [32, 16]}

            config_data = {
                "h100": {
                    "batch_32_hidden_4096": config_1,
                    "batch_64_hidden_2048": config_2,
                    "default": default_config,
                },
                "a100": {"batch_16_hidden_1024": config_3},
            }
            config_file = Path(temp_dir) / "test_kernel.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            manager = ConfigManager(base_dir=temp_dir)

            h100_configs = manager.get_platform_configs("test_kernel", "h100")
            assert len(h100_configs) == 3
            assert "batch_32_hidden_4096" in h100_configs
            assert "batch_64_hidden_2048" in h100_configs
            assert "default" in h100_configs
            for config in h100_configs.values():
                assert isinstance(config, helion.Config)

            # Verify specific config details
            assert h100_configs["batch_32_hidden_4096"].num_warps == 4
            assert h100_configs["default"].num_stages == 7

            a100_configs = manager.get_platform_configs("test_kernel", "a100")
            assert len(a100_configs) == 1
            assert "batch_16_hidden_1024" in a100_configs
            assert isinstance(a100_configs["batch_16_hidden_1024"], helion.Config)
            assert a100_configs["batch_16_hidden_1024"].num_warps == 2

            nonexistent_configs = manager.get_platform_configs("test_kernel", "v100")
            assert len(nonexistent_configs) == 0

    def test_singleton_returns_same_instance(self):
        """Test that ConfigManager returns the same instance on repeated calls."""
        manager1 = ConfigManager(base_dir="/tmp/test_singleton")
        manager2 = ConfigManager(base_dir="/tmp/test_singleton")

        assert manager1 is manager2

    def test_singleton_with_default_base_dir(self):
        """Test singleton behavior with default base directory."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()

        assert manager1 is manager2
        assert manager1._base_dir == manager2._base_dir

    def test_singleton_error_on_different_base_dir(self):
        """Test that ConfigManager raises error when created with different base_dir."""
        ConfigManager(base_dir="/tmp/first_dir")

        with pytest.raises(ValueError, match="singleton already exists"):
            ConfigManager(base_dir="/tmp/different_dir")

    def test_reset_instance_allows_new_base_dir(self):
        """Test that reset_instance allows creating with a new base_dir."""
        manager1 = ConfigManager(base_dir="/tmp/first_dir")
        assert manager1._base_dir == Path("/tmp/first_dir").resolve()

        ConfigManager.reset_instance()

        manager2 = ConfigManager(base_dir="/tmp/second_dir")
        assert manager2._base_dir == Path("/tmp/second_dir").resolve()
        assert manager1 is not manager2

    def test_get_instance_returns_existing(self):
        """Test that get_instance returns the existing singleton."""
        manager1 = ConfigManager(base_dir="/tmp/test_get_instance")
        manager2 = ConfigManager.get_instance()

        assert manager1 is manager2

    def test_get_instance_raises_if_not_initialized(self):
        """Test that get_instance raises RuntimeError if no instance exists."""
        with pytest.raises(RuntimeError, match="has not been created"):
            ConfigManager.get_instance()
