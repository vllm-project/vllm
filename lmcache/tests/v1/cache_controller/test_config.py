# SPDX-License-Identifier: Apache-2.0
"""
Test script for LMCache Controller Configuration

This script tests the ControllerConfig class and its functionality:
- Loading from environment variables
- Loading from YAML file
- Command-line parameter overrides
- Global singleton pattern
"""

# Standard
import os
import tempfile

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.config import (
    ControllerConfig,
    controller_get_or_create_config,
    load_controller_config_with_overrides,
    override_controller_config_from_dict,
)

logger = init_logger(__name__)


def test_from_env():
    """Test loading configuration from environment variables"""
    logger.info("=" * 80)
    logger.info("Testing: Loading from environment variables")
    logger.info("=" * 80)

    # Set some environment variables
    os.environ["LMCACHE_CONTROLLER_CONTROLLER_PORT"] = "9001"
    os.environ["LMCACHE_CONTROLLER_HEALTH_CHECK_INTERVAL"] = "30"

    config = ControllerConfig.from_env()
    config.validate()
    config.log_config()

    # Verify values
    assert config.controller_port == 9001
    assert config.health_check_interval == 30

    logger.info("✓ Environment variable loading test passed")


def test_from_file():
    """Test loading configuration from YAML file"""
    logger.info("=" * 80)
    logger.info("Testing: Loading from YAML file")
    logger.info("=" * 80)

    # Create a temporary YAML config file
    config_content = """
controller_port: 9002
health_check_interval: 60
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_file = f.name

    try:
        config = ControllerConfig.from_file(config_file)
        config.validate()
        config.log_config()

        # Verify values
        assert config.controller_port == 9002
        assert config.health_check_interval == 60

        logger.info("✓ File loading test passed")
    finally:
        os.unlink(config_file)


def test_override_from_dict():
    """Test overriding configuration with dictionary"""
    logger.info("=" * 80)
    logger.info("Testing: Override configuration with dictionary")
    logger.info("=" * 80)

    config = ControllerConfig.from_env()

    # Override with dictionary
    overrides = {
        "controller_port": 9003,
        "health_check_interval": 45,
    }

    override_controller_config_from_dict(config, overrides)
    config.validate()
    config.log_config()

    # Verify overrides
    assert config.controller_port == 9003
    assert config.health_check_interval == 45

    logger.info("✓ Dictionary override test passed")


def test_singleton_pattern():
    """Test thread-safe singleton pattern"""
    logger.info("=" * 80)
    logger.info("Testing: Singleton pattern")
    logger.info("=" * 80)

    # Clear any existing instance using the new reset method
    controller_get_or_create_config.reset()

    # Set environment variable for config file
    os.environ["LMCACHE_CONTROLLER_CONTROLLER_PORT"] = "9004"

    # Get config instance multiple times
    config1 = controller_get_or_create_config()
    config2 = controller_get_or_create_config()

    # Verify they are the same instance
    assert config1 is config2
    assert config1.controller_port == 9004

    # Test reset functionality
    controller_get_or_create_config.reset()
    os.environ["LMCACHE_CONTROLLER_CONTROLLER_PORT"] = "9005"
    config3 = controller_get_or_create_config()
    assert config3.controller_port == 9005
    assert config3 is not config1  # Should be a new instance after reset

    logger.info("✓ Singleton pattern test passed")


def test_to_from_dict():
    """Test dictionary serialization/deserialization"""
    logger.info("=" * 80)
    logger.info("Testing: Dictionary serialization")
    logger.info("=" * 80)

    config = ControllerConfig.from_env()
    config.controller_port = 9005

    # Convert to dictionary
    config_dict = config.to_dict()

    # Create new config from dictionary
    new_config = ControllerConfig.from_dict(config_dict)

    # Verify they are equivalent
    assert new_config.controller_port == config.controller_port

    logger.info("✓ Dictionary serialization test passed")


def test_to_from_json():
    """Test JSON serialization/deserialization"""
    logger.info("=" * 80)
    logger.info("Testing: JSON serialization")
    logger.info("=" * 80)

    config = ControllerConfig.from_env()
    config.controller_port = 9006

    # Convert to JSON
    json_str = config.to_json()

    # Create new config from JSON
    new_config = ControllerConfig.from_json(json_str)

    # Verify they are equivalent
    assert new_config.controller_port == config.controller_port

    logger.info("✓ JSON serialization test passed")


def test_controller_monitor_ports_json_parsing():
    """Test that controller_monitor_ports JSON string is correctly parsed as dict"""
    logger.info("=" * 80)
    logger.info("Testing: controller_monitor_ports JSON parsing")
    logger.info("=" * 80)

    # Test 1: From environment variable (string JSON)
    original_env = os.environ.get("LMCACHE_CONTROLLER_CONTROLLER_MONITOR_PORTS")
    os.environ["LMCACHE_CONTROLLER_CONTROLLER_MONITOR_PORTS"] = (
        '{"pull": 8300, "reply": 8400}'
    )

    try:
        config = ControllerConfig.from_env()
        config.validate()

        # Check that controller_monitor_ports is a dict, not a string
        expected_type_msg = (
            f"Expected dict but got {type(config.controller_monitor_ports)}: "
            f"{config.controller_monitor_ports}"
        )
        assert isinstance(config.controller_monitor_ports, dict), expected_type_msg
        assert config.controller_monitor_ports["pull"] == 8300
        assert config.controller_monitor_ports["reply"] == 8400

        logger.info("✓ Environment variable JSON parsing test passed")

        # Test 2: From file with YAML string
        config_content = """
controller_monitor_ports: '{"pull": 8500, "reply": 8600}'
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_file = f.name

        try:
            config2 = ControllerConfig.from_file(config_file)
            config2.validate()

            expected_type_msg2 = (
                f"Expected dict but got {type(config2.controller_monitor_ports)}: "
                f"{config2.controller_monitor_ports}"
            )
            assert isinstance(config2.controller_monitor_ports, dict), (
                expected_type_msg2
            )
            assert config2.controller_monitor_ports["pull"] == 8500
            assert config2.controller_monitor_ports["reply"] == 8600

            logger.info("✓ File YAML JSON parsing test passed")

            # Test 3: Dictionary override with JSON string
            config3 = ControllerConfig.from_env()
            override_dict = {
                "controller_monitor_ports": '{"pull": 8700, "reply": 8800}'
            }
            override_controller_config_from_dict(config3, override_dict)

            expected_type_msg3 = (
                f"Expected dict but got {type(config3.controller_monitor_ports)}: "
                f"{config3.controller_monitor_ports}"
            )
            assert isinstance(config3.controller_monitor_ports, dict), (
                expected_type_msg3
            )
            assert config3.controller_monitor_ports["pull"] == 8700
            assert config3.controller_monitor_ports["reply"] == 8800

            logger.info("✓ Dictionary override JSON parsing test passed")

            # Test 4: Dictionary override with actual dict
            config4 = ControllerConfig.from_env()
            override_dict2 = {"controller_monitor_ports": {"pull": 8900, "reply": 9000}}
            override_controller_config_from_dict(config4, override_dict2)

            expected_type_msg4 = (
                f"Expected dict but got {type(config4.controller_monitor_ports)}: "
                f"{config4.controller_monitor_ports}"
            )
            assert isinstance(config4.controller_monitor_ports, dict), (
                expected_type_msg4
            )
            assert config4.controller_monitor_ports["pull"] == 8900
            assert config4.controller_monitor_ports["reply"] == 9000

            logger.info("✓ Dictionary override with dict test passed")

        finally:
            os.unlink(config_file)

    finally:
        # Restore original environment variable
        if original_env is not None:
            os.environ["LMCACHE_CONTROLLER_CONTROLLER_MONITOR_PORTS"] = original_env
        else:
            os.environ.pop("LMCACHE_CONTROLLER_CONTROLLER_MONITOR_PORTS", None)

    logger.info("✓ controller_monitor_ports JSON parsing test passed")


def test_load_controller_config_with_overrides():
    """
    Test loading controller config with overrides,
    especially for controller_monitor_ports
    """
    logger.info("=" * 80)
    logger.info("Testing: load_controller_config_with_overrides")
    logger.info("=" * 80)

    # Test with monitor ports as JSON string in overrides
    overrides = {
        "controller_host": "localhost",
        "controller_port": 8000,
        "controller_monitor_ports": '{"pull": 8300, "reply": 8400}',
    }

    # Load config without config file (from env with overrides)
    config = load_controller_config_with_overrides(overrides=overrides)

    # Verify values
    assert config.controller_host == "localhost"
    assert config.controller_port == 8000

    expected_type_msg = (
        f"Expected dict but got {type(config.controller_monitor_ports)}: "
        f"{config.controller_monitor_ports}"
    )
    assert isinstance(config.controller_monitor_ports, dict), expected_type_msg
    assert config.controller_monitor_ports["pull"] == 8300
    assert config.controller_monitor_ports["reply"] == 8400

    # Log config to see the output
    config.log_config()

    logger.info("✓ load_controller_config_with_overrides test passed")
