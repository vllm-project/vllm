# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the intermediate tensor logging functionality.
"""

import json
from os.path import isdir
import shutil
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn

from vllm.config import IntermediateLoggingConfig
from vllm.v1.intermediates.intermediates_logging import (get_current_il_config,
                                                  get_step, increment_step,
                                                  intermediate_logging,
                                                  register_intermediate_hooks,
                                                  reset_step,
                                                  should_log_device,
                                                  should_log_module,
                                                  should_log_step)


class SimpleModel(nn.Module):
    """A simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after the test
    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def il_config(temp_output_dir):
    """Create a basic IntermediateLoggingConfig for testing."""
    return IntermediateLoggingConfig(output_dir=temp_output_dir,
                                     enabled=True,
                                     log_step_ids=[0, 1],
                                     module_call_match=[".*linear.*"])


def test_step_counter():
    """Test the step counter functionality."""
    # Reset the step counter
    reset_step()
    assert get_step() == 0

    # Increment the step counter
    increment_step()
    assert get_step() == 1

    # Increment again
    increment_step()
    assert get_step() == 2

    # Reset again
    reset_step()
    assert get_step() == 0


def test_intermediate_logging_context_manager():
    """Test the intermediate_logging context manager."""
    # Create a config
    config = IntermediateLoggingConfig(enabled=True)

    # Initially, there should be no global config
    assert get_current_il_config() is None

    # Use the context manager
    with intermediate_logging(config):
        # Inside the context, the global config should be set
        assert get_current_il_config() is not None
        assert get_current_il_config().enabled is True

    # After the context, the global config should be None again
    assert get_current_il_config() is None

    # Test with a different config
    config2 = IntermediateLoggingConfig(enabled=False)
    with intermediate_logging(config2):
        assert get_current_il_config() is not None
        assert get_current_il_config().enabled is False


def test_should_log_step():
    """Test the should_log_step function."""
    # Reset step counter
    reset_step()

    # Create configs with different step IDs
    config_all_steps = IntermediateLoggingConfig(
        enabled=True,
        log_step_ids=[]  # Empty list means log all steps
    )
    config_specific_steps = IntermediateLoggingConfig(
        enabled=True,
        log_step_ids=[0, 2, 4]  # Only log steps 0, 2, and 4
    )
    config_disabled = IntermediateLoggingConfig(enabled=False,
                                                log_step_ids=[0, 1, 2])

    # Test with all steps config
    with intermediate_logging(config_all_steps):
        assert should_log_step(config_all_steps) is True  # Step 0
        increment_step()
        assert should_log_step(config_all_steps) is True  # Step 1

    # Reset step counter
    reset_step()

    # Test with specific steps config
    with intermediate_logging(config_specific_steps):
        assert should_log_step(config_specific_steps) is True  # Step 0
        increment_step()
        assert should_log_step(config_specific_steps) is False  # Step 1
        increment_step()
        assert should_log_step(config_specific_steps) is True  # Step 2
        increment_step()
        assert should_log_step(config_specific_steps) is False  # Step 3
        increment_step()
        assert should_log_step(config_specific_steps) is True  # Step 4

    # Test with disabled config
    with intermediate_logging(config_disabled):
        assert should_log_step(config_disabled) is False  # Disabled


def test_should_log_device():
    """Test the should_log_device function."""
    # Create configs with different device filters
    config_all_devices = IntermediateLoggingConfig(
        enabled=True,
        device_names=[]  # Empty list means log all devices
    )
    config_specific_devices = IntermediateLoggingConfig(
        enabled=True,
        device_names=["cuda:0", "cpu"]  # Only log cuda:0 and cpu
    )
    config_disabled = IntermediateLoggingConfig(enabled=False,
                                                device_names=["cuda:0", "cpu"])

    # Test with all devices config
    with intermediate_logging(config_all_devices):
        assert should_log_device(config_all_devices, "cuda:0") is True
        assert should_log_device(config_all_devices, "cuda:1") is True
        assert should_log_device(config_all_devices, "cpu") is True

    # Test with specific devices config
    with intermediate_logging(config_specific_devices):
        assert should_log_device(config_specific_devices, "cuda:0") is True
        assert should_log_device(config_specific_devices, "cuda:1") is False
        assert should_log_device(config_specific_devices, "cpu") is True

    # Test with disabled config
    with intermediate_logging(config_disabled):
        assert should_log_device(config_disabled, "cuda:0") is False
        assert should_log_device(config_disabled, "cpu") is False


def test_should_log_module(simple_model):
    """Test the should_log_module function."""
    # Create configs with different module name filters
    config_all_modules = IntermediateLoggingConfig(
        enabled=True,
        module_call_match=None  # None means log all modules
    )
    config_specific_modules = IntermediateLoggingConfig(
        enabled=True,
        module_call_match=[".*linear.*"
                           ]  # Only log modules with "linear" in the name
    )
    config_disabled = IntermediateLoggingConfig(enabled=False,
                                                module_call_match=[".*"])

    # Test with all modules config
    with intermediate_logging(config_all_modules):
        assert should_log_module(config_all_modules, "linear1",
                                 simple_model.linear1) is True
        assert should_log_module(config_all_modules, "relu",
                                 simple_model.relu) is True

    # Test with specific modules config
    with intermediate_logging(config_specific_modules):
        assert should_log_module(config_specific_modules, "linear1",
                                 simple_model.linear1) is True
        assert should_log_module(config_specific_modules, "relu",
                                 simple_model.relu) is False

    # Test with disabled config
    with intermediate_logging(config_disabled):
        assert should_log_module(config_disabled, "linear1",
                                 simple_model.linear1) is False
        assert should_log_module(config_disabled, "relu",
                                 simple_model.relu) is False


def test_register_hooks(simple_model, il_config):
    """Test registering hooks on a model."""
    # Register hooks
    logger_instance = register_intermediate_hooks(simple_model, il_config)

    # Check that hooks were registered
    assert len(logger_instance.hooks) > 0

    # Remove hooks
    logger_instance.remove_hooks()

    # Check that hooks were removed
    assert len(logger_instance.hooks) == 0


@mock.patch('vllm.v1.intermediates.intermediates_logging.dump_intermediates_to_json')
@mock.patch('vllm.v1.intermediates.intermediates_logging.save_tensors')
def test_forward_hooks(mock_save_tensors, mock_dump_json, simple_model,
                       il_config, temp_output_dir):
    """Test that forward hooks are called during model execution."""
    mock_save_tensors.return_value = None
    # Register hooks
    with intermediate_logging(il_config):
        logger_instance = register_intermediate_hooks(simple_model, il_config)

        # Create input tensor
        input_tensor = torch.randn(2, 10)

        # Reset step counter
        reset_step()

        # Forward pass
        simple_model(input_tensor)

        # Check that the step counter was incremented
        assert get_step() == 1

        # Check that dump_intermediates_to_json and save_tensors were called
        assert mock_dump_json.called
        assert mock_save_tensors.called
        

        # Remove hooks
        logger_instance.remove_hooks()


def test_end_to_end(simple_model, il_config, temp_output_dir):
    """Test the entire intermediate logging workflow end-to-end."""
    # Register hooks
    with intermediate_logging(il_config):
        logger_instance = register_intermediate_hooks(simple_model, il_config)

        # Create input tensor
        input_tensor = torch.randn(2, 10)

        # Reset step counter
        reset_step()

        # Forward pass
        simple_model(input_tensor)

        # Check that output directories were created
        root_dir = Path(il_config._output_run_dir)
        assert root_dir.exists()
        step_dir = root_dir / "step_0"
        assert step_dir.exists()

        module_dirs = list(step_dir.glob("*"))
        print(f"{module_dirs=}")
        assert len(module_dirs) > 0

        # Check that input and output files were created
        for module_dir in module_dirs:
            print(f"{module_dir=}")
            if os.path.isdir(module_dir):
                inputs_json = module_dir / "inputs.json"
                outputs_json = module_dir / "outputs.json"

                # Check that JSON files exist
                assert inputs_json.exists()
                assert outputs_json.exists()

                # Check that JSON files contain valid data
                with open(inputs_json) as f:
                    inputs_data = json.load(f)
                    assert "type" in inputs_data

                with open(outputs_json) as f:
                    outputs_data = json.load(f)
                    assert "type" in outputs_data

                # Check that tensor files exist
                tensor_files = list(module_dir.glob("*.pt"))
                assert len(tensor_files) > 0

        # Remove hooks
        logger_instance.remove_hooks()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
