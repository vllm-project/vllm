# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures for tests."""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def configs_dir():
    """Get the configs directory."""
    return (
        Path(__file__).parent.parent / "src" / "vllm_rocm_autotuner_configs" / "configs"
    )


@pytest.fixture
def clean_env():
    """Clean and restore environment."""
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)


@pytest.fixture
def mock_model_dir(tmp_path):
    """Create a mock model directory."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()

    config = {
        "architectures": ["GptOssForCausalLM"],
        "num_hidden_layers": 36,
        "hidden_size": 2880,
        "num_attention_heads": 64,
    }

    (model_dir / "config.json").write_text(json.dumps(config, indent=2))
    return model_dir


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample config file for testing."""
    config = {
        "default_config": {
            "env_vars": {"TEST_VAR": "test"},
            "cli_args": {"test-arg": 123},
        },
        "model_configs": {
            "test/model": {
                "signature": "GptOssForCausalLM_36L_2880H_64A",
                "recipes": [
                    {
                        "name": "optimal",
                        "rank": 1,
                        "env_vars": {"VLLM_TEST": "1"},
                        "cli_args": {"block-size": 64},
                    }
                ],
            }
        },
    }

    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config, indent=2))
    return config_file
