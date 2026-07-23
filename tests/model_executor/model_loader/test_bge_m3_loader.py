# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from unittest.mock import patch

import pytest

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@pytest.fixture
def loader():
    """Initialize a lightweight DefaultModelLoader for unit testing."""
    load_config = LoadConfig(load_format="hf")
    return DefaultModelLoader(load_config)


def test_optional_weight_loading(loader):
    """
    Verify if the loader handles missing weights correctly.

    Old version (main): Raises RuntimeError when weights are missing.
    New version (fix): Supports 'optional' parameter to prevent crash.
    """
    model_path = "fake_model_path"
    patterns = ["non_existent_file.pt"]

    # Check if the 'optional' parameter is supported in this version
    sig = inspect.signature(loader._prepare_weights)
    has_optional = "optional" in sig.parameters

    # Mock environment to simulate missing files
    with (
        patch(
            "vllm.model_executor.model_loader.default_loader.os.path.isdir",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.model_loader.default_loader.glob.glob", return_value=[]
        ),
        patch(
            "vllm.model_executor.model_loader.default_loader.download_weights_from_hf"
        ),
    ):
        if not has_optional:
            # Case 1: Old version (main)
            # It should raise RuntimeError because 'optional' is not supported.
            with pytest.raises(RuntimeError, match="Cannot find any model weights"):
                loader._prepare_weights(
                    model_name_or_path=model_path,
                    subfolder=None,
                    revision=None,
                    fall_back_to_pt=True,
                    allow_patterns_overrides=patterns,
                )
        else:
            # Case 2: New version (fix)
            # It should NOT raise error when 'optional=True' is passed.
            # hf_folder, hf_weights_files, use_safetensors
            _, hf_weights_files, _ = loader._prepare_weights(
                model_name_or_path=model_path,
                subfolder=None,
                revision=None,
                fall_back_to_pt=True,
                allow_patterns_overrides=patterns,
                optional=True,  # This is the new parameter
            )
            # Success: empty list returned instead of crashing
            assert hf_weights_files == []


def test_safetensors_detection_robustness(loader):
    """
    Verify if the loader correctly detects safetensors format
    for custom patterns like 'sparse_linear.safetensors'.
    """
    model_path = "fake_model_path"
    # This pattern is used in BGE-M3 model
    patterns = ["sparse_linear.safetensors"]

    # Mock glob to 'find' the file
    with (
        patch(
            "vllm.model_executor.model_loader.default_loader.os.path.isdir",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.model_loader.default_loader.glob.glob",
            return_value=["sparse_linear.safetensors"],
        ),
    ):
        # Prepare arguments dynamically for compatibility
        sig = inspect.signature(loader._prepare_weights)
        kwargs = {
            "model_name_or_path": model_path,
            "subfolder": None,
            "revision": None,
            "fall_back_to_pt": True,
            "allow_patterns_overrides": patterns,
        }
        if "optional" in sig.parameters:
            kwargs["optional"] = True

        _, _, use_safetensors = loader._prepare_weights(**kwargs)

        # In the fixed version, this must be True
        # because the pattern ends with .safetensors.
        # In the old version, this was False
        # because it only matched exactly "*.safetensors".
        assert use_safetensors is True, (
            "Safetensors should be detected by file extension"
        )
