# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion Kernel test utils"""

import pytest
import torch

from vllm.kernels.helion.config_manager import ConfigManager


def skip_if_platform_unsupported(op_name: str):
    try:
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        platform = get_canonical_gpu_name()

        try:
            config_manager = ConfigManager.get_instance()
        except RuntimeError:
            config_manager = ConfigManager()

        configs = config_manager.get_platform_configs(op_name, platform)
        if len(configs) == 0:
            pytest.skip(f"Current GPU platform not supported for {op_name} kernel")

    except (ImportError, RuntimeError, KeyError):
        pytest.skip(f"Error detecting platform support for {op_name} kernel")
