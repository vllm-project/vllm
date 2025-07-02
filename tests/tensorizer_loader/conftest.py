# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig


@pytest.fixture(autouse=True)
def cleanup():
    cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture(autouse=True)
def tensorizer_config():
    config = TensorizerConfig(tensorizer_uri="vllm")
    return config
