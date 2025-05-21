# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    Tensorizer only tested on V0 so far.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')


@pytest.fixture(autouse=True)
def cleanup():
    cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture(autouse=True)
def tensorizer_config():
    config = TensorizerConfig(tensorizer_uri="vllm")
    return config
