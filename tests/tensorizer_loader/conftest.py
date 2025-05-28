# SPDX-License-Identifier: Apache-2.0
from typing import Callable

import pytest
import os

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig


@pytest.fixture(autouse=True)
def allow_insecure_serialization():
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

@pytest.fixture(autouse=True)
def cleanup():
    cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture(autouse=True)
def tensorizer_config():
    config = TensorizerConfig(tensorizer_uri="vllm")
    return config


def assert_from_collective_rpc(engine: LLM,
                               closure: Callable,
                               closure_kwargs: dict):
    res = engine.collective_rpc(method=closure, kwargs=closure_kwargs)
    return all(res)
