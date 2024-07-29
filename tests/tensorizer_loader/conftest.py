# isort: skip_file

import contextlib
import gc

import pytest
import ray
import torch

from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig


@pytest.fixture(autouse=True)
def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()



@pytest.fixture(autouse=True)
def tensorizer_config():
    config = TensorizerConfig(tensorizer_uri="vllm")
    return config