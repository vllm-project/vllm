import pytest
import torch

from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.config import (ModelConfig, SchedulerConfig, CacheConfig,
                         ParallelConfig, VllmConfig, DeviceConfig)

DEVICE = torch.device("cpu")

def get_vllm_config():
    model_config = ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
    )
    device_config = DeviceConfig(
        device="cpu",
    )
    cache_config = CacheConfig()
    scheduler_config = SchedulerConfig()
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        device_config=device_config,
    )
    return vllm_config

@pytest.fixture
def model_runner():
    return CPUModelRunner(get_vllm_config(), DEVICE)

def test_execute_model(model_runner: CPUModelRunner):
    with pytest.raises(NotImplementedError):
        model_runner.execute_model(None)