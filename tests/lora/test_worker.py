# SPDX-License-Identifier: Apache-2.0

import os
import random
import tempfile
from typing import Union
from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig)
from vllm.lora.models import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.v1.worker.gpu_worker import Worker as V1Worker
from vllm.worker.worker import Worker


@pytest.fixture(autouse=True)
def v1(run_with_both_engines_lora):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


@patch.dict(os.environ, {"RANK": "0"})
def test_worker_apply_lora(sql_lora_files):

    def set_active_loras(worker: Union[Worker, V1Worker],
                         lora_requests: list[LoRARequest]):
        lora_mapping = LoRAMapping([], [])
        if isinstance(worker, Worker):
            # v0 case
            worker.model_runner.set_active_loras(lora_requests, lora_mapping)
        else:
            # v1 case
            worker.model_runner.lora_manager.set_active_adapters(
                lora_requests, lora_mapping)

    worker_cls = V1Worker if envs.VLLM_USE_V1 else Worker

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            "meta-llama/Llama-2-7b-hf",
            task="auto",
            tokenizer="meta-llama/Llama-2-7b-hf",
            tokenizer_mode="auto",
            trust_remote_code=False,
            seed=0,
            dtype="float16",
            revision=None,
            enforce_eager=True,
        ),
        load_config=LoadConfig(
            download_dir=None,
            load_format="dummy",
        ),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
        ),
        scheduler_config=SchedulerConfig("generate", 32, 32, 32),
        device_config=DeviceConfig("cuda"),
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=1.0,
            swap_space=0,
            cache_dtype="auto",
        ),
        lora_config=LoRAConfig(max_lora_rank=8, max_cpu_loras=32,
                               max_loras=32),
    )
    worker = worker_cls(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method=f"file://{tempfile.mkstemp()[1]}",
    )

    worker.init_device()
    worker.load_model()

    set_active_loras(worker, [])
    assert worker.list_loras() == set()

    n_loras = 32
    lora_requests = [
        LoRARequest(str(i + 1), i + 1, sql_lora_files) for i in range(n_loras)
    ]

    set_active_loras(worker, lora_requests)
    assert worker.list_loras() == {
        lora_request.lora_int_id
        for lora_request in lora_requests
    }

    for i in range(32):
        random.seed(i)
        iter_lora_requests = random.choices(lora_requests,
                                            k=random.randint(1, n_loras))
        random.shuffle(iter_lora_requests)
        iter_lora_requests = iter_lora_requests[:-random.randint(0, n_loras)]
        set_active_loras(worker, lora_requests)
        assert worker.list_loras().issuperset(
            {lora_request.lora_int_id
             for lora_request in iter_lora_requests})
