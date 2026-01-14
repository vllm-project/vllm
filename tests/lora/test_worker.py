# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random
import tempfile
from unittest.mock import patch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.load import LoadConfig
from vllm.config.lora import LoRAConfig
from vllm.lora.model_manager import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.v1.worker.gpu_worker import Worker

MODEL_PATH = "Qwen/Qwen3-0.6B"
NUM_LORAS = 16


@patch.dict(os.environ, {"RANK": "0"})
def test_worker_apply_lora(qwen3_lora_files):
    def set_active_loras(worker: Worker, lora_requests: list[LoRARequest]):
        lora_mapping = LoRAMapping([], [])

        worker.model_runner.lora_manager.set_active_adapters(
            lora_requests, lora_mapping
        )

    model_config = ModelConfig(
        MODEL_PATH,
        seed=0,
        dtype="float16",
        max_model_len=127,
        enforce_eager=True,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(
            download_dir=None,
            load_format="dummy",
        ),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
            runner_type="generate",
            max_num_batched_tokens=32,
            max_num_seqs=32,
            max_num_partial_prefills=32,
        ),
        device_config=DeviceConfig(current_platform.device_name),
        cache_config=CacheConfig(
            block_size=16,
            swap_space=0,
            cache_dtype="auto",
        ),
        lora_config=LoRAConfig(
            max_lora_rank=8, max_cpu_loras=NUM_LORAS, max_loras=NUM_LORAS
        ),
    )
    worker = Worker(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method=f"file://{tempfile.mkstemp()[1]}",
    )

    worker.init_device()
    worker.load_model()

    set_active_loras(worker, [])
    assert worker.list_loras() == set()

    lora_requests = [
        LoRARequest(str(i + 1), i + 1, qwen3_lora_files) for i in range(NUM_LORAS)
    ]

    set_active_loras(worker, lora_requests)
    assert worker.list_loras() == {
        lora_request.lora_int_id for lora_request in lora_requests
    }

    for i in range(NUM_LORAS):
        random.seed(i)
        iter_lora_requests = random.choices(
            lora_requests, k=random.randint(1, NUM_LORAS)
        )
        random.shuffle(iter_lora_requests)
        iter_lora_requests = iter_lora_requests[: -random.randint(0, NUM_LORAS)]
        set_active_loras(worker, lora_requests)
        assert worker.list_loras().issuperset(
            {lora_request.lora_int_id for lora_request in iter_lora_requests}
        )
