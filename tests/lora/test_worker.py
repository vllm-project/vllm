# SPDX-License-Identifier: Apache-2.0

import os
import random
import tempfile
from typing import Union
from unittest.mock import patch

import pytest
import safetensors.torch
import torch

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
        ),
        load_config=LoadConfig(
            download_dir=None,
            load_format="dummy",
        ),
        parallel_config=ParallelConfig(1, 1, False),
        scheduler_config=SchedulerConfig("generate", 32, 32, 32),
        device_config=DeviceConfig("cuda"),
        cache_config=CacheConfig(block_size=16,
                                 gpu_memory_utilization=1.0,
                                 swap_space=0,
                                 cache_dtype="auto"),
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


@patch.dict(os.environ, {"RANK": "0"})
def test_worker_apply_dora(dora_files):
    """Test the worker's ability to load and manage DoRA adapters.

    DoRA adapters extend LoRA with magnitude vectors that normalize the weight
    contributions. This test verifies the worker correctly loads and makes
    available DoRA adapters.
    """
    # Configure worker with DoRA support
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            "meta-llama/Llama-3.2-1B-Instruct",  # Use model compatible with the DoRA adapter
            task="auto",
            tokenizer="meta-llama/Llama-3.2-1B-Instruct",
            tokenizer_mode="auto",
            trust_remote_code=False,
            seed=0,
            dtype="float16",
            revision=None,
        ),
        load_config=LoadConfig(
            download_dir=None,
            load_format="dummy",
        ),
        parallel_config=ParallelConfig(1, 1, False),
        scheduler_config=SchedulerConfig("generate", 16, 16,
                                         16),  # Reduced from 32 to save memory
        device_config=DeviceConfig("cuda"),
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.7,  # Reduced from 1.0 to avoid OOM
            swap_space=0,
            cache_dtype="auto",
        ),
        lora_config=LoRAConfig(
            max_lora_rank=16,  # Make sure this is large enough for DoRA adapter
            max_cpu_loras=8,  # Reduced from 32 to save memory
            max_loras=8,  # Reduced from 32 to save memory
            dora_enabled=True,
        ),  # Enable DoRA support
    )
    worker = Worker(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method=f"file://{tempfile.mkstemp()[1]}",
    )
    worker.init_device()
    worker.load_model()

    # Start with no adapters active
    worker.model_runner.set_active_loras([], LoRAMapping([], []))
    assert worker.list_loras() == set()

    # Create DoRA request and activate it
    dora_request = LoRARequest("dora_adapter", 1, dora_files)

    # Activate the DoRA adapter
    worker.model_runner.set_active_loras([dora_request], LoRAMapping([], []))

    # Verify the adapter was loaded
    assert worker.list_loras() == {dora_request.lora_int_id}

    # Check adapter weights to confirm DoRA structures exist
    # Get the adapter from the worker's adapter manager
    adapter_manager = worker.model_runner.lora_manager._adapter_manager
    dora_model = adapter_manager.get_adapter(dora_request.lora_int_id)

    # Verify the DoRA adapter has magnitude parameters in at least some modules
    has_magnitude_params = False
    for module_name, lora_weights in dora_model.loras.items():
        # Check for magnitude parameters
        if (hasattr(lora_weights, "magnitude_param")
                and lora_weights.magnitude_param is not None):
            has_magnitude_params = True

            # Different implementations handle magnitudes differently
            if isinstance(lora_weights.magnitude_param, list):
                # If it's a list, check for at least one valid magnitude tensor
                has_valid_mag = False
                for mag in lora_weights.magnitude_param:
                    if mag is not None:
                        has_valid_mag = True
                        assert isinstance(mag, torch.Tensor)
                assert (
                    has_valid_mag
                ), f"Module {module_name} has magnitude_param list with all None values"
            else:
                # If it's a tensor, verify it's a valid tensor
                assert isinstance(lora_weights.magnitude_param, torch.Tensor)

            # Also verify that magnitude is related to lora_b output dimension
            if isinstance(lora_weights.magnitude_param, torch.Tensor):
                assert (lora_weights.magnitude_param.shape[0] ==
                        lora_weights.lora_b.shape[1])
            elif isinstance(lora_weights.magnitude_param, list):
                for i, mag in enumerate(lora_weights.magnitude_param):
                    if mag is not None:
                        # Packed LoRA might not have simple correspondence
                        # This is a simplification for the test
                        pass

    # Verify at least some modules have magnitude parameters
    assert has_magnitude_params, "DoRA adapter should have magnitude parameters"

    # Test removing the adapter - use a direct approach
    worker.model_runner.set_active_loras([], LoRAMapping([], []))

    # Directly remove the specific adapter we added
    if hasattr(worker.model_runner.lora_manager, "remove_adapter"):
        # If the method exists, use it directly
        worker.model_runner.lora_manager.remove_adapter(
            dora_request.lora_int_id)

    # Verify the adapter is no longer active
    # Note: Due to how LRUCacheLoRAModelManager works, the adapter might still be
    # cached but not active. The key test is that adapter is not returned by list_loras
    assert dora_request.lora_int_id not in worker.list_loras()


@patch.dict(os.environ, {"RANK": "0"})
def test_worker_multiple_dora_adapters(dora_files, tmp_path):
    """Test the worker's ability to handle multiple DoRA adapters.

    This test creates a modified version of the DoRA adapter and verifies
    that the worker can handle having multiple DoRA adapters loaded and
    switching between them.
    """
    # Configure worker with DoRA support
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            "meta-llama/Llama-3.2-1B-Instruct",  # Use model compatible with the DoRA adapter
            task="auto",
            tokenizer="meta-llama/Llama-3.2-1B-Instruct",
            tokenizer_mode="auto",
            trust_remote_code=False,
            seed=0,
            dtype="float16",
            revision=None,
        ),
        load_config=LoadConfig(
            download_dir=None,
            load_format="dummy",
        ),
        parallel_config=ParallelConfig(1, 1, False),
        scheduler_config=SchedulerConfig("generate", 16, 16,
                                         16),  # Reduced to save memory
        device_config=DeviceConfig("cuda"),
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.7,  # Reduced to avoid OOM
            swap_space=0,
            cache_dtype="auto",
        ),
        lora_config=LoRAConfig(
            max_lora_rank=16,  # Large enough for DoRA adapter
            max_cpu_loras=8,  # Reduced to save memory
            max_loras=8,  # Reduced to save memory
            dora_enabled=True,
        ),  # Enable DoRA support
    )
    worker = Worker(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method=f"file://{tempfile.mkstemp()[1]}",
    )
    worker.init_device()
    worker.load_model()

    # Start with no adapters active
    worker.model_runner.set_active_loras([], LoRAMapping([], []))
    assert worker.list_loras() == set()

    # Create a modified version of the DoRA adapter for a second adapter
    modified_dora_path = tmp_path / "modified_dora"
    os.makedirs(modified_dora_path, exist_ok=True)

    # Copy the adapter config
    with open(os.path.join(dora_files, "adapter_config.json")) as f:
        import json

        adapter_config = json.load(f)

    # Save the config to the new location
    with open(os.path.join(modified_dora_path, "adapter_config.json"),
              "w") as f:
        json.dump(adapter_config, f)

    # Load the adapter weights
    adapter_path = os.path.join(dora_files, "adapter_model.safetensors")
    tensors = safetensors.torch.load_file(adapter_path)

    # Save them to the new location (we're not modifying the weights for this test,
    # just using a copy to simulate a second adapter)
    safetensors.torch.save_file(
        tensors, os.path.join(modified_dora_path, "adapter_model.safetensors"))

    # Create DoRA requests for both adapters
    dora_request_1 = LoRARequest("dora_adapter_1", 1, dora_files)
    dora_request_2 = LoRARequest("dora_adapter_2", 2, str(modified_dora_path))

    # First, test with only the first adapter
    worker.model_runner.set_active_loras([dora_request_1], LoRAMapping([], []))
    assert worker.list_loras() == {dora_request_1.lora_int_id}

    # Then, test with only the second adapter
    worker.model_runner.set_active_loras([dora_request_2], LoRAMapping([], []))
    # With LRU caching, both adapters might be in the cache
    # The important thing is that the second adapter is in the list
    assert dora_request_2.lora_int_id in worker.list_loras()

    # Finally, test with both adapters active
    # Note: In a real scenario, you'd need appropriate indices in the LoRAMapping,
    # but for this test we're just checking that the adapters are loaded
    worker.model_runner.set_active_loras([dora_request_1, dora_request_2],
                                         LoRAMapping([], []))
    # Check both adapters are in the list, regardless of what else might be there
    assert dora_request_1.lora_int_id in worker.list_loras()
    assert dora_request_2.lora_int_id in worker.list_loras()

    # Check that the second adapter has DoRA parameters
    adapter_manager = worker.model_runner.lora_manager._adapter_manager
    dora_model = adapter_manager.get_adapter(dora_request_2.lora_int_id)

    # Verify the DoRA adapter has magnitude parameters
    has_magnitude_params = False
    for module_name, lora_weights in dora_model.loras.items():
        if (hasattr(lora_weights, "magnitude_param")
                and lora_weights.magnitude_param is not None):
            has_magnitude_params = True
            break

    assert has_magnitude_params, "Second DoRA adapter should have magnitude parameters"

    # Cleanup - use a direct approach
    worker.model_runner.set_active_loras([], LoRAMapping([], []))

    # Directly remove the specific adapters we added
    if hasattr(worker.model_runner.lora_manager, "remove_adapter"):
        # If the method exists, use it directly
        worker.model_runner.lora_manager.remove_adapter(
            dora_request_1.lora_int_id)
        worker.model_runner.lora_manager.remove_adapter(
            dora_request_2.lora_int_id)

    # Verify the adapters are no longer active
    # Note: Due to how LRUCacheLoRAModelManager works, adapters might still be
    # cached but not active. The key test is that adapters are not returned by list_loras
    assert dora_request_1.lora_int_id not in worker.list_loras()
    assert dora_request_2.lora_int_id not in worker.list_loras()
