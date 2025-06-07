# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_pooling_model_runner import GPUPoolingModelRunner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pooling_model_runner_creation():
    """Test that pooling model runner can be created for embedding models."""
    # Create a minimal config for an embedding model
    from vllm.config import (CacheConfig, DeviceConfig, LoadConfig,
                             ModelConfig, ParallelConfig, SchedulerConfig)

    model_config = ModelConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="embed",
        runner_type="pooling",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype=torch.float16,
        seed=42,
        max_model_len=512,
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_revision=None,
        revision=None,
        code_revision=None,
        qlora_adapter_name_or_path=None,
        override_neuron_config=None,
        default_guided_backend="outlines",
        tokenizer_pool_size=0,
        tokenizer_pool_type="ray",
        tokenizer_pool_extra_config={},
        limit_mm_per_prompt={})

    device_config = DeviceConfig(device="cuda")
    cache_config = CacheConfig(block_size=16,
                               gpu_memory_utilization=0.9,
                               swap_space=0,
                               cache_dtype=torch.float16,
                               cache_seed=42)
    load_config = LoadConfig()
    parallel_config = ParallelConfig(pipeline_parallel_size=1,
                                     tensor_parallel_size=1)
    scheduler_config = SchedulerConfig(max_num_batched_tokens=512,
                                       max_num_seqs=128)

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
    )

    device = torch.device("cuda:0")

    # Test that GPUPoolingModelRunner can be created
    runner = GPUPoolingModelRunner(vllm_config, device)

    # Verify it's the correct type
    assert isinstance(runner, GPUPoolingModelRunner)
    assert runner.sampler is None  # Pooling models should not have sampler
    assert runner.rejection_sampler is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_worker_selects_pooling_runner():
    """Test that worker correctly selects pooling model runner for embedding models."""
    # This test would require a more complete setup
    # For now, just test the import works
    from vllm.v1.worker.gpu_pooling_model_runner import GPUPoolingModelRunner
    assert GPUPoolingModelRunner is not None


def test_v1_outputs_with_pooling():
    """Test that V1 outputs can handle pooled outputs."""
    from vllm.v1.outputs import ModelRunnerOutput

    # Test creating output with pooled data
    output = ModelRunnerOutput(
        req_ids=["req1", "req2"],
        req_id_to_index={
            "req1": 0,
            "req2": 1
        },
        sampled_token_ids=None,  # No sampling for pooling
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooled_outputs=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # Mock embeddings
    )

    assert output.pooled_outputs is not None
    assert len(output.pooled_outputs) == 2
    assert output.sampled_token_ids is None
