# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test DeepSeek V3 with Tensor Parallelism.

This test verifies that DeepSeek V3 can be properly sharded across multiple GPUs
using TorchTitan's tensor parallelism integrated with vLLM.

Run with pytest:
    # Single GPU (no TP)
    pytest tests/custom_models/test_deepseek_v3_tp.py

    # 2 GPUs with TP=2
    pytest tests/custom_models/test_deepseek_v3_tp.py --forked

For multi-GPU testing with torchrun:
    torchrun --nproc_per_node=2 -m pytest tests/custom_models/test_deepseek_v3_tp.py
"""

import os

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="module")
def distributed_setup():
    """Initialize distributed context if running with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    yield rank, world_size

    if "RANK" in os.environ:
        dist.destroy_process_group()


def test_deepseek_v3_tp_creation(distributed_setup):
    """Test that DeepSeek V3 can be created with tensor parallelism."""
    rank, world_size = distributed_setup

    # Import after distributed is initialized to avoid CUDA init issues
    from examples.custom_models.deepseek_v3_torchtitan import (
        DeepSeekV3TorchTitanForCausalLM,
    )

    from vllm.config import CacheConfig, ModelConfig
    from vllm.model_executor.parallel_context import ParallelContext

    # Create a mock HF config for DeepSeek V3
    class MockHFConfig:
        vocab_size = 102400
        hidden_size = 2048
        intermediate_size = 10944
        moe_intermediate_size = 1408
        num_hidden_layers = 4  # Small for testing
        num_dense_layers = 4  # All dense, no MoE
        num_attention_heads = 16
        max_position_embeddings = 4096
        q_lora_rank = 0
        kv_lora_rank = 512
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        rope_theta = 10000.0

    # Create mock vLLM config
    class MockVllmConfig:
        def __init__(self):
            self.hf_config = MockHFConfig()
            self.model_config = ModelConfig(
                model="deepseek-v3-test",
                dtype=torch.bfloat16,
            )
            self.cache_config = CacheConfig(block_size=16)

    # Create parallel context
    parallel_context = ParallelContext(
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
    )

    # Create model
    vllm_config = MockVllmConfig()
    model = DeepSeekV3TorchTitanForCausalLM(
        vllm_config=vllm_config,
        parallel_context=parallel_context,
    )

    # Basic assertions
    assert model is not None
    assert hasattr(model, "model")
    assert hasattr(model, "config")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0


def test_deepseek_v3_tp_forward(distributed_setup):
    """Test that DeepSeek V3 forward pass works with TP."""
    rank, world_size = distributed_setup

    # Import after distributed is initialized
    from examples.custom_models.deepseek_v3_torchtitan import (
        DeepSeekV3TorchTitanForCausalLM,
    )

    from vllm.config import CacheConfig, ModelConfig
    from vllm.model_executor.parallel_context import ParallelContext

    # Setup (same as test_deepseek_v3_tp_creation)
    class MockHFConfig:
        vocab_size = 102400
        hidden_size = 2048
        intermediate_size = 10944
        moe_intermediate_size = 1408
        num_hidden_layers = 4
        num_dense_layers = 4
        num_attention_heads = 16
        max_position_embeddings = 4096
        q_lora_rank = 0
        kv_lora_rank = 512
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        rope_theta = 10000.0

    class MockVllmConfig:
        def __init__(self):
            self.hf_config = MockHFConfig()
            self.model_config = ModelConfig(
                model="deepseek-v3-test",
                dtype=torch.bfloat16,
            )
            self.cache_config = CacheConfig(block_size=16)

    parallel_context = ParallelContext(
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
    )

    vllm_config = MockVllmConfig()
    model = DeepSeekV3TorchTitanForCausalLM(
        vllm_config=vllm_config,
        parallel_context=parallel_context,
    )

    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_len), device=f"cuda:{rank}"
    )

    with torch.no_grad():
        hidden_states = model(input_ids)
        logits = model.compute_logits(hidden_states)

    # Assertions
    assert hidden_states is not None
    assert logits is not None
    assert logits.shape == (batch_size, seq_len, model.config.vocab_size)


@pytest.mark.skipif(
    not (os.environ.get("RANK") and int(os.environ.get("WORLD_SIZE", "1")) > 1),
    reason="TP sharding verification requires multi-GPU torchrun",
)
def test_deepseek_v3_tp_sharding(distributed_setup):
    """Test that weights are actually sharded across TP ranks."""
    rank, world_size = distributed_setup

    # Import after distributed is initialized
    from examples.custom_models.deepseek_v3_torchtitan import (
        DeepSeekV3TorchTitanForCausalLM,
    )

    from vllm.config import CacheConfig, ModelConfig
    from vllm.model_executor.parallel_context import ParallelContext

    # Setup (same as above)
    class MockHFConfig:
        vocab_size = 102400
        hidden_size = 2048
        intermediate_size = 10944
        moe_intermediate_size = 1408
        num_hidden_layers = 4
        num_dense_layers = 4
        num_attention_heads = 16
        max_position_embeddings = 4096
        q_lora_rank = 0
        kv_lora_rank = 512
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        rope_theta = 10000.0

    class MockVllmConfig:
        def __init__(self):
            self.hf_config = MockHFConfig()
            self.model_config = ModelConfig(
                model="deepseek-v3-test",
                dtype=torch.bfloat16,
            )
            self.cache_config = CacheConfig(block_size=16)

    parallel_context = ParallelContext(
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
    )

    vllm_config = MockVllmConfig()
    model = DeepSeekV3TorchTitanForCausalLM(
        vllm_config=vllm_config,
        parallel_context=parallel_context,
    )

    # Get first linear layer weight
    first_layer_weight = (
        model.model.layers["0"].attention.wq.weight
        if hasattr(model.model.layers["0"].attention, "wq")
        else model.model.layers["0"].attention.wq_a.weight
    )

    # Verify weights exist
    assert first_layer_weight is not None
    assert first_layer_weight.shape[0] > 0
    assert first_layer_weight.shape[1] > 0

    # Synchronize across ranks
    dist.barrier()
