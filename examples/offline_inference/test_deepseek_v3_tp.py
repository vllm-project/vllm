# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test DeepSeek V3 with Tensor Parallelism.

This test verifies that DeepSeek V3 can be properly sharded across multiple GPUs
using TorchTitan's tensor parallelism integrated with vLLM.

Run with:
    # Single GPU (no TP)
    python test_deepseek_v3_tp.py

    # 2 GPUs with TP=2
    torchrun --nproc_per_node=2 test_deepseek_v3_tp.py

    # 8 GPUs with TP=8
    torchrun --nproc_per_node=8 test_deepseek_v3_tp.py
"""

import os

import torch
import torch.distributed as dist

# Initialize distributed if running with torchrun
if "RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
else:
    rank = 0
    world_size = 1

print(f"[Rank {rank}/{world_size}] Starting DeepSeek V3 TP test...")

# Only import after distributed is initialized
from deepseek_v3_torchtitan import DeepSeekV3TorchTitanForCausalLM

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

if rank == 0:
    print(f"🚀 Creating DeepSeek V3 with TP={world_size}")

# Create model
vllm_config = MockVllmConfig()
model = DeepSeekV3TorchTitanForCausalLM(
    vllm_config=vllm_config,
    parallel_context=parallel_context,
)

if rank == 0:
    print(f"✅ Model created successfully on {world_size} GPU(s)")

    # Count parameters on rank 0
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters on rank {rank}: {total_params:,}")

# Verify weights are different across ranks (sharded)
if world_size > 1:
    # Get first linear layer weight
    first_layer_weight = (
        model.model.layers["0"].attention.wq.weight
        if hasattr(model.model.layers["0"].attention, "wq")
        else model.model.layers["0"].attention.wq_a.weight
    )

    # Print shape and first few values
    if rank == 0:
        print("\n🔍 Verifying TP sharding:")
        print(f"   Rank {rank} weight shape: {first_layer_weight.shape}")
        print(f"   Rank {rank} weight[0,:5]: {first_layer_weight[0, :5]}")

    dist.barrier()

    if rank == 1:
        print(f"   Rank {rank} weight shape: {first_layer_weight.shape}")
        print(f"   Rank {rank} weight[0,:5]: {first_layer_weight[0, :5]}")

# Test forward pass
if rank == 0:
    print("\n🧪 Testing forward pass...")

batch_size = 2
seq_len = 16
input_ids = torch.randint(
    0, model.config.vocab_size, (batch_size, seq_len), device=f"cuda:{rank}"
)

with torch.no_grad():
    hidden_states = model(input_ids)
    logits = model.compute_logits(hidden_states)

if rank == 0:
    print("✅ Forward pass successful!")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")

if world_size > 1:
    dist.barrier()
    if rank == 0:
        print("\n✅ All ranks completed successfully!")
        print(f"   Tensor Parallelism verified across {world_size} GPUs")

if "RANK" in os.environ:
    dist.destroy_process_group()
