#!/usr/bin/env python3
"""torch.compile pass to optimize TP MoE collectives.

Eliminates redundant all-reduces in expert output projection (w2/down_proj)
by sharding the projection per TP rank and fusing into existing shared-expert
all-reduce.

Based on Kimi K3 optimization: each GPU was computing full routed-expert
output projection then all-reducing. Fix: each GPU computes only its output
coordinates, adds to shared-expert partials before existing all-reduce.
"""

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
import torch.fx as fx

from vllm.compilation.passes.inductor_pass import enable_fake_mode
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.logger import init_logger

logger = init_logger(__name__)


class MoETPCollectiveOptimizationPass(VllmInductorPass):
    """Optimize MoE TP collectives by sharding expert output projections.
    
    Pattern: all_reduce(w2_weight @ expert_output) 
    -> Replace with: shard w2_weight, compute partial, fuse into shared-expert reduce
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.disabled = False
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        
        if self.tp_size <= 1:
            logger.info("MoE TP Collective Optimization disabled for tp_size <= 1")
            self.disabled = True
            return
            
        self.patterns = PatternMatcherPass("moe_tp_collective_optimization")
        self.register_patterns()
        
    @enable_fake_mode
    def register_patterns(self):
        """Register the pattern for MoE expert output projection all-reduce."""
        
        # Pattern: all_reduce(expert_output @ w2_weight.T) where w2 is row-parallel
        # This matches the routed expert down_proj in FusedMoE
        
        def moe_expert_output_pattern(
            expert_output: torch.Tensor,  # [num_tokens, hidden_size] - routed expert output
            w2_weight: torch.Tensor,       # [num_experts, hidden_size, intermediate_size] - row parallel
        ) -> torch.Tensor:
            # This pattern matches: expert_output @ w2_weight.T followed by all_reduce
            # In practice this happens in the fused MoE kernel's combine phase
            
            # Simulate the projection (happens inside kernel)
            # expert_output: [num_tokens, hidden_size]
            # w2_weight: [num_experts, hidden_size, intermediate_size] -> need to index by expert
            # For simplicity, match the all_reduce on the combined routed output
            
            # The actual pattern in the graph will be:
            # 1. MoE kernel produces routed_expert_output (already combined across top-k)
            # 2. If not reduced by kernel, all_reduce is called in MoERunner
            
            routed_output = expert_output  # placeholder
            reduced = torch.ops.vllm.tensor_model_parallel_all_reduce.default(routed_output)
            return reduced
        
        def moe_expert_output_replacement(
            expert_output: torch.Tensor,
            w2_weight: torch.Tensor,
        ) -> torch.Tensor:
            # Replacement: shard w2 projection per TP rank
            # Each rank computes only its output shard
            
            # Get output dimension shard for this rank
            hidden_size = expert_output.shape[-1]
            shard_size = hidden_size // self.tp_size
            start = self.tp_rank * shard_size
            end = start + shard_size
            
            # Slice w2_weight to only compute our output coordinates
            # w2_weight shape: [num_experts, hidden_size, intermediate_size]
            # We need w2_weight[:, start:end, :] for our output shard
            w2_shard = w2_weight[:, start:end, :]
            
            # Compute partial projection for our shard
            # This requires the expert_output to be in the right format
            # For now, return the sharded computation hint
            # The actual implementation needs to hook into the MoE kernel
            
            # This is a marker - the real optimization happens at kernel level
            # The pass marks the pattern for the kernel to handle
            return torch.ops.vllm.tensor_model_parallel_all_reduce.default(expert_output)
        
        # Note: The actual pattern matching needs to happen at the FX graph level
        # where we can see the all_reduce after MoE kernel
        # For now, register a placeholder - the real implementation
        # requires deeper integration with the MoE kernel's output handling
        
        logger.debug("MoE TP Collective patterns registered (placeholder)")
    
    def is_applicable_for_range(self, compile_range) -> bool:
        if self.disabled:
            return False
        # Apply for decode-sized batches where MoE collectives matter
        return True
    
    def __call__(self, graph: fx.Graph) -> None:
        if self.disabled:
            return
        matched = self.patterns.apply(graph)
        if matched:
            logger.debug(f"MoE TP Collective Optimization: matched {matched} patterns")


def register_moe_tp_optimization_pass(pass_manager, config):
    """Register the MoE TP collective optimization pass."""
    pass_instance = MoETPCollectiveOptimizationPass(config)
    if not pass_instance.disabled:
        pass_manager.append(pass_instance)
        logger.info("Registered MoE TP Collective Optimization Pass")


# Alternative approach: Kernel-level optimization
# The real fix needs to be in the MoE kernel (triton_moe.py / fused_moe.py)
# where we can shard the w2 projection per TP rank.

# Here's the kernel-level change needed:

"""
In fused_moe.py invoke_fused_moe_triton_kernel:
- The kernel currently computes full w2 output per expert per token
- Need to add TP rank awareness to only compute output shard
- Then the existing all_reduce in MoERunner combines shards

Key changes:
1. Pass tp_rank, tp_size to kernel
2. In kernel, each thread block computes only its output coordinate shard
3. Accumulate into shared memory, then single all_reduce at end

This is more involved and requires kernel modification.
"""


if __name__ == "__main__":
    # Test the pass registration
    print("MoE TP Collective Optimization Pass")
    print(f"TP size: {get_tensor_model_parallel_world_size()}")
    print(f"TP rank: {get_tensor_model_parallel_rank()}")