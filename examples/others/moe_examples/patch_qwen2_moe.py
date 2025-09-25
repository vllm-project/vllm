#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen2MoE Patch Module

This module applies Pick & Ban routing patches to Qwen2MoE models.
It should be imported before any vLLM model loading.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def apply_qwen2_moe_patch():
    """Apply Pick & Ban routing patch to Qwen2MoE."""
    
    try:
        from vllm.model_executor.layers.fused_moe.pick_ban_routing import (
            create_pick_and_ban_routing_function,
        )
        
        # Define key experts for Qwen1.5-MoE (example configuration)
        key_experts_per_layer = {
            i: {8, 15, 23}
            for i in range(24)  # Assuming 24 layers
        }
        
        # Create Pick & Ban routing function
        custom_routing_function = create_pick_and_ban_routing_function(
            key_experts_per_layer=key_experts_per_layer,
            lambda_threshold=0.7,
            tau_threshold=0.9,
        )
        
        # Patch Qwen2MoE
        from vllm.model_executor.models import qwen2_moe
        
        original_qwen2_moe_sparse = qwen2_moe.Qwen2MoeSparseMoeBlock
        
        class PatchedQwen2MoeSparseMoeBlock(original_qwen2_moe_sparse):
            def __init__(self, config, quant_config=None, prefix=""):
                print(
                    f"🚀🚀🚀 PATCHED Qwen2MoeSparseMoeBlock.__init__ called! "
                    f"prefix={prefix}"
                )
                print("🚀🚀🚀 Worker process patch applied!")
                
                # Initialize the module without calling parent __init__ to avoid
                # double allocation
                torch.nn.Module.__init__(self)
                
                # Set up the module attributes manually
                self.config = config
                self.quant_config = quant_config
                
                # Set up tensor parallel size (from original implementation)
                from vllm.distributed import get_tensor_model_parallel_world_size
                
                self.tp_size = get_tensor_model_parallel_world_size()
                
                if self.tp_size > config.num_experts:
                    raise ValueError(
                        f"Tensor parallel size {self.tp_size} is greater than "
                        f"the number of experts {config.num_experts}."
                    )
                
                # Create experts with custom routing directly
                print(f"🔧 Creating FusedMoE with custom_routing_function for {prefix}")
                from vllm.model_executor.layers.fused_moe import FusedMoE
                
                self.experts = FusedMoE(
                    num_experts=config.num_experts,
                    top_k=config.num_experts_per_tok,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    reduce_results=False,
                    renormalize=config.norm_topk_prob,
                    quant_config=quant_config,
                    custom_routing_function=custom_routing_function,  # Our routing!
                    prefix=f"{prefix}.experts",
                )
                
                # Create gate layer (from original implementation)
                from vllm.model_executor.layers.linear import ReplicatedLinear
                
                self.gate = ReplicatedLinear(
                    config.hidden_size,
                    config.num_experts,
                    bias=False,
                    quant_config=None,
                )
                
                # Create shared expert (from original implementation)
                if config.shared_expert_intermediate_size > 0:
                    self.shared_expert = qwen2_moe.Qwen2MoeMLP(
                        hidden_size=config.hidden_size,
                        intermediate_size=config.shared_expert_intermediate_size,
                        hidden_act=config.hidden_act,
                        quant_config=quant_config,
                        reduce_results=self.experts.must_reduce_shared_expert_outputs(),
                    )
                else:
                    self.shared_expert = None
                
                # Create shared_expert_gate (from original implementation)
                self.shared_expert_gate = torch.nn.Linear(
                    config.hidden_size, 1, bias=False
                )
                
                print(f"✅ FusedMoE created with custom routing for {prefix}")
            
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                """Forward method from original Qwen2MoeSparseMoeBlock."""
                import torch.nn.functional as F
                
                # NOTE: hidden_states can have either 1D or 2D shape.
                orig_shape = hidden_states.shape
                hidden_dim = hidden_states.shape[-1]
                hidden_states = hidden_states.view(-1, hidden_dim)
                shared_output = None
                if self.shared_expert is not None:
                    shared_output = self.shared_expert(hidden_states)
                    if self.shared_expert_gate is not None:
                        shared_output = (
                            F.sigmoid(self.shared_expert_gate(hidden_states))
                            * shared_output
                        )

                # router_logits: (num_tokens, n_experts)
                router_logits, _ = self.gate(hidden_states)
                final_hidden_states = self.experts(
                    hidden_states=hidden_states, router_logits=router_logits
                )
                if shared_output is not None:
                    final_hidden_states = final_hidden_states + shared_output
                if self.tp_size > 1:
                    final_hidden_states = (
                        self.experts.maybe_all_reduce_tensor_model_parallel(
                            final_hidden_states
                        )
                    )

                return final_hidden_states.view(orig_shape)
        
        qwen2_moe.Qwen2MoeSparseMoeBlock = PatchedQwen2MoeSparseMoeBlock
        
        print("✅ Qwen2MoE patch applied successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to apply Qwen2MoE patch: {e}")
        import traceback
        
        traceback.print_exc()
        return False


# Auto-apply patch when this module is imported
if __name__ != "__main__":
    apply_qwen2_moe_patch()